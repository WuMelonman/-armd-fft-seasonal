import math
import torch
import torch.nn.functional as F

from torch import nn
from einops import reduce
from tqdm.auto import tqdm
from functools import partial
from Models.autoregressive_diffusion.linear import Linear
from Models.autoregressive_diffusion.model_utils import default, identity, extract


# gaussian diffusion trainer class

pred_len = 96# 预测长度（未来要预测的时间步数）

def linear_beta_schedule(timesteps):
    # 线性 beta 调度：beta 从 beta_start 线性增长到 beta_end
    # beta 控制每一步加入多少噪声（噪声强度），beta 越大加噪越强
    scale = 1000 / timesteps  # 论文/代码里常见的缩放：保证不同 timesteps 下 beta 范围“相对一致”
    beta_start = scale * 0.0001  # 起始噪声强度（很小）
    beta_end = scale * 0.02      # 结束噪声强度（更大）
    # 返回长度为 timesteps 的 beta 序列，float64 是为了数值更稳定（后面会转 float32 buffer）
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
     # 余弦 beta 调度（DDPM/DDIM 常用改进）：让 alpha_cumprod 按余弦曲线衰减
    # 经验上比线性更稳定，尤其在高维数据上更常用
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


class ARMD(nn.Module):
    def __init__(
            self,
            seq_length,  # 输入序列长度（历史窗口长度）
            feature_size,  # 特征维度（变量数/通道数）
            n_layer_enc=3,  # encoder 层数（这里可能留作扩展，当前片段未使用）
            n_layer_dec=6,  # decoder 层数（同上）
            d_model=None,  # 模型隐藏维度（同上）
            timesteps=1000,  # 扩散步数 T（训练时 forward diffusion 的总步数）
            sampling_timesteps=None,  # 采样步数（DDIM/加速采样用；<= timesteps）
            loss_type='l1',  # 损失类型（L1/L2等，后面训练用）
            beta_schedule='cosine',  # beta 调度类型：linear/cosine
            n_heads=4,  # 注意力头数（当前片段未使用）
            mlp_hidden_times=4,  # MLP 扩展倍数（当前片段未使用）
            eta=0.,  # DDIM eta，控制采样随机性（0 通常更确定）
            attn_pd=0.,  # attention dropout（当前片段未使用）
            resid_pd=0.,  # residual dropout（当前片段未使用）
            w_grad=True,  # 是否使用梯度相关机制（与你的 Linear 模块有关）
            **kwargs
    ):
        super(ARMD, self).__init__()

        self.eta = eta
        self.seq_length = seq_length
        self.pred_len = seq_length  # 供 ARMDTrendWrapper 对齐用
        self.feature_size = feature_size
        # n_feat=feature_size：每个时间点的特征数
        # n_channel=seq_length：序列长度当作 channel/维度（作者的实现习惯）
        self.model = Linear(n_feat=feature_size, n_channel=seq_length, w_grad=w_grad, **kwargs)

        if beta_schedule == 'linear':
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')

        alphas = 1. - betas # alpha_t = 1 - beta_t
        alphas_cumprod = torch.cumprod(alphas, dim=0) # \bar{alpha}_t
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)# pad 在最前面补 1（对应 t=0 时 \bar{alpha}_{-1} 视为 1）

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)# 存储总扩散步数 T
        self.loss_type = loss_type# 训练时用的损失类型

        # sampling related parameters
        # -------- sampling 相关参数 --------
        # sampling_timesteps: 采样时走多少步（用于加速；比如训练 T=1000，采样只走 50/100）
        # default(...)：如果 sampling_timesteps=None，就用 timesteps（即不加速）
        self.sampling_timesteps = default(
            sampling_timesteps, timesteps)  # default num sampling timesteps to number of timesteps at training

        assert self.sampling_timesteps <= timesteps# 采样步数不能超过训练步数
        self.fast_sampling = self.sampling_timesteps < timesteps# True 表示使用快速采样（如 DDIM）

        # helper function: register_buffer
        # register_buffer 的意义：把常量张量挂到模型上（会跟着 .to(device) / .cuda() / 保存加载）
        # 并且不参与梯度更新（不是 Parameter）
        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)# \bar{alpha}_{t}
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)# \bar{alpha}_{t-1}

        # calculations for diffusion q(x_t | x_{t-1}) and others
        # -------- diffusion 前向 q(x_t | x_{t-1}) 常用的预计算项 --------
        # x_t = sqrt(\bar{alpha}_t) * x0 + sqrt(1-\bar{alpha}_t) * noise
        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))#论文中常用常数公式

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # 上述公式是 q(x_{t-1} | x_t, x_0)的方差

        register_buffer('posterior_variance', posterior_variance)

        #posterior_variance 在 t=0 处可能非常小甚至 0，取 log 会 -inf，所以 clamp
        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min=1e-20)))
        # 后验均值的系数：
        # posterior_mean = coef1 * x0 + coef2 * x_t
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # -------- loss reweighting（训练加权）--------
        # 这里是对不同 t 的损失做重加权，避免某些步对训练贡献过大/过小
        # 形式与一些扩散实现中的 SNR/权重设计相关（作者自己的设计）
        register_buffer('loss_weight', torch.sqrt(alphas) * torch.sqrt(1. - alphas_cumprod) / betas / 100)

    def predict_noise_from_start(self, x_t, t, x0):
        # 根据当前时刻的带噪样本 x_t 和预测/已知的 x0
        # 反推出在第 t 步加入的噪声 ε
        #
        # 推导自前向扩散公式：
        #   x_t = sqrt(ᾱ_t) * x0 + sqrt(1 - ᾱ_t) * ε
        # =>
        #   ε = (sqrt(1 / ᾱ_t) * x_t - x0) / sqrt(1 / ᾱ_t - 1)
        #
        # sqrt_recip_alphas_cumprod      = sqrt(1 / ᾱ_t)
        # sqrt_recipm1_alphas_cumprod    = sqrt(1 / ᾱ_t - 1)
        return (
                (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) /
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )
    
    def predict_start_from_noise(self, x_t, t, noise):
        # 根据当前带噪样本 x_t 和模型预测的噪声 ε
        # 反推出原始干净样本 x0
        #
        # 同样来自前向扩散公式变形：
        #   x0 = sqrt(1 / ᾱ_t) * x_t - sqrt(1 / ᾱ_t - 1) * ε
        #
        # 这是 diffusion 反向过程中最常用的一个变换
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        # 计算后验分布 q(x_{t-1} | x_t, x_0) 的参数
        #
        # 在 DDPM 中，该后验是一个高斯分布：
        #   q(x_{t-1} | x_t, x_0) = N(posterior_mean, posterior_variance)
        #
        # posterior_mean 是 x_start 和 x_t 的线性组合
        posterior_mean = (
                extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        # 后验方差 Var[q(x_{t-1} | x_t, x0)]
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)# 后验方差的 log 形式（用于数值稳定和采样）
        return posterior_mean, posterior_variance, posterior_log_variance_clipped
    
    def output(self, x, t, training=False):
        model_output = self.model(x, t, training=training)
        return model_output
        # 将当前带噪样本 x 和时间步 t 输入到去噪网络中
        # model 的输出含义取决于作者设计：
        #   - 在 ARMD 中，这里输出的是 x_start（x0 的预测）

    def model_predictions(self, x, t, clip_x_start=False, training=False):
        # 统一封装模型预测逻辑：
        # 输入 x_t 和 t
        # 输出：
        #   - pred_noise: 预测的噪声 ε
        #   - x_start:   预测的原始样本 x0
        #论文公式(6)
        if training:
            training = False
        maybe_clip = partial(torch.clamp, min=-2, max=2) if clip_x_start else identity
        x_start = self.output(x, t, training)
        #x_start = maybe_clip(x_start)
        pred_noise = self.predict_noise_from_start(x, t, x_start)
        return pred_noise, x_start

    def p_mean_variance(self, x, t, clip_denoised=True):
        # 反向扩散一步中，计算 p(x_{t-1} | x_t) 的参数
        #
        # 1. 通过模型预测 x_start
        # 2. 用 q_posterior 构造高斯分布参数
        _, x_start = self.model_predictions(x, t)
        # 对预测的 x_start 做裁剪，限制在合理范围（[-1, 1]）
        if clip_denoised:
            x_start.clamp_(-1., 1.)
        # 计算反向一步所需的均值和方差
        model_mean, posterior_variance, posterior_log_variance = \
            self.q_posterior(x_start=x_start, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    def p_sample(self, x, t: int, clip_denoised=True):
        # 执行一次反向采样：x_t -> x_{t-1}

        # 将标量 t 扩展成 batch 维度，方便 extract 使用
        batched_times = torch.full((x.shape[0],), t, device=x.device, dtype=torch.long)
        model_mean, _, model_log_variance, x_start = \
            self.p_mean_variance(x=x, t=batched_times, clip_denoised=clip_denoised)
        # 当 t > 0 时，加入随机噪声（采样）
        # 当 t == 0 时，不再加噪声，直接输出最终结果
        noise = torch.randn_like(x) if t > 0 else 0.
        #均值＋方差采样
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start

    @torch.no_grad()
    def sample(self, x):
        # 标准采样流程（完整 DDPM 反向扩散）
        # 从 t = T-1 一直迭代到 t = 0
        device = self.betas.device
        shape = x.shape
        # 取输入序列的前 pred_len 作为初始“噪声序列”
        # 注意：这里不是纯随机噪声，而是条件生成（基于历史）
        img = x[:, :pred_len, :]

        for t in tqdm(reversed(range(0, self.num_timesteps)),
                      desc='sampling loop time step', total=self.num_timesteps):
            # 每一步执行一次反向扩散：x_t -> x_{t-1}
            img, _ = self.p_sample(img, t)
        return img

    @torch.no_grad()
    def fast_sample(self, x, clip_denoised=True):
        # 快速采样（DDIM 风格）
        # 与 sample() 的区别：
        #   - 不走完整 T 步
        #   - 使用确定性/半确定性更新公式
        shape = x.shape
        batch, device, total_timesteps, sampling_timesteps, eta = \
            shape[0], self.betas.device, self.num_timesteps, self.sampling_timesteps, self.eta

        # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)

        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))  # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]
        img = x[:, :pred_len, :]

        for time, time_next in tqdm(time_pairs, desc='sampling loop time step'):
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            pred_noise, x_start, *_ = self.model_predictions(img, time_cond, clip_x_start=clip_denoised)
            if time_next < 0:
                img = x_start
                continue
            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]
            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            sigma = 0
            c = (1 - alpha_next - sigma ** 2).sqrt()
            noise = 0
            img = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise

        return img

    def generate_mts(self, x):
        # 根据是否启用 fast_sampling，选择采样策略
        sample_fn = self.fast_sample if self.fast_sampling else self.sample
        return sample_fn(x)

    @property
    def loss_fn(self):
        # 根据 loss_type 返回对应的损失函数
        if self.loss_type == 'l1':
            return F.l1_loss
        elif self.loss_type == 'l2':
            return F.mse_loss
        else:
            raise ValueError(f'invalid loss type {self.loss_type}')

    def q_sample(self, x_start, t, noise=None):
        #滑动窗口
        index = int(t[0])+1
        x_middle = x_start[:,pred_len-index:-index,:]
        return x_middle

    def _train_loss(self, x_start, t, target=None, noise=None, training=True):
        # 训练阶段的 loss 计算函数
        # x_start: 输入的完整序列（通常是 history + future 拼接）
        # t: diffusion step（batch 中所有样本相同）
        # target / noise: 可选外部传入，通常为 None

        # 如果没有提供 noise，则生成与 x_start 同形状的标准高斯噪声
        noise = default(noise, lambda: torch.randn_like(x_start))
        # 仅当未传入 target 时，使用 x_start 的未来段作为 target；若外部已传 target 则不覆盖
        if target is None:
            target = x_start[:, pred_len:, :]
        x = self.q_sample(x_start=x_start, t=t, noise=noise)  # noise sample
        # 将中间态 x 和时间步 t 输入模型
        # model_out 是模型预测的 x_start（即 x0 的预测）
        model_out = self.output(x, t, training)
        # 取当前时间步 t 对应的 sqrt(alpha_bar_t)
        alpha = self.sqrt_alphas_cumprod[t[0]]

        # 取 sqrt(1 - alpha_bar_t)
        minus_alpha = self.sqrt_one_minus_alphas_cumprod[t[0]]

        # 根据 diffusion 的解析公式，反推出“真实噪声”
        # 这里的 target 作为 x0
        target_noise = (x - target * alpha) / minus_alpha
        # 根据模型预测的 x0（model_out），反推出“预测噪声”
        pred_noise = (x - model_out * alpha) / minus_alpha

        # 计算噪声层面的损失（L1 或 L2）
        train_loss = self.loss_fn(pred_noise, target_noise, reduction='none')

        # 将除 batch 外的维度全部展平并取 mean
        train_loss = reduce(train_loss, 'b ... -> b (...)', 'mean')

        # 对不同 t 的 loss 进行加权（diffusion reweighting）
        train_loss = train_loss * extract(self.loss_weight, t, train_loss.shape)

        # 返回 batch 维度上的平均 loss
        return train_loss.mean()

    def forward(self, x, **kwargs):
        # 前向接口（供 PyTorch 训练调用）
        # x.shape 通常是 [batch, time, feature]
        b, c, n, device, feature_size, = *x.shape, x.device, self.feature_size
        # 校验特征维度是否一致
        assert n == feature_size, f'number of variable must be {feature_size}'
        # 为整个 batch 采样一个 diffusion step t（所有样本相同）
        t = torch.randint(0, self.num_timesteps, (1,), device=device).repeat(b).long()

        return self._train_loss(x_start=x, t=t, **kwargs)

    def langevin_fn(
        self,
        coef,
        partial_mask,
        tgt_embs,
        learning_rate,
        sample,
        mean,
        sigma,
        t,
        coef_=0.
    ):
    
        if t[0].item() < self.num_timesteps * 0.05:
            K = 0
        elif t[0].item() > self.num_timesteps * 0.9:
            K = 3
        elif t[0].item() > self.num_timesteps * 0.75:
            K = 2
            learning_rate = learning_rate * 0.5
        else:
            K = 1
            learning_rate = learning_rate * 0.25

        input_embs_param = torch.nn.Parameter(sample)

        with torch.enable_grad():
            for i in range(K):
                optimizer = torch.optim.Adagrad([input_embs_param], lr=learning_rate)
                optimizer.zero_grad()

                x_start = self.output(x=input_embs_param, t=t)

                if sigma.mean() == 0:
                    logp_term = coef * ((mean - input_embs_param) ** 2 / 1.).mean(dim=0).sum()
                    infill_loss = (x_start[partial_mask] - tgt_embs[partial_mask]) ** 2
                    infill_loss = infill_loss.mean(dim=0).sum()
                else:
                    logp_term = coef * ((mean - input_embs_param)**2 / sigma).mean(dim=0).sum()
                    infill_loss = (x_start[partial_mask] - tgt_embs[partial_mask]) ** 2
                    infill_loss = (infill_loss/sigma.mean()).mean(dim=0).sum()
            
                loss = logp_term + infill_loss
                loss.backward()
                optimizer.step()
                epsilon = torch.randn_like(input_embs_param.data)
                input_embs_param = torch.nn.Parameter((input_embs_param.data + coef_ * sigma.mean().item() * epsilon).detach())

        sample[~partial_mask] = input_embs_param.data[~partial_mask]
        return sample

if __name__ == '__main__':
    pass
