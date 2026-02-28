import os
import sys
import time
import torch
import numpy as np

from pathlib import Path
from tqdm.auto import tqdm
from ema_pytorch import EMA
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_
from Utils.io_utils import instantiate_from_config, get_model_parameters_info


sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

def cycle(dl):
    """
       将 dataloader 变成一个无限数据流
       每次 next() 返回一个 batch，跑完一轮 dataloader 后自动重新开始
    """
    while True:
        for data in dl:
            yield data
#返回一个值
#冻结函数执行状态
#下次 next() 从冻结处继续

class Trainer(object):
    def __init__(self, config, args, model, dataloader, logger=None):
        super().__init__()
        self.model = model
        self.device = self.model.betas.device  # 模型参数所在设备（GPU / CPU）
        self.train_num_steps = config['solver']['max_epochs'] # 训练的总 step 数（不是 epoch）
        self.gradient_accumulate_every = config['solver']['gradient_accumulate_every'] # 梯度累积的步数
        self.save_cycle = config['solver']['save_cycle']# 每隔多少 step 保存一次模型
        self.dl = cycle(dataloader['dataloader'])# 无限 dataloader
        self.step = 0
        self.milestone = 0
        self.args = args
        self.logger = logger

        self.results_folder = Path(config['solver']['results_folder'] + f'_{model.seq_length}')# 存 checkpoint 的目录
        os.makedirs(self.results_folder, exist_ok=True)

        start_lr = config['solver'].get('base_lr', 1.0e-4)# 初始学习率
        ema_decay = config['solver']['ema']['decay'] # EMA 参数
        ema_update_every = config['solver']['ema']['update_interval'] # EMA 参数

        self.opt = Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=start_lr, betas=[0.9, 0.96])
        self.ema = EMA(self.model, beta=ema_decay, update_every=ema_update_every).to(self.device)# EMA 模型（用于采样）

        sc_cfg = config['solver']['scheduler']# 学习率调度器
        sc_cfg['params']['optimizer'] = self.opt
        self.sch = instantiate_from_config(sc_cfg)

        if self.logger is not None:# 打印模型参数量
            self.logger.log_info(str(get_model_parameters_info(self.model)))
        self.log_frequency = 100

    def save(self, milestone, verbose=False):
        """保存当前模型参数、EMA 和优化器状态"""
        if self.logger is not None and verbose:
            self.logger.log_info('Save current model to {}'.format(str(self.results_folder / f'checkpoint-{milestone}.pt')))
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'ema': self.ema.state_dict(),
            'opt': self.opt.state_dict(),
        }
        torch.save(data, str(self.results_folder / f'checkpoint-{milestone}.pt'))

    def load(self, milestone, verbose=False):
        """加载某个 checkpoint"""
        if self.logger is not None and verbose:
            self.logger.log_info('Resume from {}'.format(str(self.results_folder / f'checkpoint-{milestone}.pt')))
        device = self.device
        data = torch.load(str(self.results_folder / f'checkpoint-{milestone}.pt'), map_location=device)
        self.model.load_state_dict(data['model'])
        self.step = data['step']
        self.opt.load_state_dict(data['opt'])
        self.ema.load_state_dict(data['ema'])
        self.milestone = milestone

    def train(self):
        """ARMD 扩散模型的训练循环"""
        device = self.device
        step = 0
        if self.logger is not None:
            tic = time.time()
            self.logger.log_info('{}: start training...'.format(self.args.name), check_primary=False)

        with tqdm(initial=step, total=self.train_num_steps) as pbar:#tqdm 是一个 进度条库，用来把 for / while 循环的运行进度实时显示在终端。
            while step < self.train_num_steps:
                total_loss = 0.
                for _ in range(self.gradient_accumulate_every):
                    # 取一个 batch (B, T, N)
                    # 调用 ARMD.forward → diffusion loss
                    data = next(self.dl).to(device)
                    loss = self.model(data, target=data)
                    loss = loss / self.gradient_accumulate_every # 均分到多个 accumulation step
                    loss.backward()
                    total_loss += loss.item()

                pbar.set_description(f'loss: {total_loss:.6f}')

                clip_grad_norm_(self.model.parameters(), 1.0)# 防止梯度爆炸
                # 更新模型参数
                self.opt.step()
                # 更新学习率
                self.sch.step(total_loss)
                # 清空梯度
                self.opt.zero_grad()
                self.step += 1
                step += 1
                # 更新 EMA 模型
                self.ema.update()
                # 保存模型
                with torch.no_grad():
                    if self.step != 0 and self.step % self.save_cycle == 0:
                        self.milestone += 1
                        self.save(self.milestone)
                        # self.logger.log_info('saved in {}'.format(str(self.results_folder / f'checkpoint-{self.milestone}.pt')))
                    
                    if self.logger is not None and self.step % self.log_frequency == 0:
                        # info = '{}: train'.format(self.args.name)
                        # info = info + ': Epoch {}/{}'.format(self.step, self.train_num_steps)
                        # info += ' ||'
                        # info += '' if loss_f == 'none' else ' Fourier Loss: {:.4f}'.format(loss_f.item())
                        # info += '' if loss_r == 'none' else ' Reglarization: {:.4f}'.format(loss_r.item())
                        # info += ' | Total Loss: {:.6f}'.format(total_loss)
                        # self.logger.log_info(info)
                        self.logger.add_scalar(tag='train/loss', scalar_value=total_loss, global_step=self.step) # 记录 loss

                pbar.update(1)

        print('training complete')
        if self.logger is not None:
            self.logger.log_info('Training done, time: {:.2f}'.format(time.time() - tic))

    def sample(self, num, size_every, shape=None):
        """
            用 EMA 模型生成无条件时间序列样本
        """
        if self.logger is not None:
            tic = time.time()
            self.logger.log_info('Begin to sample...')
        samples = np.empty([0, shape[0], shape[1]])
        #print(samples.shape)
        num_cycle = int(num // size_every) + 1
        # 用 EMA 模型采样
        for _ in range(num_cycle):
            sample = self.ema.ema_model.generate_mts(batch_size=size_every)
            #print(sample.shape)
            samples = np.row_stack([samples, sample.detach().cpu().numpy()])
            torch.cuda.empty_cache()

        if self.logger is not None:
            self.logger.log_info('Sampling done, time: {:.2f}'.format(time.time() - tic))
        return samples

    def sample_forecast(self, raw_dataloader, shape=None): # shape = (pred_len, feature_dim)
        """
            这个函数的目的：
            - 对 raw_dataloader 里的每一个 batch（批次）做一次“预测未来”
            - 返回两样东西：
                1) samples: 模型预测出来的未来序列（numpy）
                2) reals: 真实的未来序列（numpy），用来做评估对比（比如 MSE/MAE）
            这里的“条件预测”指：模型不是从纯噪声开始生成，而是“看着输入 x（历史）去预测未来”。
        """
        if self.logger is not None:
            tic = time.time()
            self.logger.log_info('Begin to sample...')
        # samples: 用来装【所有 batch 的预测结果】
        # np.empty([0, pred_len, feature_dim]) 表示：
        # reals: 用来装【所有 batch 的真实未来】
        samples = np.empty([0, shape[0], shape[1]])
        reals = np.empty([0, shape[0], shape[1]])

        for idx, batch in enumerate(raw_dataloader):
            # 兼容两种 dataloader 输出格式：
            # 1) batch = x
            # 2) batch = (x, t_m)
            #
            # t_m 可能是 time_mark（时间特征，如小时/星期/月份编码），
            # 但你这里实际上没用到它，只是接收一下避免报错。
            if len(batch)==2:
                x, t_m = batch
                x, t_m = x.to(self.device), t_m.to(self.device)
            else:
                x = batch
                x = x.to(self.device)
            # generate_mts(x) 表示“条件生成/预测”：
            # 模型会根据输入 x（里面包含历史部分）去生成未来部分
            #
            # sample 的形状通常是：
            # sample: (B, pred_len, N)
            sample = self.ema.ema_model.generate_mts(x)
            # 把 sample 从 torch Tensor 变成 numpy，拼到 samples 里
            #
            # sample.detach(): 从计算图里“切断”，表示不需要梯度（推理阶段）
            # .cpu(): 从 GPU 搬回 CPU（numpy 只能在 CPU 上）
            # .numpy(): 转成 numpy 数组
            #
            # np.row_stack: 沿着第 0 维拼接
            # 也就是把当前 batch 的预测结果接到 samples 后面
            samples = np.row_stack([samples, sample.detach().cpu().numpy()])
            #reals = None
            reals = np.row_stack([reals, x[:,shape[0]:,:].detach().cpu().numpy()])
            # 清空 GPU 缓存（不是清空显存占用，只是释放缓存块给 PyTorch allocator）
            # 在循环推理时有时能减少显存碎片，避免 OOM
            torch.cuda.empty_cache()

        if self.logger is not None:
            self.logger.log_info('Sampling done, time: {:.2f}'.format(time.time() - tic))
        return samples, reals

