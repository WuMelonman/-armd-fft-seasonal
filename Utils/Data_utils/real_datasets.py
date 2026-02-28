import os
import torch
import numpy as np
import pandas as pd

from scipy import io
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
from Models.autoregressive_diffusion.model_utils import normalize_to_neg_one_to_one, unnormalize_to_zero_to_one
from Utils.masking_utils import noise_mask
from statsmodels.tsa.seasonal import STL


class CustomDataset(Dataset):
    def __init__(
        self, 
        name,# 数据集名称（如 'etth'）
        data_root, # CSV 文件路径
        window=64, # 滑动窗口长度（一个样本包含多少时间步）
        proportion=0.8,  # 训练集比例
        save2npy=True, # 是否保存 ground truth 到 npy
        neg_one_to_one=True,# 是否归一化到 [-1,1]（本实现中未启用）
        seed=123,# 随机种子（用于数据划分、mask）
        period='train',# 当前 Dataset 类型：'train' 或 'test'
        output_dir='./OUTPUT',# 保存 npy 文件的根目录
        predict_length=None,# 预测任务：未来预测步长
        missing_ratio=None,# 插补任务：缺失比例
        style='separate', # mask 生成方式
        distribution='geometric', # mask 分布
        mean_mask_length=3,# 平均缺失长度

        # ===== ADD: decomposition switches =====
        decompose = False,  # 是否做季节-趋势分解
        decompose_method = 'stl',  # 'stl'（先只实现这个）
        seasonal_period = 24,  # ETTh1 小时数据：24（天周期）
        predict_component = 'raw',  # 'raw' 或 'trend'
        return_aux_in_test = True  # test阶段是否在dataset里保留 season/raw future 供外部取用
    ):

        super(CustomDataset, self).__init__()
        # 限制 period 只能是 train 或 test
        assert period in ['train', 'test'], 'period must be train or test.'
        # 训练集不能设置预测长度或缺失比例
        if period == 'train':
            assert ~(predict_length is not None or missing_ratio is not None), ''
        # 保存基础属性
        self.name, self.pred_len, self.missing_ratio = name, predict_length, missing_ratio
        self.style, self.distribution, self.mean_mask_length = style, distribution, mean_mask_length


        # ===== ADD =====
        self.decompose = decompose
        self.decompose_method = decompose_method
        self.seasonal_period = seasonal_period
        self.predict_component = predict_component
        self.return_aux_in_test = return_aux_in_test

        # 读取 CSV 数据并拟合 scaler
        # rawdata: 原始时间序列 [T, C]
        # scaler : StandardScaler（在全量数据上 fit）
        self.rawdata, self.scaler = self.read_data(data_root, self.name)
        # 保存样本和 mask 的目录
        self.dir = os.path.join(output_dir, 'samples')
        os.makedirs(self.dir, exist_ok=True)

        self.window, self.period = window, period # window: 窗口长度  # period: train / test
        self.len, self.var_num = self.rawdata.shape[0], self.rawdata.shape[-1] # 序列长度 T、变量数 C
        self.sample_num_total = max(self.len - self.window + 1, 0)# 理论上可生成的窗口总数 N = T - window + 1
        self.save2npy = save2npy
        #self.auto_norm = neg_one_to_one
        self.auto_norm = False

        self.data = self.__normalize(self.rawdata)
        # 对整条时间序列做标准化
        # rawdata [T,C] -> data [T,C]

        # ===== ADD: 如果只预测 trend，就把“滑窗数据源”换成 trend_scaled_full =====
        if self.decompose and self.predict_component == 'trend':
            trend_raw, season_raw = self._decompose_full_series(self.rawdata)
            deseason_raw = self.rawdata - season_raw  # raw - season = trend + resid
            deseason_scaled_full = self.__normalize(deseason_raw)
            train, inference = self.__getsamples(deseason_scaled_full, proportion, seed)
            self.samples = train if period == 'train' else inference

            # 只在 test 阶段准备 season/raw，用于加回与算指标
            if period == 'test' and self.return_aux_in_test:
                season_windows_raw = self._make_windows(season_raw)
                season_train, season_test = self.divide(season_windows_raw, proportion, seed)
                self.season_samples_raw = season_test

                raw_windows_raw = self._make_windows(self.rawdata)
                raw_train, raw_test = self.divide(raw_windows_raw, proportion, seed)
                self.raw_samples_raw = raw_test

        else:
            # 原逻辑：用标准化后的 self.data 生成 samples
            train, inference = self.__getsamples(self.data, proportion, seed)
            self.samples = train if period == 'train' else inference
        #train, inference = self.__getsamples(self.data, proportion, seed)
        # 将标准化后的序列滑窗并划分 train / test
        # train, inference 的 shape 都是 [N, window, C]
        #self.samples = train if period == 'train' else inference # 根据 period 决定当前 Dataset 使用哪一部分


        # 只有 test 阶段才会构造 mask
        if period == 'test':
            if missing_ratio is not None:
                self.masking = self.mask_data(seed)
            # 预测任务：将窗口最后 predict_length 步 mask 掉
            elif predict_length is not None:
                masks = np.ones(self.samples.shape)
                masks[:, -predict_length:, :] = 0
                self.masking = masks.astype(bool)
            else:
                raise NotImplementedError()
        self.sample_num = self.samples.shape[0]# 当前 Dataset 实际样本数
    # ===== ADD =====
    def _decompose_full_series(self, rawdata_2d):
        """
        输入 rawdata_2d: [T, C] 原尺度
        输出:
          trend_raw:  [T, C]
          season_raw: [T, C]
        """
        if self.decompose_method != 'stl':
            raise ValueError(f"Unknown decompose_method: {self.decompose_method}")

        T, C = rawdata_2d.shape
        trend = np.zeros_like(rawdata_2d)
        season = np.zeros_like(rawdata_2d)

        for c in range(C):
            res = STL(rawdata_2d[:, c], period=self.seasonal_period, robust=True).fit()
            trend[:, c] = res.trend
            season[:, c] = res.seasonal

        return trend, season

        # ===== ADD =====


    def _make_windows(self, data_full):
        """
        把整段 [T, C] 变成窗口 [N, window, C]，不做 divide，不保存文件
        """
        x = np.zeros((self.sample_num_total, self.window, self.var_num))
        for i in range(self.sample_num_total):
            x[i, :, :] = data_full[i:i + self.window, :]
        return x
    def __getsamples(self, data, proportion, seed):
        """
            将完整序列 data [T,C] 切分为滑动窗口样本
            并按比例划分 train / test
        """

        # 初始化窗口样本数组
        # shape: [N, window, C]
        x = np.zeros((self.sample_num_total, self.window, self.var_num))
        for i in range(self.sample_num_total):
            start = i
            end = i + self.window
            x[i, :, :] = data[start:end, :]
        # 顺序划分 train / test（不打乱）
        train_data, test_data = self.divide(x, proportion, seed)

        if self.save2npy:
            if 1 - proportion > 0:
                np.save(os.path.join(self.dir, f"{self.name}_ground_truth_{self.window}_test.npy"), self.unnormalize(test_data))
            np.save(os.path.join(self.dir, f"{self.name}_ground_truth_{self.window}_train.npy"), self.unnormalize(train_data))
            if self.auto_norm:
                if 1 - proportion > 0:
                    np.save(os.path.join(self.dir, f"{self.name}_norm_truth_{self.window}_test.npy"), unnormalize_to_zero_to_one(test_data))
                np.save(os.path.join(self.dir, f"{self.name}_norm_truth_{self.window}_train.npy"), unnormalize_to_zero_to_one(train_data))
            else:
                if 1 - proportion > 0:
                    np.save(os.path.join(self.dir, f"{self.name}_norm_truth_{self.window}_test.npy"), test_data)
                np.save(os.path.join(self.dir, f"{self.name}_norm_truth_{self.window}_train.npy"), train_data)

        return train_data, test_data

    def normalize(self, sq):
        # 将窗口样本展平后做标准化，再恢复形状
        d = sq.reshape(-1, self.var_num)
        d = self.scaler.transform(d)
        if self.auto_norm:
            d = normalize_to_neg_one_to_one(d)
        return d.reshape(-1, self.window, self.var_num)

    def unnormalize(self, sq):
        # 将窗口样本反标准化
        d = self.__unnormalize(sq.reshape(-1, self.var_num))
        return d.reshape(-1, self.window, self.var_num)
    
    def __normalize(self, rawdata):
        # 对整条序列做 StandardScaler 标准化
        data = self.scaler.transform(rawdata)
        if self.auto_norm:
            data = normalize_to_neg_one_to_one(data)
        return data

    def __unnormalize(self, data): # 反标准化
        if self.auto_norm:
            data = unnormalize_to_zero_to_one(data)
        x = data
        return self.scaler.inverse_transform(x)
    
    @staticmethod
    def divide(data, ratio, seed=2023):
        """
            按比例顺序划分 train / test
            不进行随机打乱
        """
        size = data.shape[0]
        # # 记录随机状态（虽然这里并未打乱）
        st0 = np.random.get_state()
        np.random.seed(seed)

        regular_train_num = int(np.ceil(size * ratio))# 训练样本数量
        #id_rdm = np.random.permutation(size)
        id_rdm = np.arange(size) # 顺序索引

        # 划分索引
        regular_train_id = id_rdm[:regular_train_num]
        irregular_train_id = id_rdm[regular_train_num:]
        # 切片得到 train / test
        regular_data = data[regular_train_id, :]
        irregular_data = data[irregular_train_id, :]

        # Restore RNG.
        np.random.set_state(st0)
        return regular_data, irregular_data

    @staticmethod
    def read_data(filepath, name=''):
        """
        从 CSV 读取时间序列数据
        返回：
        - data   : 原始序列 [T, C]
        - scaler : StandardScaler（已 fit）
        """
        df = pd.read_csv(filepath, header=0)
        # ETTh 数据第一列是时间戳，删除
        if name == 'etth':
            df.drop(df.columns[0], axis=1, inplace=True)
        data = df.values
        #scaler = MinMaxScaler()
        # 标准化器在全量数据上拟合
        scaler = StandardScaler()
        scaler = scaler.fit(data)
        return data, scaler
    
    def mask_data(self, seed=2023):
        masks = np.ones_like(self.samples)
        # Store the state of the RNG to restore later.
        st0 = np.random.get_state()
        np.random.seed(seed)

        for idx in range(self.samples.shape[0]):
            x = self.samples[idx, :, :]  # (seq_length, feat_dim) array
            mask = noise_mask(x, self.missing_ratio, self.mean_mask_length, self.style,
                              self.distribution)  # (seq_length, feat_dim) boolean array
            masks[idx, :, :] = mask

        if self.save2npy:
            np.save(os.path.join(self.dir, f"{self.name}_masking_{self.window}.npy"), masks)

        # Restore RNG.
        np.random.set_state(st0)
        return masks.astype(bool)

    def __getitem__(self, ind):
        if self.period == 'test':
            x = self.samples[ind, :, :]  # (seq_length, feat_dim) array
            m = self.masking[ind, :, :]  # (seq_length, feat_dim) boolean array
            return torch.from_numpy(x).float(), torch.from_numpy(m)
        x = self.samples[ind, :, :]  # (seq_length, feat_dim) array
        return torch.from_numpy(x).float()

    def __len__(self):
        return self.sample_num
    

class fMRIDataset(CustomDataset):
    def __init__(
        self, 
        proportion=1., 
        **kwargs
    ):
        super().__init__(proportion=proportion, **kwargs)

    @staticmethod
    def read_data(filepath, name=''):
        """Reads a single .csv
        """
        data = io.loadmat(filepath + '/sim4.mat')['ts']
        scaler = MinMaxScaler()
        scaler = scaler.fit(data)
        return data, scaler
