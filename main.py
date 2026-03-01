"""
ARMD: Auto-Regressive Moving Diffusion Models for Time Series Forecasting

Entry point for training + forecasting.

Usage:
  python3 main.py --config_path ./Config/etth1.yaml
"""

import os
import torch
import numpy as np
import random
import argparse
import matplotlib.pyplot as plt

import warnings

from trend_utils.armd_trend_wrapper import ARMDTrendWrapper

warnings.filterwarnings("ignore")

from engine.solver import Trainer
from trend_utils.armd_trend_wrapper import ARMDTrendWrapper
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from torch.utils.data import Dataset, DataLoader
from gluonts.dataset.repository.datasets import get_dataset
from gluonts.dataset.multivariate_grouper import MultivariateGrouper
from Utils.io_utils import load_yaml_config, instantiate_from_config


"""加载YAML配置文件

        Args:
            path: YAML文件路径

        Returns:
            config: 解析后的配置字典
"""
from Models.autoregressive_diffusion.model_utils import normalize_to_neg_one_to_one, unnormalize_to_zero_to_one
from Data.build_dataloader import build_dataloader, build_dataloader_cond

###################################################
# 固定随机种子：使实验可以复现（结果更稳定）
###################################################
def set_seed(seed):
    """
    Set the random seed for reproducibility.
    设定随机种子，确保多次运行的实验结果一致
    """
    random.seed(seed)# Python 随机
    np.random.seed(seed)# NumPy 随机
    
    # Set the seed for PyTorch
    torch.manual_seed(seed)# PyTorch CPU 随机
    torch.cuda.manual_seed_all(seed)# PyTorch GPU 随机
    # CuDNN 相关配置，确保严格可复现
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Python 内部 hash 随机性
    os.environ['PYTHONHASHSEED'] = str(seed)


# 设置固定种子
set_seed(2025)

#class Args_Example:
#    def __init__(self) -> None:
#        self.config_path = './Config/etth.yaml'
#        self.save_dir = './forecasting_exp'
#        self.gpu = 0
#        os.makedirs(self.save_dir, exist_ok=True)

###################################################
# Args 类：封装命令行参数，用于统一管理路径、GPU等
###################################################
class Args_Example:
    def __init__(self, config_path, save_dir, gpu):
        # 配置文件路径
        self.config_path = config_path
        # 实验输出目录（模型、日志）
        self.save_dir = save_dir
        # GPU ID
        self.gpu = gpu
        # 不存在则创建目录
        os.makedirs(self.save_dir, exist_ok=True)

###################################################
# 命令行参数解析，例如：
# python run.py --config_path ./Config/etth.yaml --gpu 0
###################################################
def parse_arguments():
    parser = argparse.ArgumentParser(description="Process configuration and directories.")
    parser.add_argument('--config_path', type=str, required=True,
                        help='Path to the configuration file.')
    parser.add_argument('--save_dir', type=str, default='./forecasting_exp',
                        help='Directory to save experiment results.')
    parser.add_argument('--gpu', type=int, default=0,
                        help='Specify which GPU to use.')
    
    args = parser.parse_args()
    # 返回解析后的参数对象
    return args

###################################################
# 主流程入口
###################################################
if __name__ == "__main__":
    #args =  Args_Example()
    # 从命令行获取参数
    args_parsed = parse_arguments()#Namespace(config_path='./Config/etth1.yaml', gpu=0, save_dir='./forecasting_exp')
    # 包装到 Args_Example 类（方便后续使用）
    args = Args_Example(args_parsed.config_path, args_parsed.save_dir, args_parsed.gpu)
    #print(args),地址
    seq_len = 96# One-shot 预测长度
    configs = load_yaml_config(args.config_path) # 读取 YAML 配置（包含模型结构、超参数、数据设置等）
    #print(configs)

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')# 选择 GPU 或 CPU
    ###################################################
    # 根据配置文件自动创建模型 (instantiate_from_config)
    ###################################################
    armd = instantiate_from_config(configs['model']).to(device)

    feature_size = configs['model']['params']['feature_size']

    model = ARMDTrendWrapper(armd=armd, feature_size=feature_size).to(device)
    #configs['solver']['max_epochs']=100
    ###################################################
    # 构建训练集 dataloader
    ###################################################
    dataloader_info = build_dataloader(configs, args) #print(dataloader_info) 地址
    dataloader = dataloader_info['dataloader']
    ###################################################

    trainer = Trainer(config=configs, args=args, model=model, dataloader={'dataloader':dataloader})# 初始化 Trainer（包含优化器、损失函数、训练流程等）
    trainer.train() # 训练模型
    ###################################################
    # -----------   推理（预测）阶段   ----------------
    ###################################################
    args.mode = 'predict'#预测，并非填补缺失值
    args.pred_len = seq_len# 预测步长设置，用于 dataloader_cond
    test_dataloader_info = build_dataloader_cond(configs, args) # 构建测试集 dataloader（有条件的数据）condition

    # 取得标准化后的测试数据
    test_scaled = test_dataloader_info['dataset'].samples
    scaler = test_dataloader_info['dataset'].scaler
    seq_length, feat_num = seq_len*2, test_scaled.shape[-1]# 序列长度 & 变量数
    pred_length = seq_len
    real = test_scaled
    test_dataset = test_dataloader_info['dataset'] # 提取测试 data/dataloader
    test_dataloader = test_dataloader_info['dataloader']
    ###################################################
    # 调用 diffusion 模型进行预测
    # 返回：
    #   sample → 预测值
    #   real_  → 对齐后的真实值
    ###################################################

    sample, real_ = trainer.sample_forecast(test_dataloader, shape=[seq_len, feat_num])
    mask = test_dataset.masking # 掩码，用于忽略某些值（若数据有缺失）

    ###################################################
    # 计算 MSE/MAE 误差
    ###################################################
    mse = mean_squared_error(sample.reshape(-1), real_.reshape(-1))
    mae = mean_absolute_error(sample.reshape(-1), real_.reshape(-1))
    print(mse, mae)

    print("sample shape:", np.array(sample).shape)
    print("real_  shape:", np.array(real_).shape)

    s = np.array(sample).reshape(-1)
    r = np.array(real_).reshape(-1)

    print("sample stats: min/max/mean/std", s.min(), s.max(), s.mean(), s.std())
    print("real_  stats: min/max/mean/std", r.min(), r.max(), r.mean(), r.std())

    # print("L2 diff mean:", np.mean((s - r) ** 2))
    # print("MAE mean    :", np.mean(np.abs(s - r)))

    # # 核心：看 sample 和 real_ 是否几乎一样
    # print("corrcoef:", np.corrcoef(s[:5000], r[:5000])[0, 1])
    # print("max abs diff:", np.max(np.abs(s - r)))

   