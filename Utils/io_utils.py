import os
import sys
import yaml
import json
import torch
import random
import warnings
import importlib
import numpy as np


def load_yaml_config(path):
    """加载YAML配置文件

        Args:
            path: YAML文件路径

        Returns:
            config: 解析后的配置字典
    """
    with open(path) as f:
        config = yaml.full_load(f)# 使用full_load安全地加载YAML
    return config

def instantiate_from_config(config):
    """根据配置动态实例化类

       Args:
           config: 包含'target'键（类路径）和可选'params'键（参数字典）的配置

       Returns:
           实例化的对象

       Raises:
           KeyError: 如果配置中缺少'target'键
    """
    if config is None:
        return None
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    module, cls = config["target"].rsplit(".", 1)
    #Models.autoregressive_diffusion.armd.ARMD分割成module_path = "Models.autoregressive_diffusion.armd", class_name = "ARMD"
    cls = getattr(importlib.import_module(module, package=None), cls)
    # 动态导入模块并获取类对象
    # importlib.import_module(module, package=None): 动态导入指定模块
    # getattr(module_obj, cls): 从导入的模块对象中获取指定的类对象，从armd.py文件中获取ARMD类
    return cls(**config.get("params", dict()))
    # 使用反射获取的类进行实例化
    # config.get("params", dict()): 获取config中的"params"键对应的值，如果不存在则返回空字典
    # **操作符: 将字典解包为关键字参数传递给类的构造函数
    # 最终返回类的实例对象

def save_config_to_yaml(config, path):
    """将配置保存为YAML文件

       Args:
           config: 配置字典
           path: 保存路径，必须以.yaml结尾
    """
    assert path.endswith('.yaml')#"保存路径必须以.yaml结尾"
    with open(path, 'w') as f:
        f.write(yaml.dump(config))
        f.close()
    # 写入YAML格式的配置
    # 不需要显式调用f.close()，with语句会自动处理

def save_dict_to_json(d, path, indent=None):
    """将字典保存为JSON文件

        Args:
            d: 要保存的字典
            path: 保存路径
            indent: JSON缩进格式
    """
    json.dump(d, open(path, 'w'), indent=indent)# 使用with语句确保文件正确关闭

def load_dict_from_json(path):
    """从JSON文件加载字典

       Args:
           path: JSON文件路径

       Returns:
           解析后的字典
    """
    return json.load(open(path, 'r'))# 使用with语句确保文件正确关闭

def write_args(args, path):
    """将命令行参数写入文件

        Args:
            args: 参数对象（通常来自argparse）
            path: 写入的文件路径
    """
    # 获取所有非私有属性（不以_开头的属性）
    args_dict = dict((name, getattr(args, name)) for name in dir(args)if not name.startswith('_'))# 过滤私有属性
    with open(path, 'a') as args_file: # 'a'模式表示追加写入
        # 写入PyTorch和cuDNN版本信息
        args_file.write('==> torch version: {}\n'.format(torch.__version__))
        args_file.write('==> cudnn version: {}\n'.format(torch.backends.cudnn.version()))
        # 写入执行的命令行
        args_file.write('==> Cmd:\n')
        args_file.write(str(sys.argv))
        # 写入所有参数
        args_file.write('\n==> args:\n')
        # 按字母顺序排序
        for k, v in sorted(args_dict.items()):
            args_file.write('  %s: %s\n' % (str(k), str(v)))
        args_file.close()# 不需要显式调用close()，with语句会自动处理

def seed_everything(seed, cudnn_deterministic=False):
    """设置所有随机数生成器的种子以确保可重复性

    Args:
        seed: 随机种子值
        cudnn_deterministic: 是否启用cuDNN确定性模式（可能会降低性能）
    """
    if seed is not None:
        print(f"Global seed set to {seed}")
        random.seed(seed)# Python随机模块
        np.random.seed(seed) # NumPy随机生成器
        torch.manual_seed(seed)# PyTorch CPU随机种子
        torch.cuda.manual_seed_all(seed)# PyTorch GPU随机种子（所有设备）
        torch.backends.cudnn.deterministic = False# 默认关闭确定性模式
    # 启用cuDNN确定性模式，确保可重复性但可能降低性能
    if cudnn_deterministic:
        torch.backends.cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')
        """'已启用确定性训练模式。'
            '这将开启CUDNN确定性设置，'
            '可能会显著降低训练速度！'
            '从检查点恢复时可能会出现意外行为。'
        """

def merge_opts_to_config(config, opts):
    """通过命令行选项修改配置

        Args:
            config: 原始配置字典
            opts: 命令行选项列表，格式为[name1, value1, name2, value2, ...]

        Returns:
            修改后的配置字典
    """
    def modify_dict(c, nl, v):
        """递归修改嵌套字典

                Args:
                    c: 当前层级的字典
                    name_list: 用点分隔的路径列表，如['model', 'layers']
                    value: 要设置的值
        """
        if len(nl) == 1:# 到达目标层级，转换值类型以匹配原始类型
            c[nl[0]] = type(c[nl[0]])(v)
        else:
            # print(nl) # 继续深入下一层级
            c[nl[0]] = modify_dict(c[nl[0]], nl[1:], v)
        return c

    # 验证选项数量为偶数（名称-值对）
    if opts is not None and len(opts) > 0:
        #"每个选项应包含名称和值！长度应为偶数！"
        assert len(opts) % 2 == 0, "each opts should be given by the name and values! The length shall be even number!"
        for i in range(len(opts) // 2): # 处理所有选项对
            name = opts[2*i]
            value = opts[2*i+1]
            config = modify_dict(config, name.split('.'), value)
    return config 

def modify_config_for_debug(config):
    """修改配置以用于调试模式

       Args:
           config: 原始配置

       Returns:
           修改后的配置（减少数据加载器工作进程和批量大小）
       """
    config['dataloader']['num_workers'] = 0# 调试时设为0以避免多进程问题
    config['dataloader']['batch_size'] = 1# 调试时使用最小批量大小
    return config

def get_model_parameters_info(model):
    """获取模型参数统计信息

      Args:
          model: PyTorch模型

      Returns:
          包含各子模块参数数量的字典，参数数量已格式化为易读形式
    """
    # for mn, m in model.named_modules():
    parameters = {'overall': {'trainable': 0, 'non_trainable': 0, 'total': 0}}
    # 遍历所有子模块
    for child_name, child_module in model.named_children():
        parameters[child_name] = {'trainable': 0, 'non_trainable': 0}
        # 统计子模块的参数
        for pn, p in child_module.named_parameters():
            if p.requires_grad:
                parameters[child_name]['trainable'] += p.numel()# 可训练参数
            else:
                parameters[child_name]['non_trainable'] += p.numel()# 不可训练参数
        parameters[child_name]['total'] = parameters[child_name]['trainable'] + parameters[child_name]['non_trainable']
        # 计算子模块参数总数
        parameters['overall']['trainable'] += parameters[child_name]['trainable']
        parameters['overall']['non_trainable'] += parameters[child_name]['non_trainable']
        parameters['overall']['total'] += parameters[child_name]['total']
    
    # format the numbers
    def format_number(num):
        """将数字格式化为易读形式（K/M/G）
                Args:
                    num: 原始数字

                Returns:
                    格式化后的字符串，如'1.5M'
        """
        K = 2**10# 1024
        M = 2**20 # 1048576
        G = 2**30
        if num > G: # K
            uint = 'G'
            num = round(float(num)/G, 2)
        elif num > M:
            uint = 'M'
            num = round(float(num)/M, 2)
        elif num > K:
            uint = 'K'
            num = round(float(num)/K, 2)
        else:
            uint = ''
        
        return '{}{}'.format(num, uint)
    
    def format_dict(d):
        """递归格式化字典中的所有数值
            Args:
            d: 要格式化的字典
        """
        for k, v in d.items():
            if isinstance(v, dict):
                format_dict(v)# 递归处理嵌套字典
            else:
                d[k] = format_number(v) # 格式化数值

    # 格式化所有参数数量
    format_dict(parameters)
    return parameters

def format_seconds(seconds):
    """将秒数格式化为易读的时间字符串

       Args:
           seconds: 总秒数

       Returns:
           格式化的时间字符串，如'01d:12h:30m:45s'
    """
    h = int(seconds // 3600) # 计算天、小时、分钟、秒
    m = int(seconds // 60 - h * 60)
    s = int(seconds % 60)

    d = int(h // 24)
    h = h - d * 24

    if d == 0:
        if h == 0:
            if m == 0:
                ft = '{:02d}s'.format(s)# 仅显示秒
            else:
                ft = '{:02d}m:{:02d}s'.format(m, s)#分钟和秒
        else:
           ft = '{:02d}h:{:02d}m:{:02d}s'.format(h, m, s)
 
    else:
        ft = '{:d}d:{:02d}h:{:02d}m:{:02d}s'.format(d, h, m, s)

    return ft


def class_from_string(class_name):
    module, cls = class_name.rsplit(".", 1)
    cls = getattr(importlib.import_module(module, package=None), cls)
    return cls

def get_all_file(dir, end_with='.h5'):
    if isinstance(end_with, str):
        end_with = [end_with]
    filenames = []
    for root, dirs, files in os.walk(dir):
        for f in files:
            for ew in end_with:
                if f.endswith(ew):
                    filenames.append(os.path.join(root, f))
                    break
    return filenames

def get_sub_dirs(dir, abs=True):
    sub_dirs = os.listdir(dir)
    if abs:
        sub_dirs = [os.path.join(dir, s) for s in sub_dirs]
    return sub_dirs

def get_model_buffer(model):
    state_dict = model.state_dict()
    buffers_ = {}
    params_ = {n: p for n, p in model.named_parameters()}

    for k in state_dict:
        if k not in params_:
            buffers_[k] = state_dict[k]
    return buffers_
