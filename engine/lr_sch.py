import math
from torch import inf
from torch.optim.optimizer import Optimizer


class ReduceLROnPlateauWithWarmup(object):
    """

    在 ReduceLROnPlateau 的基础上增加 warmup 机制的学习率调度器

    核心思想：
    1. 前 warmup 个 epoch：线性升高学习率
    2. warmup 结束后：若监控指标在 patience 个 epoch 内无明显改善，则降低学习率
    当监控指标停止改善时，降低学习率。

    在训练过程中，一旦学习出现停滞，通常将学习率降低
    2–10 倍可以带来更好的收敛效果。该调度器会读取一个
    监控指标（如验证集 loss），如果在连续 `patience`
    个 epoch 内未观察到明显改善，则降低学习率。

    参数说明：
        optimizer (Optimizer):
            被包装的 PyTorch 优化器。

        mode (str):
            取值为 `min` 或 `max`。
            - `min`：当监控指标（如 loss）停止下降时，降低学习率；
            - `max`：当监控指标（如 accuracy）停止上升时，降低学习率。
            默认值：`'min'`。

        factor (float):
            学习率缩放因子。学习率更新规则为：
                new_lr = lr * factor
            默认值：0.1。

        patience (int):
            在降低学习率之前，允许指标连续未改善的 epoch 数。
            例如，当 `patience = 2` 时：
            - 前 2 个未改善的 epoch 会被忽略；
            - 若第 3 个 epoch 指标仍未改善，则触发学习率下降。
            默认值：10。

        threshold (float):
            用于判断“是否显著改善”的阈值，避免将微小数值波动
            误判为性能提升。
            默认值：1e-4。

        threshold_mode (str):
            取值为 `rel` 或 `abs`，表示阈值的计算方式：
            - `rel`（相对模式）：
                - `max` 模式下：dynamic_threshold = best * (1 + threshold)
                - `min` 模式下：dynamic_threshold = best * (1 - threshold)
            - `abs`（绝对模式）：
                - `max` 模式下：dynamic_threshold = best + threshold
                - `min` 模式下：dynamic_threshold = best - threshold
            默认值：`'rel'`。

        cooldown (int):
            在学习率被降低之后，等待多少个 epoch 才恢复
            正常的指标监控与学习率调整。
            冷却期内将忽略所有“未改善”的 epoch。
            默认值：0。

        min_lr (float 或 list):
            学习率下限。
            - 若为标量，则对所有参数组使用同一个下限；
            - 若为列表，则分别指定每个参数组的最小学习率。
            默认值：0。

        eps (float):
            学习率最小变化量阈值。
            若新旧学习率之差小于 `eps`，则忽略本次更新。
            默认值：1e-8。

        verbose (bool):
            若为 True，则在每次学习率更新时打印日志信息。
            默认值：False。

        warmup_lr (float 或 None):
            warmup 结束时要达到的目标学习率。
            若为 None，则不启用 warmup 机制。

        warmup (int):
            warmup 的 epoch 数。
            在 warmup 阶段内，学习率将从初始值线性增加到 `warmup_lr`。

    """
    def __init__(self, optimizer, mode='min', factor=0.1, patience=10,
                 threshold=1e-4, threshold_mode='rel', cooldown=0,
                 min_lr=0, eps=1e-8, verbose=False, warmup_lr=None,
                 warmup=0):
        # factor 必须 < 1，用于学习率衰减
        if factor >= 1.0:
            raise ValueError('Factor should be < 1.0.')
        self.factor = factor

        # ===== 绑定 optimizer =====
        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer

        if isinstance(min_lr, list) or isinstance(min_lr, tuple):
            if len(min_lr) != len(optimizer.param_groups):
                raise ValueError("expected {} min_lrs, got {}".format(
                    len(optimizer.param_groups), len(min_lr)))
            self.min_lrs = list(min_lr)
        else:
            self.min_lrs = [min_lr] * len(optimizer.param_groups)

        self.patience = patience
        self.verbose = verbose
        self.cooldown = cooldown
        self.cooldown_counter = 0
        self.mode = mode
        self.threshold = threshold
        self.threshold_mode = threshold_mode

        self.warmup_lr = warmup_lr
        self.warmup = warmup
        
        self.best = None
        self.num_bad_epochs = None
        self.mode_worse = None  # the worse value for the chosen mode
        self.eps = eps
        self.last_epoch = 0
        self._init_is_better(mode=mode, threshold=threshold,
                             threshold_mode=threshold_mode)
        self._reset()

    def _prepare_for_warmup(self):
        if self.warmup_lr is not None:
            if isinstance(self.warmup_lr, (list, tuple)):
                if len(self.warmup_lr) != len(self.optimizer.param_groups):
                    raise ValueError("expected {} warmup_lrs, got {}".format(
                        len(self.optimizer.param_groups), len(self.warmup_lr)))
                self.warmup_lrs = list(self.warmup_lr)
            else:
                self.warmup_lrs = [self.warmup_lr] * len(self.optimizer.param_groups)
        else:
            self.warmup_lrs = None
        if self.warmup > self.last_epoch:
            curr_lrs = [group['lr'] for group in self.optimizer.param_groups]
            self.warmup_lr_steps = [max(0, (self.warmup_lrs[i] - curr_lrs[i])/float(self.warmup)) for i in range(len(curr_lrs))]
        else:
            self.warmup_lr_steps = None

    def _reset(self):
        """Resets num_bad_epochs counter and cooldown counter."""
        self.best = self.mode_worse
        self.cooldown_counter = 0
        self.num_bad_epochs = 0

    def step(self, metrics):
        # convert `metrics` to float, in case it's a zero-dim Tensor
        current = float(metrics)
        epoch = self.last_epoch + 1
        self.last_epoch = epoch

        if epoch <= self.warmup:
            self._increase_lr(epoch)
        else:
            if self.is_better(current, self.best):
                self.best = current
                self.num_bad_epochs = 0
            else:
                self.num_bad_epochs += 1

            if self.in_cooldown:
                self.cooldown_counter -= 1
                self.num_bad_epochs = 0  # ignore any bad epochs in cooldown

            if self.num_bad_epochs > self.patience:
                self._reduce_lr(epoch)
                self.cooldown_counter = self.cooldown
                self.num_bad_epochs = 0

            self._last_lr = [group['lr'] for group in self.optimizer.param_groups]

    def _reduce_lr(self, epoch):
        for i, param_group in enumerate(self.optimizer.param_groups):
            old_lr = float(param_group['lr'])
            new_lr = max(old_lr * self.factor, self.min_lrs[i])
            if old_lr - new_lr > self.eps:
                param_group['lr'] = new_lr
                if self.verbose:
                    print('Epoch {:5d}: reducing learning rate'
                          ' of group {} to {:.4e}.'.format(epoch, i, new_lr))

    def _increase_lr(self, epoch):
        # used for warmup
        for i, param_group in enumerate(self.optimizer.param_groups):
            old_lr = float(param_group['lr'])
            new_lr = max(old_lr + self.warmup_lr_steps[i], self.min_lrs[i])
            param_group['lr'] = new_lr
            if self.verbose:
                print('Epoch {:5d}: increasing learning rate'
                        ' of group {} to {:.4e}.'.format(epoch, i, new_lr))

    @property
    def in_cooldown(self):
        return self.cooldown_counter > 0

    def is_better(self, a, best):
        if self.mode == 'min' and self.threshold_mode == 'rel':
            rel_epsilon = 1. - self.threshold
            return a < best * rel_epsilon

        elif self.mode == 'min' and self.threshold_mode == 'abs':
            return a < best - self.threshold

        elif self.mode == 'max' and self.threshold_mode == 'rel':
            rel_epsilon = self.threshold + 1.
            return a > best * rel_epsilon

        else:  # mode == 'max' and epsilon_mode == 'abs':
            return a > best + self.threshold

    def _init_is_better(self, mode, threshold, threshold_mode):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if threshold_mode not in {'rel', 'abs'}:
            raise ValueError('threshold mode ' + threshold_mode + ' is unknown!')

        if mode == 'min':
            self.mode_worse = inf
        else:  # mode == 'max':
            self.mode_worse = -inf

        self.mode = mode
        self.threshold = threshold
        self.threshold_mode = threshold_mode

        self._prepare_for_warmup()

    def state_dict(self):
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)
        self._init_is_better(mode=self.mode, threshold=self.threshold, threshold_mode=self.threshold_mode)


# class CosineAnnealingLRWithWarmup(object):
#     """
#     adjust lr:
#
#     args:
#         warmup_lr: float or None, the learning rate to be touched after warmup
#         warmup: int, the number of steps to warmup
#     """
#
#     def __init__(self, optimizer, T_max, last_epoch=-1, verbose=False,
#                  min_lr=0, warmup_lr=None, warmup=0):
#         self.optimizer = optimizer
#         self.T_max = T_max
#         self.last_epoch = last_epoch
#         self.verbose = verbose
#         self.warmup_lr = warmup_lr
#         self.warmup = warmup
#
#         if isinstance(min_lr, list) or isinstance(min_lr, tuple):
#             if len(min_lr) != len(optimizer.param_groups):
#                 raise ValueError("expected {} min_lrs, got {}".format(
#                     len(optimizer.param_groups), len(min_lr)))
#             self.min_lrs = list(min_lr)
#         else:
#             self.min_lrs = [min_lr] * len(optimizer.param_groups)
#         self.max_lrs = [lr for lr in self.min_lrs]
#
#         self._prepare_for_warmup()
#
#     def step(self):
#         epoch = self.last_epoch + 1
#         self.last_epoch = epoch
#
#         if epoch <= self.warmup:
#             self._increase_lr(epoch)
#         else:
#             self._reduce_lr(epoch)
#
#     def _reduce_lr(self, epoch):
#         for i, param_group in enumerate(self.optimizer.param_groups):
#             progress = float(epoch - self.warmup) / float(max(1, self.T_max - self.warmup))
#             factor = max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
#             old_lr = float(param_group['lr'])
#             new_lr = max(self.max_lrs[i] * factor, self.min_lrs[i])
#             param_group['lr'] = new_lr
#             if self.verbose:
#                 print('Epoch {:5d}: reducing learning rate'
#                         ' of group {} to {:.4e}.'.format(epoch, i, new_lr))
#
#     def _increase_lr(self, epoch):
#         # used for warmup
#         for i, param_group in enumerate(self.optimizer.param_groups):
#             old_lr = float(param_group['lr'])
#             new_lr = old_lr + self.warmup_lr_steps[i]
#             param_group['lr'] = new_lr
#             self.max_lrs[i] = max(self.max_lrs[i], new_lr)
#             if self.verbose:
#                 print('Epoch {:5d}: increasing learning rate'
#                         ' of group {} to {:.4e}.'.format(epoch, i, new_lr))
#
#     def _prepare_for_warmup(self):
#         if self.warmup_lr is not None:
#             if isinstance(self.warmup_lr, (list, tuple)):
#                 if len(self.warmup_lr) != len(self.optimizer.param_groups):
#                     raise ValueError("expected {} warmup_lrs, got {}".format(
#                         len(self.optimizer.param_groups), len(self.warmup_lr)))
#                 self.warmup_lrs = list(self.warmup_lr)
#             else:
#                 self.warmup_lrs = [self.warmup_lr] * len(self.optimizer.param_groups)
#         else:
#             self.warmup_lrs = None
#         if self.warmup > self.last_epoch:
#             curr_lrs = [group['lr'] for group in self.optimizer.param_groups]
#             self.warmup_lr_steps = [max(0, (self.warmup_lrs[i] - curr_lrs[i])/float(self.warmup)) for i in range(len(curr_lrs))]
#         else:
#             self.warmup_lr_steps = None
#
#
#     def state_dict(self):
#         return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}
#
#     def load_state_dict(self, state_dict):
#         self.__dict__.update(state_dict)
#         self._prepare_for_warmup()