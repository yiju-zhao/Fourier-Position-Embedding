"""
Adapted from
[MosaiclML](https://github.com/mosaicml/examples.git) and
[minGPT](https://github.com/karpathy/minGPT.git)
"""

from __future__ import annotations

import logging
import math
import sys
from abc import abstractmethod
from collections import defaultdict
from functools import partial
from typing import (
    Callable,
    Dict,
    Iterable,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Set,
    Tuple,
    cast,
    Union
)

import torch
import torch.backends.cuda
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum

from .aliases import PathOrStr
from .beam_search import BeamSearch, Constraint, FinalSequenceScorer, Sampler
from .config import (
    ActivationCheckpointingStrategy,
    ActivationType,
    BlockType,
    CheckpointType,
    FSDPWrapStrategy,
    InitFnType,
    LayerNormType,
    ModelConfig,
    ShardedCheckpointerType,
    TrainConfig,
)
from .exceptions import OLMoConfigurationError
# from .initialization import init_normal
from .torch_util import ensure_finite_, get_cumulative_document_lengths

if sys.version_info.minor > 8:
    from collections.abc import MutableMapping
elif sys.version_info.minor == 8:
    from typing import MutableMapping
else:
    raise SystemExit("This script supports Python 3.8 or higher")

__all__ = [
    "LayerNormBase",
    "LayerNorm",
    "RMSLayerNorm",
    "RotaryEmbedding",
    "FourierEmbedding",
    "Activation",
    "GELU",
    "ReLU",
    "SwiGLU",
    "OLMoBlock",
    "OLMoSequentialBlock",
    "OLMo",
    "OLMoOutput",
    "OLMoGenerateOutput",
]


log = logging.getLogger(__name__)

def init_normal(
    module: Union[nn.Linear, nn.Embedding],
    std: float,
    init_cutoff_factor: Optional[float] = None,
):
    # weights
    if init_cutoff_factor is not None:
        cutoff_value = init_cutoff_factor * std
        nn.init.trunc_normal_(module.weight, mean=0.0, std=std, a=-cutoff_value, b=cutoff_value)
    else:
        nn.init.normal_(module.weight, mean=0.0, std=std)

    # biases
    if isinstance(module, nn.Linear) and module.bias is not None:
        nn.init.zeros_(module.bias)


def activation_checkpoint_function(cfg: ModelConfig):
    preserve_rng_state = not (
        (cfg.attention_dropout == 0.0) and (cfg.embedding_dropout == 0.0) and (cfg.residual_dropout == 0.0)
    )
    from torch.utils.checkpoint import checkpoint

    return partial(
        checkpoint,
        preserve_rng_state=preserve_rng_state,
        use_reentrant=False,
    )


def should_checkpoint_block(strategy: Optional[ActivationCheckpointingStrategy], block_idx: int) -> bool:
    if strategy is None:
        return False
    elif (
        (strategy == ActivationCheckpointingStrategy.whole_layer)
        or (strategy == ActivationCheckpointingStrategy.one_in_two and block_idx % 2 == 0)
        or (strategy == ActivationCheckpointingStrategy.one_in_three and block_idx % 3 == 0)
        or (strategy == ActivationCheckpointingStrategy.one_in_four and block_idx % 4 == 0)
        or (strategy == ActivationCheckpointingStrategy.two_in_three and block_idx % 3 != 0)
        or (strategy == ActivationCheckpointingStrategy.three_in_four and block_idx % 4 != 0)
    ):
        return True
    else:
        return False


class BufferCache(dict, MutableMapping[str, torch.Tensor]):
    """
    Cache for attention biases and other things that would normally be stored as buffers.
    We avoid using buffers because we've run into various issues doing so with FSDP.
    In general it appears the way FSDP handles buffers is not well-defined.
    It doesn't shard them but apparently it does synchronize them across processes, which we want to avoid
    since (A) it isn't necessary, and (B) we sometimes have `-inf` in these biases which might get turned into
    NaNs when they're synchronized due to casting or some other issue.
    """


def _non_meta_init_device(config: ModelConfig) -> torch.device:
    if config.init_device is not None and config.init_device != "meta":
        return torch.device(config.init_device)
    else:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Dropout(nn.Dropout):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.p == 0.0:
            return input
        else:
            return F.dropout(input, self.p, self.training, self.inplace)


class LayerNormBase(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        *,
        size: Optional[int] = None,
        elementwise_affine: Optional[bool] = True,
    ):
        super().__init__()
        self.config = config
        self.eps = config.layer_norm_eps
        self.normalized_shape = (size or config.d_model,)
        if elementwise_affine or (elementwise_affine is None and self.config.layer_norm_with_affine):
            self.weight = nn.Parameter(torch.ones(self.normalized_shape, device=config.init_device))
            use_bias = self.config.bias_for_layer_norm
            if use_bias is None:
                use_bias = self.config.include_bias
            if use_bias:
                self.bias = nn.Parameter(torch.zeros(self.normalized_shape, device=config.init_device))
            else:
                self.register_parameter("bias", None)
        else:
            self.register_parameter("bias", None)
            self.register_parameter("weight", None)

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @classmethod
    def build(cls, config: ModelConfig, size: Optional[int] = None, **kwargs) -> LayerNormBase:
        if config.layer_norm_type == LayerNormType.default:
            return LayerNorm(config, size=size, low_precision=False, **kwargs)
        elif config.layer_norm_type == LayerNormType.low_precision:
            return LayerNorm(config, size=size, low_precision=True, **kwargs)
        elif config.layer_norm_type == LayerNormType.rms:
            return RMSLayerNorm(config, size=size, **kwargs)
        else:
            raise NotImplementedError(f"Unknown LayerNorm type: '{config.layer_norm_type}'")

    def _cast_if_autocast_enabled(self, tensor: torch.Tensor, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        # NOTE: `is_autocast_enabled()` only checks for CUDA autocast, so we use the separate function
        # `is_autocast_cpu_enabled()` for CPU autocast.
        # See https://github.com/pytorch/pytorch/issues/110966.
        if tensor.device.type == "cuda" and torch.is_autocast_enabled():
            return tensor.to(dtype=dtype if dtype is not None else torch.get_autocast_gpu_dtype())
        elif tensor.device.type == "cpu" and torch.is_autocast_cpu_enabled():
            return tensor.to(dtype=dtype if dtype is not None else torch.get_autocast_cpu_dtype())
        else:
            return tensor

    def reset_parameters(self):
        if self.weight is not None:
            torch.nn.init.ones_(self.weight)  # type: ignore
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)  # type: ignore


class LayerNorm(LayerNormBase):
    """
    The default :class:`LayerNorm` implementation which can optionally run in low precision.
    """

    def __init__(
        self,
        config: ModelConfig,
        size: Optional[int] = None,
        low_precision: bool = False,
        elementwise_affine: Optional[bool] = None,
    ):
        super().__init__(config, size=size, elementwise_affine=elementwise_affine)
        self.low_precision = low_precision

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.low_precision:
            module_device = x.device
            downcast_x = self._cast_if_autocast_enabled(x)
            downcast_weight = (
                self._cast_if_autocast_enabled(self.weight) if self.weight is not None else self.weight
            )
            downcast_bias = self._cast_if_autocast_enabled(self.bias) if self.bias is not None else self.bias
            with torch.autocast(enabled=False, device_type=module_device.type):
                return F.layer_norm(
                    downcast_x, self.normalized_shape, weight=downcast_weight, bias=downcast_bias, eps=self.eps
                )
        else:
            return F.layer_norm(x, self.normalized_shape, weight=self.weight, bias=self.bias, eps=self.eps)


class RMSLayerNorm(LayerNormBase):
    """
    RMS layer norm, a simplified :class:`LayerNorm` implementation
    """

    def __init__(
        self,
        config: ModelConfig,
        size: Optional[int] = None,
        elementwise_affine: Optional[bool] = None,
    ):
        super().__init__(config, size=size, elementwise_affine=elementwise_affine)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.autocast(enabled=False, device_type=x.device.type):
            og_dtype = x.dtype
            x = x.to(torch.float32)
            variance = x.pow(2).mean(-1, keepdim=True)
            x = x * torch.rsqrt(variance + self.eps)
            x = x.to(og_dtype)

        if self.weight is not None:
            if self.bias is not None:
                return self.weight * x + self.bias
            else:
                return self.weight * x
        else:
            return x
        
class RotaryEmbedding(nn.Module):
    """
    旋转位置编码（RoPE，Rotary Positional Embeddings）

    参考文献：https://arxiv.org/abs/2104.09864

    RoPE 通过将每个 query/key 向量在复平面上"旋转"来注入位置信息，而不是像原始 Transformer 那样"加"上位置向量。
    具体来说，每对偶/奇隐藏维度被视为一个复数。对于位置 t 和频率索引 i，旋转角度为 φ = t · θᵢ，其中：

        θᵢ = 1 / θ^{2i/d}

    θ 通常取 10,000。旋转可用如下 2×2 矩阵实现：

        (x₀, x₁) → (x₀ · cos φ − x₁ · sin φ,
                    x₀ · sin φ + x₁ · cos φ)

    这种方式可以保持向量点积仅与相对位置有关，使模型既能感知相对位置，也能保留绝对位置信息。

    本实现在原始 RoPE 基础上做了大量扩展：
    - 支持可学习频率（freq_learnable），可选每个头不同（rope_no_repetition）
    - 支持多种频率分布（uniform, gaussian, exponential, linear, constant）
    - 支持频率裁剪（clamp_floor_freq, clamp_upper_freq），可零化或对齐到线性网格（clamp_to_linear）
    - 支持对 embedding 层（prefix="embed"）和 attention 层（prefix="attn"）分别应用 RoPE

    该类的主要工作是构造逆频率表 inv_freq，缓存正弦/余弦张量以加速推理，并处理各种配置选项，便于研究不同位置编码变体。
    """

    def __init__(
        self, 
        config: ModelConfig, 
        cache: BufferCache, 
        dim: int = None,
        n_channels: int = None,
        prefix: str = "attn",
        freq_distribution: str = None,
        freq_learnable: bool = None,
        use_rope_cache: bool = True,
        include_neg_freq: bool = None,
        floor_freq_ratio: float = None,
        clamp_floor_freq: bool = None,
        clamp_floor_to_zero: bool = None,
        upper_freq_ratio: float = None,
        clamp_upper_freq: float = None,
        clamp_upper_to_zero: bool = None,
        zero_freq_ratio: float = None,
        clamp_to_linear: bool = None,
        clamp_to_linear_mode: str = None,
        init_upper_freq: float = None,
        init_floor_freq: float = None,
    ):
        super().__init__()
        self.config = config
        self.__cache = cache
        
        self.prefix = prefix
        self.suffix = "rope"
        self.use_rope_cache = use_rope_cache
        
        self.freq_distribution = freq_distribution if freq_distribution is not None else self.config.rope_init_distribution
        if self.freq_distribution not in ["constant", "linear", "uniform", "gaussian", "exponential"]:
            raise ValueError(f"Unsupported frequency distribution: {self.freq_distribution}")
        
        self.freq_learnable = freq_learnable if freq_learnable is not None else self.config.rope_learnable
        self.include_neg_freq = include_neg_freq if include_neg_freq is not None else self.config.rope_include_neg_freq
        
        if dim is not None:
            self.dim = dim
        elif self.prefix == "attn":
            self.dim = self.config.d_model // self.config.n_heads
        else:
            self.dim = self.config.d_model
        
        if n_channels is not None:
            self.n_channels = n_channels
        elif self.prefix == "attn" and self.config.rope_learnable and self.config.rope_no_repetition:
            self.n_channels = self.config.n_heads
        else:
            self.n_channels = 1
            
        self.init_floor_freq = init_floor_freq if init_floor_freq is not None else self.config.rope_init_floor_freq
        self.init_upper_freq = init_upper_freq if init_upper_freq is not None else self.config.rope_init_upper_freq
        
        self.clamp_floor_freq = clamp_floor_freq if clamp_floor_freq is not None else self.config.rope_clamp_floor_freq
        if self.clamp_floor_freq:
            self.floor_freq_ratio = floor_freq_ratio if floor_freq_ratio is not None else self.config.rope_floor_freq_ratio
            self.floor_freq = 2*torch.pi/self.config.max_sequence_length * self.floor_freq_ratio
            
            self.clamp_floor_to_zero = clamp_floor_to_zero if clamp_floor_to_zero is not None else self.config.rope_clamp_floor_to_zero
            self.clamp_floor_value = 0.0 if self.clamp_floor_to_zero else 2*torch.pi/self.config.max_sequence_length*self.clamp_floor_ratio
        else:
            self.floor_freq = 0.0
        
        self.clamp_upper_freq = clamp_upper_freq if clamp_upper_freq is not None else self.config.rope_clamp_upper_freq
        if self.clamp_upper_freq:
            self.upper_freq_ratio = upper_freq_ratio if upper_freq_ratio is not None else self.config.rope_upper_freq_ratio
            self.upper_freq = torch.pi * self.upper_freq_ratio
            
            self.clamp_upper_to_zero = clamp_upper_to_zero if clamp_upper_to_zero is not None else self.config.rope_clamp_upper_to_zero
            self.clamp_upper_value = 0.0 if self.clamp_upper_to_zero else torch.pi * self.clamp_upper_ratio
        else:
            self.upper_freq = 1.0
            
        if self.clamp_floor_freq or self.clamp_upper_freq:
            assert self.upper_freq >= self.floor_freq
        
        self.zero_freq_ratio = zero_freq_ratio if zero_freq_ratio is not None else self.config.rope_zero_freq_ratio
        
        self.clamp_to_linear = clamp_to_linear if clamp_to_linear is not None else self.config.rope_clamp_to_linear
        self.clamp_to_linear_mode = clamp_to_linear_mode if clamp_to_linear_mode is not None else self.config.rope_clamp_to_linear_mode
        
        if self.freq_learnable:
            self.inv_freq = nn.Parameter(
                self.get_inv_freq(self.dim, _non_meta_init_device(config)),
                requires_grad=True
            )
        else:
            self.inv_freq = self.get_inv_freq(self.dim, _non_meta_init_device(config))
            
            if self.use_rope_cache:
                # Warm up cache.
                self.get_rotary_embedding(config.max_sequence_length, _non_meta_init_device(config))
    
        
    def extra_yarn(self, inv_freq = None) -> torch.Tensor:
        # YaRN 核心函数：通过混合 extra 和 inter 频率解决长度外推问题
        """
        YaRN（Yet Another RoPE eXtrapolatioN）——RoPE长度外推增强

        YaRN 主要解决"长度外推"问题：模型通常只在固定长度（如2k token）上训练，但推理时希望能泛化到更长序列。
        核心思想是"混合"两种频率表：
        1. 标准 RoPE 的频率 inv_freq_extra（训练长度内外一致）
        2. 压缩频谱的 inv_freq_inter（通过 len_extra_yarn_scale 缩放频率，使长距离旋转变慢）

        通过线性 ramp（由 len_extra_yarn_beta_fast/beta_slow 控制）生成 mask，
        在 ramp 区间平滑过渡，两端分别用纯 extra 或纯 inter 频率。
        最终返回的 inv_freq 可直接用于后续位置编码。

        这种方法能让模型在远超训练长度的上下文中依然稳定推理，避免位置编码外推时的数值爆炸。
        """
        assert self.config.len_extra_orig_length != 0
        
        def find_correction_dim(num_rotations, dim, base=self.config.rope_theta, orig_max_position_embeddings=self.config.len_extra_orig_length):
            # 根据给定的旋转次数计算对应的维度索引
            # 公式来源于 RoPE 频率计算的逆向推导
            return (dim * math.log(orig_max_position_embeddings/(num_rotations * 2 * math.pi)))/(2 * math.log(base))

        def find_correction_range(low_rot, high_rot, dim, base=self.config.rope_theta, orig_max_position_embeddings=self.config.len_extra_orig_length):
            # 计算需要进行频率混合的维度范围
            # low_rot 和 high_rot 是 beta_fast 和 beta_slow 参数，控制混合的频率范围
            low = math.floor(find_correction_dim(
                low_rot, dim, base, orig_max_position_embeddings))
            high = math.ceil(find_correction_dim(
                high_rot, dim, base, orig_max_position_embeddings))
            return max(low, 0), min(high, dim-1)  # 确保索引在合法范围内

        def linear_ramp_mask(low, fast, dim):
            # 生成线性过渡掩码，用于平滑混合两种频率
            # 在 [low, fast] 区间内从 0 线性增长到 1
            if low == fast:
                fast += 0.001  # 防止除零错误
            linear_func = (torch.arange(dim, dtype=torch.float32) - low) / (fast - low)
            ramp_func = torch.clamp(linear_func, 0, 1)  # 限制在 [0, 1] 范围内
            return ramp_func
        
        if inv_freq is None:
            # 计算标准 RoPE 频率
            pos_freqs = self.config.rope_theta ** (torch.arange(0, self.dim, 2, device=_non_meta_init_device(self.config), dtype=torch.float) / self.dim)
            inv_freq_extra = 1.0 / pos_freqs  # 原始频率（训练长度内外一致）
            inv_freq_inter = 1.0 / (self.config.len_extra_yarn_scale * pos_freqs)  # 压缩频率（长距离旋转变慢）
        else:
            inv_freq_extra = inv_freq  # 使用传入的频率作为基础
            inv_freq_inter = inv_freq / self.config.len_extra_yarn_scale  # 对基础频率进行缩放
       
        # 计算混合范围并生成掩码
        low, high = find_correction_range(self.config.len_extra_yarn_beta_fast, self.config.len_extra_yarn_beta_slow, self.dim)
        inv_freq_mask = (1 - linear_ramp_mask(low, high, self.dim // 2).float().to(_non_meta_init_device(self.config))) * self.config.len_extra_yarn_factor
        # 根据掩码混合两种频率：低频用 extra，高频用 inter，中间平滑过渡
        inv_freq = inv_freq_inter * (1 - inv_freq_mask) + inv_freq_extra * inv_freq_mask
        
        return inv_freq
    
    def extra_pi(self, inv_freq = None) -> torch.Tensor:
        # Position Interpolation (PI) 方法的简单实现
        # 通过缩放频率来实现位置的线性插值，使长序列在训练范围内的位置被“压缩”
        assert self.config.len_extra_orig_length != 0
            
        # 计算压缩比例：原始训练长度 / 目标最大长度
        self.pi_ratio = self.config.len_extra_orig_length / self.config.max_sequence_length
        
        # 通过缩放频率来实现位置压缩
        inv_freq = inv_freq * self.pi_ratio
        
        return inv_freq
    
    def get_inv_freq(self, dim: int, device: torch.device) -> torch.Tensor:
        # 生成逆频率序列的核心函数，支持多种频率分布和各种裁剪策略
        if self.freq_distribution == "constant": # self.config.rope_no_rotary:
            # 常数频率分布：所有频率都相等，相当于去除位置编码
            torch.ones_like(torch.arange(0, dim, 2, device=device, dtype=torch.float))
        elif self.freq_distribution == "linear": # self.config.rope_linear:
            # 线性频率分布：频率与维度索引成正比
            inv_freq = 2*torch.pi/self.config.max_sequence_length * torch.arange(0, dim, 2, device=device, dtype=torch.float) 
            
            inv_freq[inv_freq > self.clamp_upper_ratio * torch.pi] = self.clamp_upper_value
            
            # 翻转，保持频率递减规律（与标准 RoPE 一致）
            inv_freq = inv_freq.flip(0)
            
        elif self.freq_distribution == "uniform": # self.config.rope_uniform:
            # 均匀频率分布：频率从 [0, 1] 均匀采样
            inv_freq = 1.0 * torch.rand(dim//2, device=device, dtype=torch.float)
            
        elif self.freq_distribution == "gaussian": # self.config.rope_gaussian:
            # 高斯频率分布：频率从高斯分布采样并取绝对值
            inv_freq = torch.randn(dim//2, device=device, dtype=torch.float).abs()
            inv_freq = inv_freq / inv_freq.max()  # 归一化到 [0, 1] 范围
        else:
            # 指数频率分布（标准 RoPE）：频率与维度索引成指数关系
            inv_freq = 1.0 / (
                self.config.rope_theta ** (torch.arange(0, dim, 2, device=device, dtype=torch.float) / dim)
            )
        
        # 将频率缩放到指定范围 [init_floor_freq, init_upper_freq]
        inv_freq = self.init_floor_freq + inv_freq * (self.init_upper_freq - self.init_floor_freq)
        
        # 在裁剪之前应用长度外推方法
        if self.config.len_extra and self.config.len_extra_before_clamp:
            if self.config.len_extra_type == "PI":
                inv_freq = self.extra_pi(inv_freq)  # Position Interpolation 方法
        
            elif self.config.len_extra_type == "YARN":
                inv_freq = self.extra_yarn(inv_freq)  # YaRN 方法
        
        # 频率裁剪：限制频率在指定范围内
        if self.clamp_floor_freq:
            inv_freq[inv_freq < self.floor_freq] = self.clamp_floor_value  # 低频裁剪
        if self.clamp_upper_freq:
            inv_freq[inv_freq > self.upper_freq] = self.clamp_upper_value  # 高频裁剪
            
        if self.clamp_to_linear:
            freq_ratio = inv_freq / (2*torch.pi/self.config.max_sequence_length)
            if self.clamp_to_linear_mode == "ceil":
                inv_freq = 2*torch.pi/self.config.max_sequence_length * freq_ratio.ceil()
            elif self.clamp_to_linear_mode == "floor":
                inv_freq = 2*torch.pi/self.config.max_sequence_length * freq_ratio.floor()
            elif self.clamp_to_linear_mode == "half":
                _half = (freq_ratio != 0).float() * 0.5
                inv_freq = 2*torch.pi/self.config.max_sequence_length * (freq_ratio.floor() + _half)
            elif self.clamp_to_linear_mode == "arange":
                _linear = (freq_ratio != 0).float()
                _linear[_linear!=0] = torch.arange(0, 1, 1/_linear.sum(), device=device, dtype=torch.float)
                
                inv_freq = 2*torch.pi/self.config.max_sequence_length * (freq_ratio.floor() + _linear)    
            elif self.clamp_to_linear_mode == "flip_arange":
                _linear = (freq_ratio != 0).float()
                _linear[_linear!=0] = (torch.arange(0, 1, 1/_linear.sum(), device=device, dtype=torch.float)).flip(0)
                
                inv_freq = 2*torch.pi/self.config.max_sequence_length * (freq_ratio.floor() + _linear)  
            else:
                raise ValueError(f"Unsupported clamp_to_linear_mode: {self.clamp_to_linear_mode}")
        
        if self.zero_freq_ratio >= 0.0:
            zero_freq_num = (inv_freq == 0.0).sum()
            zero_freq_num_ceil = math.ceil(dim//2 * self.zero_freq_ratio)
            
            if zero_freq_num > zero_freq_num_ceil:
                zero_freq_indeces = torch.nonzero(inv_freq == 0.0, as_tuple=False).squeeze()
                reset_freq_num = zero_freq_num - zero_freq_num_ceil

                inv_freq[zero_freq_indeces[:reset_freq_num]] = \
                    2*torch.pi/self.config.max_sequence_length*torch.arange(1, reset_freq_num+1, device=device, dtype=torch.float)
            else:
                non_zero_freq_indeces = torch.nonzero(inv_freq != 0.0, as_tuple=False).squeeze()
                reset_freq_num = zero_freq_num_ceil - zero_freq_num
                
                if self.clamp_upper_to_zero:
                    inv_freq[non_zero_freq_indeces[:reset_freq_num]] = 0.0
                else:
                    inv_freq[non_zero_freq_indeces[-reset_freq_num:]] = 0.0
                    
        if self.config.len_extra and not self.config.len_extra_before_clamp:
            if self.config.len_extra_type == "PI":
                inv_freq = self.extra_pi(inv_freq)
        
            elif self.config.len_extra_type == "YARN":
                inv_freq = self.extra_yarn(inv_freq)
        
        if self.include_neg_freq:
            inv_freq *= (-1) ** torch.arange(0, dim//2, device=device, dtype=torch.float)
        
        if self.config.fourier and self.config.fourier_ignore_zero:
            inv_freq = inv_freq[inv_freq != 0.0]
        
        if self.prefix == "embed":
            inv_freq = inv_freq # shape: (dim//2, )
        elif self.prefix == "attn":
            if self.config.rope_learnable and self.config.rope_no_repetition:
                inv_freq = inv_freq.repeat(self.n_channels, 1) # shape: (n_heads, dim//2)
            else:
                inv_freq = inv_freq[None, :] # shape: (1, dim//2)
        else:
            raise ValueError(f"Unsupported prefix: {self.prefix}")
        
        return inv_freq
        
    def get_rotary_embedding(
        self, 
        seq_len: int, 
        device: torch.device, 
        layer_idx: Optional[int] = None,
        use_rope_cache: bool = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # 生成旋转位置编码的正弦余弦值，支持缓存机制加速推理
        use_rope_cache = use_rope_cache or ((not self.freq_learnable) and self.use_rope_cache)
        
        # 检查缓存是否可用且包含足够长度的位置编码
        if use_rope_cache:
            if (
                (pos_sin := self.__cache.get(f"{self.prefix}_{self.suffix}_pos_sin")) is not None
                and (pos_cos := self.__cache.get(f"{self.prefix}_{self.suffix}_pos_cos")) is not None
                and pos_sin.shape[-2] >= seq_len
                and pos_cos.shape[-2] >= seq_len
            ):
                if pos_sin.device != device:
                    pos_sin = pos_sin.to(device)
                    self.__cache[f"{self.prefix}_{self.suffix}_pos_sin"] = pos_sin
                if pos_cos.device != device:
                    pos_cos = pos_cos.to(device)
                    self.__cache[f"{self.prefix}_{self.suffix}_pos_cos"] = pos_cos
                    
                if self.prefix == "embed":
                    return pos_sin[:, :seq_len, :], pos_cos[:, :seq_len, :]
                elif self.prefix == "attn":
                    return pos_sin[:, :, :seq_len, :], pos_cos[:, :, :seq_len, :]
                else:
                    raise ValueError(f"Unsupported prefix: {self.prefix}")

        with torch.autocast(device.type, enabled=False):
            
            # 如果频率可学习，需要对频率参数进行动态裁剪
            if self.freq_learnable:
                if self.clamp_floor_freq or self.clamp_upper_freq:
                    sign = self.inv_freq.data.sign()
                    self.inv_freq.data.abs_().clamp_(
                        2*torch.pi/self.config.max_sequence_length*self.clamp_floor_ratio, torch.pi*self.clamp_upper_ratio
                    ).mul_(sign)
                else:
                    self.inv_freq.data.clamp_(-torch.pi, torch.pi)  # 默认裁剪范围
            
            # 生成位置序列
            if self.config.rope_no_pos:
                seq = torch.ones(seq_len, device=device, dtype=torch.float) # 消融实验：全部位置都设为 1
            else:
                seq = torch.arange(seq_len, device=device, dtype=torch.float) # 正常情况：0, 1, 2, ..., seq_len-1
            
            # 计算每个位置和每个频率的相位角度
            if self.prefix == "embed":
                freqs = torch.einsum("t, d -> td", seq, self.inv_freq) # 嵌入层用：(seq_len, dim//2)
            elif self.prefix == "attn":
                freqs = torch.einsum("t, hd -> htd", seq, self.inv_freq) # 注意力层用：(1 or n_heads, seq_len, dim//2)
            else:
                raise ValueError(f"Unsupported prefix: {self.prefix}")
            
            # 根据不同的后缀生成位置编码
            if self.suffix == "fourier": 
                positions = freqs.unsqueeze(0) # FoPE：保持原始维度
            else:
                positions = torch.cat((freqs, freqs), dim=-1).unsqueeze(0) # RoPE：重复频率以匹配完整维度
                
            pos_sin, pos_cos = positions.sin(), positions.cos()  # 计算正弦余弦值

        # 将结果缓存以加速后续计算（仅在频率不可学习时）
        if (not self.freq_learnable) and self.use_rope_cache:
            self.__cache[f"{self.prefix}_{self.suffix}_pos_sin"] = pos_sin
            self.__cache[f"{self.prefix}_{self.suffix}_pos_cos"] = pos_cos
            
        return pos_sin, pos_cos

    def rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        # 实现复数旋转的关键操作：将向量的后半部分取负并与前半部分交换
        if self.prefix == "embed":
            B, T, hs = x.size()
            x = x.view(B, T, 2, hs // 2)  # 将向量重塑为偶数对
            
        elif self.prefix == "attn":
            B, nh, T, hs = x.size()
            x = x.view(B, nh, T, 2, hs // 2)  # 将向量重塑为偶数对
            
        x1, x2 = x.unbind(dim=-2)  # 分离偶数对的前后半部分
        return torch.cat((-x2, x1), dim=-1)  # 实现复数乘法中的 i 操作

    def apply_rotary_pos_emb(self, pos_sin: torch.Tensor, pos_cos: torch.Tensor, t: torch.Tensor, inverse: bool = False) -> torch.Tensor:
        # 应用旋转位置编码的核心函数，实现 (x + iy) * e^(iθ) 的复数旋转
        if not inverse:
            # 正向旋转： t * cos(θ) + rotate_half(t) * sin(θ)
            return ((t * pos_cos) + (self.rotate_half(t) * pos_sin)).to(t.dtype)
        else:
            # 逆向旋转： t * cos(θ) - rotate_half(t) * sin(θ)
            return ((t * pos_cos) - (self.rotate_half(t) * pos_sin)).to(t.dtype)
    
    def forward(
        self, 
        x: torch.Tensor, 
        all_len: int, 
        layer_idx: Optional[int] = None, 
        inverse: bool = False,
        use_rope_cache: bool = None
    ) -> torch.Tensor:
        # RotaryEmbedding 前向传播函数：对输入张量应用旋转位置编码
        
        # 如果需要全精度计算，将输入转为 float32
        if self.config.rope_full_precision:
            x_ = x.float()
        else:
            x_ = x
        
        with torch.autocast(x.device.type, enabled=False):
            x_len = x_.shape[-2]  # 获取输入序列的实际长度
            # 获取对应的正弦余弦位置编码
            pos_sin, pos_cos = self.get_rotary_embedding(all_len, x_.device, layer_idx=layer_idx, use_rope_cache=use_rope_cache)
            pos_sin = pos_sin.type_as(x_)
            pos_cos = pos_cos.type_as(x_)
            
            if self.prefix == "embed":
                # 嵌入层情况：位置编码的形状为 (1, seq_len, dim)
                x_ = self.apply_rotary_pos_emb(
                    pos_sin[:, all_len - x_len : all_len, :], 
                    pos_cos[:, all_len - x_len : all_len, :], 
                    x_,
                    inverse
                )
            elif self.prefix == "attn":
                # 注意力层情况：位置编码的形状为 (1, n_heads, seq_len, dim)
                x_ = self.apply_rotary_pos_emb(
                    pos_sin[:, :, all_len - x_len : all_len, :], 
                    pos_cos[:, :, all_len - x_len : all_len, :], 
                    x_,
                    inverse
                )
            else:
                raise ValueError(f"Unsupported prefix: {self.prefix}")
            
        return x_.type_as(x)  # 返回时恢复原始数据类型
    
class InverseRotaryEmbedding(RotaryEmbedding):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.suffix = "irope"
        
    def forward(self, x, all_len, layer_idx=None, use_rope_cache=None):
        return super().forward(x, all_len, layer_idx=layer_idx, use_rope_cache=use_rope_cache, inverse=True)

class FourierEmbedding(RotaryEmbedding):
    """
    傅里叶位置编码（FoPE，Fourier Positional Embeddings）

    FoPE 在 RoPE 的基础上进一步扩展：先用 RoPE 得到一组正弦/余弦基，再通过一个可学习（或固定）的线性变换（Fourier 矩阵）
    对这些基进行混合。可以理解为 RoPE 提供了丰富的频率，FoPE 让模型自动学习哪些频率组合对任务最有用。

    本实现的主要特性：
    - 支持降维：input_dim（RoPE 频率数）可以远大于 output_dim（实际注入模型的维度），便于探索丰富频谱而不增加内存
    - 支持每个头独立基（fourier_separate_head）或共享基
    - 支持正弦/余弦分开权重（fourier_separate_basis）或合并权重
    - 支持多种初始化方式（fourier_init*），如单位阵、加噪声、Xavier等，便于对比"从RoPE出发"还是"随机初始化"

    运行时，先和 RoPE 一样计算 pos_sin/pos_cos，再用 Fourier 权重投影得到 fourier_sin/fourier_cos，
    必要时补零，最后用和 RoPE 相同的旋转方式注入到注意力机制中。
    """

    def __init__(self, config, *args, **kwargs):
        self.config = config
        
        # 计算注意力头维度
        self.head_dim = self.config.d_model // self.config.n_heads
        # 选择输入维度：如果配置的 fourier_dim 较大则使用它，否则使用头维度
        dim = self.config.fourier_dim if self.config.fourier_dim > self.head_dim else self.head_dim
        
        # 调用父类 RotaryEmbedding 的初始化
        super().__init__(config, dim=dim, *args, **kwargs)
        
        self.suffix = "fourier"
        
        # 计算输入和输出维度
        if self.config.fourier_ignore_zero:
            # 如果忽略零频率，则输入维度为非零频率的数量
            self.input_dim = self.inv_freq.size(-1)
            self.output_dim = min(self.input_dim, self.head_dim//4) # TODO: self.head_dim//8
        else:
            # 正常情况下的输入输出维度
            self.input_dim = self.dim // 2
            self.output_dim = self.head_dim // 2
        
        # 设置输入输出张量的形状模式（用于 einsum 操作）
        if self.prefix == "embed":
            self.input_shape = "btD"  # (batch, time, input_dim)
            self.output_shape = "btd"  # (batch, time, output_dim)
        elif self.prefix == "attn":
            self.input_shape = "bhtD"  # (batch, head, time, input_dim)
            self.output_shape = "bhtd"  # (batch, head, time, output_dim)
            
        # 设置僅里叶系数的形状
        if self.prefix == "attn" and self.config.fourier_separate_head:
            # 每个注意力头都有独立的僅里叶系数
            size = (self.config.n_heads, self.input_dim, self.output_dim)
            self.coef_shape = "hDd"  # (head, input_dim, output_dim)
        else:
            # 所有注意力头共享同一个僅里叶系数
            size = (self.input_dim, self.output_dim)
            self.coef_shape = "Dd"  # (input_dim, output_dim)
        
        # 初始化僅里叶系数参数
        if self.config.fourier_separate_basis:
            # 正弦和余弦分开的系数矩阵
            self.sin_coef = nn.Parameter(
                torch.randn(size=size, device=_non_meta_init_device(self.config), dtype=torch.float),
                requires_grad=self.config.fourier_learnable
            )
            self.cos_coef = nn.Parameter(
                torch.randn(size=size, device=_non_meta_init_device(self.config), dtype=torch.float),
                requires_grad=self.config.fourier_learnable
            )
        else:
            # 正弦和余弦共享同一个系数矩阵
            self.fourier_coef = nn.Parameter(
                torch.randn(size=size, device=_non_meta_init_device(self.config), dtype=torch.float),
                requires_grad=self.config.fourier_learnable
            )
        
        self.reset_parameters()
    
    def apply_rotary_pos_emb(self, pos_sin, pos_cos, t, inverse = False):
        # FoPE 中的旋转位置编码应用：先通过僅里叶系数投影，再应用旋转
        # 通过僅里叶系数矩阵将 RoPE 频率投影到目标维度
        if self.config.fourier_separate_basis:
            # 正弦和余弦分开的系数矩阵
            if self.config.fourier_norm:
                # 对系数矩阵进行归一化（按输入维度求和）
                fourier_sin = torch.einsum(f"{self.input_shape}, {self.coef_shape} -> {self.output_shape}", pos_sin, self.sin_coef / self.sin_coef.sum(dim=-2, keepdim=True))
                fourier_cos = torch.einsum(f"{self.input_shape}, {self.coef_shape} -> {self.output_shape}", pos_cos, self.cos_coef / self.cos_coef.sum(dim=-2, keepdim=True))
            else:
                # 直接使用系数矩阵
                fourier_sin = torch.einsum(f"{self.input_shape}, {self.coef_shape} -> {self.output_shape}", pos_sin, self.sin_coef)
                fourier_cos = torch.einsum(f"{self.input_shape}, {self.coef_shape} -> {self.output_shape}", pos_cos, self.cos_coef)
        else:
            # 正弦和余弦共享同一个系数矩阵
            if self.config.fourier_norm:
                # 对系数矩阵进行归一化
                fourier_sin = torch.einsum(f"{self.input_shape}, {self.coef_shape} -> {self.output_shape}", pos_sin, self.fourier_coef / self.fourier_coef.sum(dim=-2, keepdim=True))
                fourier_cos = torch.einsum(f"{self.input_shape}, {self.coef_shape} -> {self.output_shape}", pos_cos, self.fourier_coef / self.fourier_coef.sum(dim=-2, keepdim=True))
            else:
                # 直接使用系数矩阵
                fourier_sin = torch.einsum(f"{self.input_shape}, {self.coef_shape} -> {self.output_shape}", pos_sin, self.fourier_coef)
                fourier_cos = torch.einsum(f"{self.input_shape}, {self.coef_shape} -> {self.output_shape}", pos_cos, self.fourier_coef)
        
        # 如果忽略零频率，需要在高维度上补零以匹配头维度
        if self.config.fourier_ignore_zero:
            fourier_sin = F.pad(input=fourier_sin, pad=(0, self.head_dim//2-fourier_sin.size(-1)), mode="constant", value=1)
            fourier_cos = F.pad(input=fourier_cos, pad=(0, self.head_dim//2-fourier_cos.size(-1)), mode="constant", value=1)
        
        # 将频率重复一次以匹配完整的头维度（因为旋转是成对进行的）
        fourier_sin = torch.cat((fourier_sin, fourier_sin), dim=-1)
        fourier_cos = torch.cat((fourier_cos, fourier_cos), dim=-1)
        
        # 使用僅里叶变换后的频率进行旋转
        if not inverse:
            # 正向旋转：使用 FoPE 投影后的频率
            return ((t * fourier_cos) + (self.rotate_half(t) * fourier_sin)).to(t.dtype)
        else:
            # 逆向旋转：使用 FoPE 投影后的频率
            return ((t * fourier_cos) - (self.rotate_half(t) * fourier_sin)).to(t.dtype)
    
    def get_step_eye(self, _param):
        # 生成步长单位矩阵：当输入维度 > 输出维度时使用
        # 这种初始化方式在输入维度上按步长采样，实现降维效果
        _param = torch.zeros_like(_param)
        
        step = math.ceil(self.input_dim / self.output_dim)  # 计算采样步长
        for i in range(self.output_dim):
            if i*step < self.input_dim:
                _param[..., i*step, i] = 1.0  # 在对角线上按步长设置 1
        
        return _param
    
    def reset_parameters(self):
        # 重置僅里叶系数参数，支持多种初始化策略
        with torch.no_grad():
            if self.config.fourier_separate_basis:
                if self.config.fourier_init == "eye":
                    if self.input_dim == self.output_dim:
                        if self.prefix == "attn" and self.config.fourier_separate_head:
                            for i in range(self.sin_coef.size(0)):
                                torch.nn.init.eye_(self.sin_coef[i]) 
                                torch.nn.init.eye_(self.cos_coef[i])
                        else:
                            torch.nn.init.eye_(self.sin_coef)
                            torch.nn.init.eye_(self.cos_coef)
                    else:   
                        self.sin_coef.data = self.get_step_eye(self.sin_coef)
                        self.cos_coef.data = self.get_step_eye(self.cos_coef)
                        
                elif self.config.fourier_init == "eye_norm":
                    torch.nn.init.normal_(self.sin_coef, std=self.config.fourier_init_norm_gain)
                    torch.nn.init.normal_(self.cos_coef, std=self.config.fourier_init_norm_gain)
                    
                    if self.input_dim == self.output_dim:
                        self.sin_coef += torch.eye(self.input_dim, device=self.sin_coef.device)
                        self.cos_coef += torch.eye(self.input_dim, device=self.cos_coef.device)
                    else:
                        self.sin_coef += self.get_step_eye(self.sin_coef)
                        self.cos_coef += self.get_step_eye(self.cos_coef)
                    
                elif self.config.fourier_init == "eye_xavier_norm":
                    torch.nn.init.xavier_normal_(self.sin_coef, gain=self.config.fourier_init_norm_gain)
                    torch.nn.init.xavier_normal_(self.cos_coef, gain=self.config.fourier_init_norm_gain)
                    
                    if self.input_dim == self.output_dim:    
                        self.sin_coef += torch.eye(self.input_dim, device=self.sin_coef.device)
                        self.cos_coef += torch.eye(self.input_dim, device=self.cos_coef.device)
                    else:
                        self.sin_coef += self.get_step_eye(self.sin_coef)
                        self.cos_coef += self.get_step_eye(self.cos_coef)
                
                elif self.config.fourier_init == "eye_xavier_uniform":
                    torch.nn.init.xavier_uniform_(self.sin_coef, gain=self.config.fourier_init_norm_gain)
                    torch.nn.init.xavier_uniform_(self.cos_coef, gain=self.config.fourier_init_norm_gain)
                    
                    if self.input_dim == self.output_dim:
                        self.sin_coef += torch.eye(self.input_dim, device=self.sin_coef.device)
                        self.cos_coef += torch.eye(self.input_dim, device=self.cos_coef.device)
                    else:
                        self.sin_coef += self.get_step_eye(self.sin_coef)
                        self.cos_coef += self.get_step_eye(self.cos_coef)
                
                elif self.config.fourier_init == "xavier_norm":
                    torch.nn.init.xavier_normal_(self.sin_coef)
                    torch.nn.init.xavier_normal_(self.cos_coef)
                elif self.config.fourier_init == "xavier_uniform":
                    torch.nn.init.xavier_uniform_(self.sin_coef)
                    torch.nn.init.xavier_uniform_(self.cos_coef)
                else:
                    raise ValueError(f"Unsupported init method: {self.config.fourier_init}")
            else:
                if self.config.fourier_init == "eye":
                    
                    if self.input_dim == self.output_dim:
                        if self.prefix == "attn" and self.config.fourier_separate_head:
                            for i in range(self.fourier_coef.size(0)):
                                torch.nn.init.eye_(self.fourier_coef[i])
                        else:
                            torch.nn.init.eye_(self.fourier_coef)
                    else:
                            
                        self.fourier_coef.data = self.get_step_eye(self.fourier_coef)
                        
                elif self.config.fourier_init == "eye_norm":
                    torch.nn.init.normal_(self.fourier_coef, std=self.config.fourier_init_norm_gain)
                    
                    if self.input_dim == self.output_dim:    
                        self.fourier_coef += torch.eye(self.input_dim, device=self.fourier_coef.device)
                    else:
                        self.fourier_coef += self.get_step_eye(self.fourier_coef)
                    
                elif self.config.fourier_init == "eye_xavier_norm":
                    torch.nn.init.xavier_normal_(self.fourier_coef, gain=self.config.fourier_init_norm_gain)
                    
                    if self.input_dim == self.output_dim:
                        self.fourier_coef += torch.eye(self.input_dim, device=self.fourier_coef.device)
                    else:
                        self.fourier_coef += self.get_step_eye(self.fourier_coef)
                    
                elif self.config.fourier_init == "eye_xavier_uniform":
                    torch.nn.init.xavier_uniform_(self.fourier_coef, gain=self.config.fourier_init_norm_gain) 
                    
                    if self.input_dim == self.output_dim:
                        self.fourier_coef += torch.eye(self.input_dim, device=self.fourier_coef.device)
                    else:
                        self.fourier_coef += self.get_step_eye(self.fourier_coef)
                            
                elif self.config.fourier_init == "xavier_norm":
                    torch.nn.init.xavier_normal_(self.fourier_coef)
                elif self.config.fourier_init == "xavier_uniform":
                    torch.nn.init.xavier_uniform_(self.fourier_coef)
                else:
                    raise ValueError(f"Unsupported init method: {self.config.fourier_init}")

    def __repr__(self):
        return f"{self.__class__.__name__}(fourier_dim={self.input_dim})"

class InverseFourierEmbedding(FourierEmbedding):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.suffix = "ifourier"
        
    def forward(self, x, all_len, layer_idx=None, use_rope_cache=None):
        return super().forward(x, all_len, layer_idx=layer_idx, use_rope_cache=use_rope_cache, inverse=True)

class Activation(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @property
    @abstractmethod
    def output_multiplier(self) -> float:
        raise NotImplementedError

    @classmethod
    def build(cls, config: ModelConfig) -> Activation:
        if config.activation_type == ActivationType.gelu:
            return cast(Activation, GELU(approximate="none"))
        elif config.activation_type == ActivationType.relu:
            return cast(Activation, ReLU(inplace=False))
        elif config.activation_type == ActivationType.swiglu:
            return SwiGLU(config)
        else:
            raise NotImplementedError(f"Unknown activation: '{config.activation_type}'")


class GELU(nn.GELU):
    @property
    def output_multiplier(self) -> float:
        return 1.0


class ReLU(nn.ReLU):
    @property
    def output_multiplier(self) -> float:
        return 1.0


class SwiGLU(Activation):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x

    @property
    def output_multiplier(self) -> float:
        return 0.5


def causal_attention_bias(seq_len: int, device: torch.device) -> torch.FloatTensor:
    att_bias = torch.triu(
        torch.ones(seq_len, seq_len, device=device, dtype=torch.float),
        diagonal=1,
    )
    att_bias.masked_fill_(att_bias == 1, torch.finfo(att_bias.dtype).min)
    return att_bias.view(1, 1, seq_len, seq_len)  # type: ignore


def get_causal_attention_bias(cache: BufferCache, seq_len: int, device: torch.device) -> torch.Tensor:
    if (causal_bias := cache.get("causal_attention_bias")) is not None and causal_bias.shape[-1] >= seq_len:
        if causal_bias.device != device:
            causal_bias = causal_bias.to(device)
            cache["causal_attention_bias"] = causal_bias
        return causal_bias
    with torch.autocast(device.type, enabled=False):
        causal_bias = causal_attention_bias(seq_len, device)
    cache["causal_attention_bias"] = causal_bias
    return causal_bias


def alibi_attention_bias(seq_len: int, config: ModelConfig, device: torch.device) -> torch.FloatTensor:
    alibi_bias = torch.arange(1 - seq_len, 1, dtype=torch.float, device=device).view(1, 1, 1, seq_len)

    # shape: (1, 1, seq_len, seq_len)
    alibi_bias = alibi_bias - torch.arange(1 - seq_len, 1, dtype=torch.float, device=device).view(1, 1, seq_len, 1)
    alibi_bias.abs_().mul_(-1)

    # shape: (n_heads,)
    m = torch.arange(1, config.n_heads + 1, dtype=torch.float, device=device)
    m.mul_(config.alibi_bias_max / config.n_heads)

    # shape: (1, n_heads, seq_len, seq_len)
    return alibi_bias * (1.0 / (2 ** m.view(1, config.n_heads, 1, 1)))  # type: ignore

class OLMoBlock(nn.Module):
    """
    A base class for transformer block implementations.
    """

    def __init__(self, layer_id: int, config: ModelConfig, cache: BufferCache):
        super().__init__()
        self.layer_id = layer_id
        self.config = config
        self.hidden_size = (
            config.mlp_hidden_size if config.mlp_hidden_size is not None else config.mlp_ratio * config.d_model
        )
        self.__cache = cache
        assert config.d_model % config.n_heads == 0

        self._activation_checkpoint_fn = None

        # Dropout.
        self.dropout = Dropout(config.residual_dropout)

        # Layer norms.
        self.k_norm: Optional[LayerNormBase] = None
        self.q_norm: Optional[LayerNormBase] = None
        self.v_norm: Optional[LayerNormBase] = None
        self.out_norm: Optional[LayerNormBase] = None
        if config.attention_layer_norm:
            assert config.effective_n_kv_heads is not None
            self.k_norm = LayerNormBase.build(
                config,
                size=(config.d_model // config.n_heads) * config.effective_n_kv_heads,
                elementwise_affine=config.attention_layer_norm_with_affine,
            )
            self.q_norm = LayerNormBase.build(config, elementwise_affine=config.attention_layer_norm_with_affine)

        
        if config.attention_layer_norm_v:
            self.v_norm = LayerNormBase.build(
                config,
                size=(config.d_model // config.n_heads) * config.effective_n_kv_heads,
                elementwise_affine=config.attention_layer_norm_with_affine,
            )
        
        if config.attention_layer_norm_out:
            self.out_norm = LayerNormBase.build(
                config,
                size=config.d_model,
                elementwise_affine=config.attention_layer_norm_with_affine,
            )
            
        # Make sure QKV clip coefficient is positive, otherwise it's not well-defined.
        if config.clip_qkv is not None:
            assert config.clip_qkv > 0

        # Activation function.
        self.act = Activation.build(config)
        assert (self.act.output_multiplier * self.hidden_size) % 1 == 0

        self.attn_out = nn.Linear(
            config.d_model, config.d_model, bias=config.include_bias, device=config.init_device
        )
            
        # Feed-forward output projection.
        self.ff_out = nn.Linear(
            int(self.act.output_multiplier * self.hidden_size),
            config.d_model,
            bias=config.include_bias,
            device=config.init_device,
        )
        self.ff_out._is_residual = True  

        if self.config.rope:
            if self.config.fourier:
                self.pos_emb = FourierEmbedding(config, self.__cache)
            else:
                self.pos_emb = RotaryEmbedding(config, self.__cache)
        
        self.flash_attn_func = None
        self.flash_attn_varlen_func = None
        if config.flash_attention:
            try:
                from flash_attn import (  # type: ignore
                    flash_attn_func,
                    flash_attn_varlen_func,
                )

                self.flash_attn_func = flash_attn_func
                self.flash_attn_varlen_func = flash_attn_varlen_func
            except ModuleNotFoundError:
                pass

    def reset_parameters(self):
        if self.k_norm is not None:
            self.k_norm.reset_parameters()
        if self.q_norm is not None:
            self.q_norm.reset_parameters()

        if self.config.init_fn == InitFnType.normal:
            attn_out_std = ff_out_std = self.config.init_std
            cutoff_factor = self.config.init_cutoff_factor

        elif self.config.init_fn == InitFnType.mitchell:
            attn_out_std = 1 / (math.sqrt(2 * self.config.d_model * (self.layer_id + 1)))
            ff_out_std = 1 / (math.sqrt(2 * self.ff_out.in_features * (self.layer_id + 1)))
            cutoff_factor = self.config.init_cutoff_factor or 3.0

        elif self.config.init_fn == InitFnType.full_megatron:
            attn_out_std = ff_out_std = self.config.init_std / math.sqrt(2.0 * self.config.n_layers)
            cutoff_factor = self.config.init_cutoff_factor or 3.0

        else:
            raise NotImplementedError(self.config.init_fn)

        init_normal(self.attn_out, std=attn_out_std, init_cutoff_factor=cutoff_factor)
            
        init_normal(self.ff_out, std=ff_out_std, init_cutoff_factor=cutoff_factor)

    def set_activation_checkpointing(self, strategy: Optional[ActivationCheckpointingStrategy]):
        if strategy == ActivationCheckpointingStrategy.fine_grained:
            self._activation_checkpoint_fn = activation_checkpoint_function(self.config)
        else:
            self._activation_checkpoint_fn = None

    @classmethod
    def _cast_attn_bias(cls, bias: torch.Tensor, input_dtype: torch.dtype) -> torch.Tensor:
        target_dtype = input_dtype
        # NOTE: `is_autocast_enabled()` only checks for CUDA autocast, so we use the separate function
        # `is_autocast_cpu_enabled()` for CPU autocast.
        # See https://github.com/pytorch/pytorch/issues/110966.
        if bias.device.type == "cuda" and torch.is_autocast_enabled():
            target_dtype = torch.get_autocast_gpu_dtype()
        elif bias.device.type == "cpu" and torch.is_autocast_cpu_enabled():
            target_dtype = torch.get_autocast_cpu_dtype()
        if bias.dtype != target_dtype:
            bias = bias.to(target_dtype)
            ensure_finite_(bias, check_neg_inf=True, check_pos_inf=False)
        return bias

    def _scaled_dot_product_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        dropout_p: float = 0.0,
        is_causal: bool = False,
        max_doc_len: Optional[int] = None,
        cu_doc_lens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Computes scaled dot product attention on query, key and value tensors, using an optional
        attention mask if passed, and applying dropout if a probability greater than 0.0 is specified.
        """
        if max_doc_len is not None and cu_doc_lens is not None:
            assert self.flash_attn_varlen_func is not None, "flash-attn is required for document masking"
            assert attn_mask is None, "attn-mask is currently not supported with document masking"
            B, T, D = q.size(0), q.size(2), q.size(3)
            r = self.flash_attn_varlen_func(
                q.transpose(1, 2).view(B * T, -1, D),
                k.transpose(1, 2).view(B * T, -1, D),
                v.transpose(1, 2).view(B * T, -1, D),
                cu_doc_lens,
                cu_doc_lens,
                max_doc_len,
                max_doc_len,
                dropout_p=dropout_p,
                causal=is_causal,
            )
            return r.view(B, T, -1, D).transpose(1, 2)
        elif self.flash_attn_func is not None and attn_mask is None:
            # q, k, v = q.to(torch.float16), k.to(torch.float16), v.to(torch.float16)
            r = self.flash_attn_func(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), dropout_p=dropout_p, causal=is_causal
            )
            return r.transpose(1, 2)
        else:
            # torch's sdpa doesn't support GQA, so we're doing this
            assert k.size(1) == v.size(1)
            num_kv_heads = k.size(1)
            num_q_heads = q.size(1)
            if num_q_heads != num_kv_heads:
                assert num_q_heads % num_kv_heads == 0
                k = k.repeat_interleave(num_q_heads // num_kv_heads, dim=1, output_size=num_q_heads)
                v = v.repeat_interleave(num_q_heads // num_kv_heads, dim=1, output_size=num_q_heads)

            return F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask,
                dropout_p=dropout_p,
                is_causal=is_causal,
            )
    
    def attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_bias: Optional[torch.Tensor] = None,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        use_rope_cache: bool = None,
        max_doc_len: Optional[int] = None,
        cu_doc_lens: Optional[torch.Tensor] = None,
        layer_idx: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        B, T, C = q.size()  # batch size, sequence length, d_model
        dtype = k.dtype

        # Optionally apply layer norm to keys and queries.
        if self.q_norm is not None and self.k_norm is not None:
            q = self.q_norm(q).to(dtype=dtype)
            k = self.k_norm(k).to(dtype=dtype)
       
        if self.v_norm is not None:
            v = self.v_norm(v).to(dtype=dtype)
            
        # Move head forward to be next to the batch dim.
        # shape: (B, nh, T, hs)
        q = q.view(B, T, self.config.n_heads, C // self.config.n_heads).transpose(1, 2)
        # shape: (B, n_kv_h, T, hs)
        k = k.view(B, T, self.config.effective_n_kv_heads, C // self.config.n_heads).transpose(1, 2)
        # shape: (B, n_kv_h, T, hs)
        v = v.view(B, T, self.config.effective_n_kv_heads, C // self.config.n_heads).transpose(1, 2)

        if layer_past is not None:
            past_key, past_value = layer_past
            k = torch.cat((past_key, k), dim=-2)
            v = torch.cat((past_value, v), dim=-2)

        present = (k, v) if use_cache else None
        query_len, key_len = q.shape[-2], k.shape[-2]  # could be different if layer_past not None

        all_len = max(query_len, key_len)
        
        if self.config.pos_emb and (self.config.rope or self.config.fourier):
            q = self.pos_emb(q, all_len, layer_idx=layer_idx, use_rope_cache=use_rope_cache)
            k = self.pos_emb(k, all_len, layer_idx=layer_idx, use_rope_cache=use_rope_cache)

        if attention_bias is not None:
            # Resize and cast attention bias.
            # The current dtype of the attention bias might not match the dtype that the SDP attn function will
            # run in if AMP is enabled, and this can be a problem if some tokens are masked out due to padding
            # as down-casting the attention bias to the autocast precision will result in -infs, which will
            # cause the SDP attn function to produce NaNs.
            attention_bias = self._cast_attn_bias(
                attention_bias[:, :, key_len - query_len : key_len, :key_len], dtype
            )

        # Get the attention scores.
        att = self._scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attention_bias,
            dropout_p=0.0 if not self.training else self.config.attention_dropout,
            is_causal=attention_bias is None,
            max_doc_len=max_doc_len,
            cu_doc_lens=cu_doc_lens,
        ) # shape: (B, nh, T, hs)
        
        if self.out_norm is not None:
            att = self.out_norm(att)
        
        # Re-assemble all head outputs side-by-side.
        att = att.transpose(1, 2).contiguous().view(B, T, C)
        # Apply output projection.
        att = self.attn_out(att) # shape: (B, T, C)
                
        
        return att, present
        
    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        attention_bias: Optional[torch.FloatTensor] = None,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        use_rope_cache: bool = None,
        max_doc_len: Optional[int] = None,
        cu_doc_lens: Optional[torch.Tensor] = None,
        layer_idx: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        raise NotImplementedError

    @classmethod
    def build(cls, layer_id: int, config: ModelConfig, cache: BufferCache) -> OLMoBlock:
        if config.block_type == BlockType.sequential:
            return OLMoSequentialBlock(layer_id, config, cache)
        elif config.block_type == BlockType.llama:
            return OLMoLlamaBlock(layer_id, config, cache)
        else:
            raise NotImplementedError(f"Unknown block type: '{config.block_type}'")


class OLMoSequentialBlock(OLMoBlock):
    """
    This is a typical transformer block where the output is computed as ``MLP(LN(x + Attention(LN(x))))``
    (plus another skip connection). To compute it as ``LN(MLP(x + LN(Attention(x))))``,
    use the flag `norm_after`.
    """

    def __init__(self, layer_id: int, config: ModelConfig, cache: BufferCache):
        super().__init__(layer_id, config, cache)
        # Attention input projection. Projects x -> (q, k, v)
        
        head_dim = config.d_model // config.n_heads
        self.fused_dims = (
            config.d_model,
            config.effective_n_kv_heads * head_dim,
            config.effective_n_kv_heads * head_dim,
        )
        self.att_proj = nn.Linear(
            config.d_model, sum(self.fused_dims), bias=config.include_bias, device=config.init_device
        )
            
        # Feed-forward input projection.
        self.ff_proj = nn.Linear(
            config.d_model, self.hidden_size, bias=config.include_bias, device=config.init_device
        )

        # Layer norms.
        self.attn_norm = LayerNorm.build(config, size=config.d_model)
        self.ff_norm = LayerNorm.build(config, size=config.d_model)

    def reset_parameters(self):
        super().reset_parameters()
        self.attn_norm.reset_parameters()
        self.ff_norm.reset_parameters()
        # NOTE: the standard deviation for these weights does not depend on the layer.

        if self.config.init_fn == InitFnType.normal:
            std = self.config.init_std
            cutoff_factor = self.config.init_cutoff_factor
        elif self.config.init_fn == InitFnType.mitchell:
            std = 1 / math.sqrt(self.config.d_model)
            cutoff_factor = self.config.init_cutoff_factor or 3.0
        elif self.config.init_fn == InitFnType.full_megatron:
            std = self.config.init_std
            cutoff_factor = self.config.init_cutoff_factor or 3.0
        else:
            raise NotImplementedError(self.config.init_fn)

        init_normal(self.att_proj, std, cutoff_factor)
        init_normal(self.ff_proj, std, cutoff_factor)

    def forward(
        self,
        x: torch.Tensor,
        attention_bias: Optional[torch.Tensor] = None,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        use_rope_cache: bool = None,
        max_doc_len: Optional[int] = None,
        cu_doc_lens: Optional[torch.Tensor] = None,
        layer_idx: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        # Get query, key, value projections.
        # shape:
        #  - for regular attn q, k, v: (batch_size, seq_len, d_model)
        #  - for multi-query attn q: (batch_size, seq_len, d_model)
        #                      k, v: (batch_size, seq_len, d_model // n_heads)
        #  - for group query attn q: (batch_size, seq_len, d_model)
        #                      k, v: (batch_size, seq_len, d_model // n_kv_heads)

        # apply norm before
        if not self.config.norm_after:
            if self._activation_checkpoint_fn is not None:
                qkv = self._activation_checkpoint_fn(self.attn_norm, x)
            else:
                qkv = self.attn_norm(x) 
                
        qkv = self.att_proj(qkv) # shape: (batch_size, seq_len, d_model x 3)
        
        if self.config.clip_qkv is not None:
            qkv.clamp_(min=-self.config.clip_qkv, max=self.config.clip_qkv)

        q, k, v = qkv.split(self.fused_dims, dim=-1)

        # Get attention scores.
        if self._activation_checkpoint_fn is not None:
            att, cache = self._activation_checkpoint_fn(  # type: ignore
                self.attention,
                q,
                k,
                v,
                attention_bias,
                layer_past=layer_past,
                use_cache=use_cache,
                max_doc_len=max_doc_len,
                cu_doc_lens=cu_doc_lens,
            )
        else:
            att, cache = self.attention(
                q,
                k,
                v,
                attention_bias,
                layer_past=layer_past,
                use_cache=use_cache,
                use_rope_cache=use_rope_cache,
                max_doc_len=max_doc_len,
                cu_doc_lens=cu_doc_lens,
                layer_idx=layer_idx,
            )

        if self.config.norm_after:
            if self._activation_checkpoint_fn is not None:
                att = self._activation_checkpoint_fn(self.attn_norm, att)
            else:
                att = self.attn_norm(att)

        # Add attention scores.
        # shape: (B, T, C)
        x = x + self.dropout(att)

        # Add feed-forward projection.
        # shape: (batch_size, seq_len, d_model)
        og_x = x

        if not self.config.norm_after:
            if self._activation_checkpoint_fn is not None:
                x = self._activation_checkpoint_fn(self.ff_norm, x)  # type: ignore
            else:
                x = self.ff_norm(x)
        
        x = self.ff_proj(x) # shape: (batch_size, seq_len, mlp_hidden_size or mlp_ratio x d_model)
        
        if self._activation_checkpoint_fn is not None:
            x = self._activation_checkpoint_fn(self.act, x)  # type: ignore
        else:
            x = self.act(x)
        
        x = self.ff_out(x) # shape: (batch_size, seq_len, d_model)

        if self.config.norm_after:
            if self._activation_checkpoint_fn is not None:
                x = self._activation_checkpoint_fn(self.ff_norm, x)  # type: ignore
            else:
                x = self.ff_norm(x)

        x = self.dropout(x)
        x = og_x + x

        return x, cache


class OLMoLlamaBlock(OLMoBlock):
    """
    This is a transformer block where the output is computed as ``MLP(LN(x + Attention(LN(x))))``
    (plus another skip connection). This block is similar to `OLMoSequentialBlock`
    but some operations have slightly different implementations to imitate the
    behavior of Llama.
    """

    def __init__(self, layer_id: int, config: ModelConfig, cache: BufferCache):
        super().__init__(layer_id, config, cache)
        # Layer norms.
        self.attn_norm = LayerNorm.build(config)
        self.ff_norm = LayerNorm.build(config)
        self.__cache = cache

        # Attention input projection. Projects x -> (q, k, v)
        if config.multi_query_attention:
            q_proj_out_dim = config.d_model
            k_proj_out_dim = config.d_model // config.n_heads
            v_proj_out_dim = config.d_model // config.n_heads
        else:
            q_proj_out_dim = config.d_model
            k_proj_out_dim = config.d_model
            v_proj_out_dim = config.d_model
        
        self.q_proj = nn.Linear(
            config.d_model, q_proj_out_dim, bias=config.include_bias, device=config.init_device
        )
        self.k_proj = nn.Linear(
            config.d_model, k_proj_out_dim, bias=config.include_bias, device=config.init_device
        )
        self.v_proj = nn.Linear(
            config.d_model, v_proj_out_dim, bias=config.include_bias, device=config.init_device
        )

        # Feed-forward input projection.
        self.ff_proj = nn.Linear(
            config.d_model, self.hidden_size, bias=config.include_bias, device=config.init_device
        )

    def reset_parameters(self):
        super().reset_parameters()
        self.attn_norm.reset_parameters()
        self.ff_norm.reset_parameters()
        # NOTE: the standard deviation for these weights does not depend on the layer.

        if self.config.init_fn == InitFnType.normal:
            std = self.config.init_std
            cutoff_factor = self.config.init_cutoff_factor
        elif self.config.init_fn == InitFnType.mitchell:
            std = 1 / math.sqrt(self.config.d_model)
            cutoff_factor = self.config.init_cutoff_factor or 3.0
        elif self.config.init_fn == InitFnType.full_megatron:
            std = self.config.init_std
            cutoff_factor = self.config.init_cutoff_factor or 3.0
        else:
            raise NotImplementedError(self.config.init_fn)

        init_normal(self.q_proj, std, cutoff_factor)
        init_normal(self.k_proj, std, cutoff_factor)
        init_normal(self.v_proj, std, cutoff_factor)
        init_normal(self.ff_proj, std, cutoff_factor)

    def _scaled_dot_product_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        dropout_p: float = 0.0,
        is_causal: bool = False,
        max_doc_len: Optional[int] = None,
        cu_doc_lens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if max_doc_len is not None or cu_doc_lens is not None:
            raise NotImplementedError(
                f"attention document masking is not implemented for {self.__class__.__name__}"
            )

        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))

        if is_causal:
            assert attn_mask is None

            query_len, key_len = q.shape[-2], k.shape[-2]  # could be different if layer_past not None
            attn_bias = get_causal_attention_bias(self.__cache, key_len, q.device)[:, :, :query_len, :key_len]
        elif attn_mask is not None:
            attn_bias = attn_mask.to(q.dtype)
        else:
            attn_bias = torch.zeros_like(attn_weights)

        attn_weights += attn_bias
        attn_weights = nn.functional.softmax(attn_weights, dim=-1).to(q.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=dropout_p)
        return torch.matmul(attn_weights, v)

    def forward(
        self,
        x: torch.Tensor,
        attention_bias: Optional[torch.Tensor] = None,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        max_doc_len: Optional[int] = None,
        cu_doc_lens: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        # Get query, key, value projections.
        # shape:
        #  - for regular attn q, k, v: (batch_size, seq_len, d_model)
        #  - for multi-query attn q: (batch_size, seq_len, d_model)
        #                      k, v: (batch_size, seq_len, d_model // n_heads)
        x_normed = self.attn_norm(x)
        q = self.q_proj(x_normed)
        k = self.k_proj(x_normed)
        v = self.v_proj(x_normed)

        if self.config.clip_qkv is not None:
            q.clamp_(min=-self.config.clip_qkv, max=self.config.clip_qkv)
            k.clamp_(min=-self.config.clip_qkv, max=self.config.clip_qkv)
            v.clamp_(min=-self.config.clip_qkv, max=self.config.clip_qkv)

        # Get attention scores.
        att, cache = self.attention(
            q,
            k,
            v,
            attention_bias,
            layer_past=layer_past,
            use_cache=use_cache,
            max_doc_len=max_doc_len,
            cu_doc_lens=cu_doc_lens,
        )

        # Add attention scores.
        # shape: (B, T, C)
        x = x + self.dropout(att)

        # Add feed-forward projection.
        # shape: (batch_size, seq_len, d_model)
        og_x = x
        if self._activation_checkpoint_fn is not None:
            x = self._activation_checkpoint_fn(self.ff_norm, x)  # type: ignore
        else:
            x = self.ff_norm(x)
        x = self.ff_proj(x)
        if self._activation_checkpoint_fn is not None:
            x = self._activation_checkpoint_fn(self.act, x)  # type: ignore
        else:
            x = self.act(x)
        x = self.ff_out(x)
        x = self.dropout(x)
        x = og_x + x

        return x, cache


class OLMoOutput(NamedTuple):
    logits: torch.FloatTensor
    """
    A tensor of shape `(batch_size, seq_len, vocab_size)` representing the log probabilities
    for the next token *before* normalization via (log) softmax.
    """

    attn_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]]
    """
    Attention keys and values from each block.
    """

    hidden_states: Optional[Tuple[torch.Tensor]]
    """
    Hidden states from each block.
    """


class OLMoGenerateOutput(NamedTuple):
    token_ids: torch.LongTensor
    """
    The generated token IDs, a tensor of shape `(batch_size, beam_size, max_steps)`.
    These do *not* include the original input IDs.
    """

    scores: torch.FloatTensor
    """
    The scores of the generated sequences, a tensor of shape `(batch_size, beam_size)`.
    """


class OLMoBlockGroup(nn.ModuleList):
    def __init__(self, config: ModelConfig, layer_offset: int, modules: Optional[Iterable[nn.Module]] = None):
        super().__init__(modules)
        self.config = config
        self.layer_offset = layer_offset
        self.activation_checkpointing_strategy: Optional[ActivationCheckpointingStrategy] = None
        self._activation_checkpoint_fn = activation_checkpoint_function(self.config)

    def forward(
        self,
        x: torch.Tensor,
        attention_bias: Optional[torch.FloatTensor] = None,
        layers_past: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        max_doc_len: Optional[int] = None,
        cu_doc_lens: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[List[Tuple[torch.Tensor, torch.Tensor]]]]:
        attn_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = [] if use_cache else None
        for block_idx, block in enumerate(self):
            layer_past = None if layers_past is None else layers_past[block_idx]
            block_idx += self.layer_offset
            if should_checkpoint_block(self.activation_checkpointing_strategy, block_idx):
                # shape: (batch_size, seq_len, d_model)
                x, cache = self._activation_checkpoint_fn(  # type: ignore
                    block,
                    x,
                    attention_bias=attention_bias,
                    layer_past=layer_past,
                    use_cache=use_cache,
                    max_doc_len=max_doc_len,
                    cu_doc_lens=cu_doc_lens,
                    layer_idx=block_idx,
                )
            else:
                # shape: (batch_size, seq_len, d_model)
                x, cache = block(
                    x,
                    attention_bias=attention_bias,
                    layer_past=layer_past,
                    use_cache=use_cache,
                    max_doc_len=max_doc_len,
                    cu_doc_lens=cu_doc_lens,
                    layer_idx=block_idx,
                )
            if attn_key_values is not None:
                assert cache is not None
                attn_key_values.append(cache)
        return x, attn_key_values

    def reset_parameters(self):
        for block in self:
            block.reset_parameters()

    def set_activation_checkpointing(self, strategy: Optional[ActivationCheckpointingStrategy]):
        self.activation_checkpointing_strategy = strategy
        for block in self:
            block.set_activation_checkpointing(strategy)


class OLMo(nn.Module):
    def __init__(self, config: ModelConfig, init_params: bool = True):
        super().__init__()
        self.config = config
        self.__cache = BufferCache()

        # Validate config.
        if self.config.alibi and self.config.flash_attention:
            raise OLMoConfigurationError("ALiBi is currently not supported with FlashAttention")

        if self.config.embedding_size is not None and self.config.embedding_size != self.config.vocab_size:
            if self.config.embedding_size < self.config.vocab_size:
                raise OLMoConfigurationError("embedding size should be at least as big as vocab size")
            elif self.config.embedding_size % 128 != 0:
                import warnings

                warnings.warn(
                    "Embedding size is not a multiple of 128! This could hurt throughput performance.", UserWarning
                )

        self.activation_checkpointing_strategy: Optional[ActivationCheckpointingStrategy] = None
        self._activation_checkpoint_fn: Callable = activation_checkpoint_function(self.config)

        if not (
            0 < self.config.block_group_size <= self.config.n_layers
            and self.config.n_layers % self.config.block_group_size == 0
        ):
            raise OLMoConfigurationError("n layers must be divisible by block group size")

        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(False)  # this is super slow so make sure torch won't use it

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(
                    config.embedding_size or config.vocab_size, config.d_model, device=config.init_device
                ),
                emb_drop=Dropout(config.embedding_dropout),
                ln_f=LayerNorm.build(config),
            )
        )

        blocks = [OLMoBlock.build(i, config, self.__cache) for i in range(config.n_layers)]
        if self.config.block_group_size > 1:
            block_groups = [
                OLMoBlockGroup(config, i, blocks[i : i + config.block_group_size])
                for i in range(0, config.n_layers, config.block_group_size)
            ]
            self.transformer.update({"block_groups": nn.ModuleList(block_groups)})
        else:
            self.transformer.update({"blocks": nn.ModuleList(blocks)})

        if self.config.pos_emb and not (self.config.alibi or self.config.rope or self.config.fourier):
            self.transformer.update(
                {"wpe": nn.Embedding(config.max_sequence_length, config.d_model, device=config.init_device)}
            )
            
        if not config.weight_tying:
            self.transformer.update(
                {
                    "ff_out": nn.Linear(
                        config.d_model,
                        config.embedding_size or config.vocab_size,
                        bias=config.include_bias,
                        device=config.init_device,
                    )
                }
            )
            
        if config.embedding_layer_norm:
            self.transformer.update({"emb_norm": LayerNorm.build(config)})

        # When `init_device="meta"` FSDP will call `reset_parameters()` to initialize weights.
        if init_params and self.config.init_device != "meta":
            self.reset_parameters()
        self.__num_fwd_flops: Optional[int] = None
        self.__num_bck_flops: Optional[int] = None

        # Warm up cache.
        if self.config.pos_emb and self.config.alibi:
            get_causal_attention_bias(self.__cache, config.max_sequence_length, _non_meta_init_device(config))
            self.get_alibi_attention_bias(config.max_sequence_length, _non_meta_init_device(config))

    def set_activation_checkpointing(self, strategy: Optional[ActivationCheckpointingStrategy]):
        self.activation_checkpointing_strategy = strategy
        if self.config.block_group_size != 1:
            for block_group in self.transformer.block_groups:
                block_group.set_activation_checkpointing(strategy)
        else:
            for block in self.transformer.blocks:
                block.set_activation_checkpointing(strategy)

    @property
    def device(self) -> torch.device:
        device: torch.device = self.transformer.wte.weight.device  # type: ignore
        if device.type == "meta":
            return _non_meta_init_device(self.config)
        else:
            return device

    def reset_parameters(self):
        log.info("Initializing model parameters...")
        # Top-level embeddings / linear layers.

        if self.config.init_fn == InitFnType.normal:
            # Note: We may potentially want to multiply the std by a factor of sqrt(d) in case of `scale_logits`
            # and `weight_tying`. However, we are currently not using either, and may need to rethink the init logic
            # if/when we do want it.
            wte_std = self.config.emb_init_std or self.config.init_std
            wte_cutoff_factor = self.config.init_cutoff_factor
        elif self.config.init_fn == InitFnType.mitchell:
            wte_std = self.config.emb_init_std or 1.0 / math.sqrt(self.config.d_model)
            wte_cutoff_factor = self.config.init_cutoff_factor or 3.0
        elif self.config.init_fn == InitFnType.full_megatron:
            wte_std = self.config.init_std
            if self.config.emb_init_std is not None:
                wte_std = self.config.emb_init_std
            elif self.config.scale_emb_init:
                wte_std *= math.sqrt(self.config.d_model)
            wte_cutoff_factor = self.config.init_cutoff_factor or 3.0
        else:
            raise NotImplementedError(self.config.init_fn)

        init_normal(self.transformer.wte, std=wte_std, init_cutoff_factor=wte_cutoff_factor)

        if hasattr(self.transformer, "wpe"):
            if self.config.init_fn == InitFnType.normal:
                wpe_std = self.config.init_std
                wpe_cutoff_factor = self.config.init_cutoff_factor
            elif self.config.init_fn == InitFnType.mitchell:
                wpe_std = 1 / math.sqrt(self.config.d_model)
                wpe_cutoff_factor = self.config.init_cutoff_factor or 3.0
            elif self.config.init_fn == InitFnType.full_megatron:
                wpe_std = self.config.init_std
                wpe_cutoff_factor = self.config.init_cutoff_factor or 3.0
            else:
                raise NotImplementedError(self.config.init_fn)

            init_normal(self.transformer.wpe, std=wpe_std, init_cutoff_factor=wpe_cutoff_factor)

        # Top-level layer norm.
        self.transformer.ln_f.reset_parameters()  # type: ignore

        # Output weights.
        if hasattr(self.transformer, "ff_out"):
            if self.config.init_fn == InitFnType.normal:
                ff_out_std = self.config.init_std
                ff_out_cutoff_factor = self.config.init_cutoff_factor
            elif self.config.init_fn == InitFnType.mitchell:
                ff_out_std = 1 / math.sqrt(self.config.d_model)
                ff_out_cutoff_factor = self.config.init_cutoff_factor or 3.0
            elif self.config.init_fn == InitFnType.full_megatron:
                ff_out_std = 1 / math.sqrt(self.config.d_model)
                ff_out_cutoff_factor = self.config.init_cutoff_factor or 3.0
            else:
                raise NotImplementedError(self.config.init_fn)

            init_normal(self.transformer.ff_out, ff_out_std, ff_out_cutoff_factor)

        # Let the blocks handle themselves.
        if self.config.block_group_size == 1:
            for block in self.transformer.blocks:
                block.reset_parameters()
        else:
            for block_group in self.transformer.block_groups:
                block_group.reset_parameters()

    def get_alibi_attention_bias(self, seq_len: int, device: torch.device) -> torch.Tensor:
        if (alibi_bias := self.__cache.get("alibi_attention_bias")) is not None and alibi_bias.shape[
            -1
        ] >= seq_len:
            if alibi_bias.device != device:
                alibi_bias = alibi_bias.to(device)
                self.__cache["alibi_attention_bias"] = alibi_bias
            return alibi_bias
        with torch.autocast(device.type, enabled=False):
            alibi_bias = alibi_attention_bias(seq_len, self.config, device)
        self.__cache["alibi_attention_bias"] = alibi_bias
        return alibi_bias

    def forward(
        self,
        input_ids: torch.LongTensor,
        input_embeddings: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        attention_bias: Optional[torch.Tensor] = None,
        past_key_values: Optional[Sequence[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        use_rope_cache: bool = None,
        last_logits_only: bool = False,
        output_hidden_states: Optional[bool] = None,
        doc_lens: Optional[torch.Tensor] = None,
        max_doc_lens: Optional[Sequence[int]] = None,
    ) -> OLMoOutput:
        """
        :param input_ids: A tensor of shape `(batch_size, seq_len)`.
        :param input_embeddings: A tensor of shape `(batch_size, seq_len, d_model)` with input
            embeddings. When provided, it is treated as the output of the input embedding layer.
        :param attention_mask: A tensor of shape `(batch_size, seq_len)` that indicates
            which input IDs are masked. A `1` value in the mask means that
            the corresponding input ID should *not* be ignored. A `0` means
            that the corresponding input ID is masked.

            This has the same meaning as the `attention_mask` in HuggingFace's `transformers`
            library.
        :param attention_bias: A tensor of shape `(batch_size, 1, seq_len, seq_len)`,
            `(1, 1, seq_len, seq_len)`, or `(seq_len, seq_len)`. This is used
            to introduce causal or other biases.

            If the tensor is a bool or byte tensor, a `True` or `1` at `attention_bias[:, :, i, j]`
            indicates that the i-th element in the sequence is allowed to attend to the j-th
            element in the sequence.

            If the tensor is a float tensor, it will just be added to the attention
            scores before the softmax.

            The default is causal, which corresponds to a lower-diagonal byte matrix of ones.
        :param past_key_values: Pre-computed keys and values for each attention block.
            Can be used to speed up sequential decoding. The `input_ids` which have
            their past given to this model should not be passed as `input_ids` as they have already been computed.
        :param use_cache: If `True`, return key and value tensors for each block.
        :param last_logits_only: If `True`, only compute the logits for the last token of each sequence.
            This can speed up decoding when you only care about the next token.
        :param doc_lens: Document lengths to use in attention for intra-document masking.
            Shape `(batch_size, max_docs)`.
        :param max_doc_lens: Maximum document length for each instance in the batch.
        """
        output_hidden_states = output_hidden_states if output_hidden_states is not None else False

        if past_key_values:
            assert len(past_key_values) == self.config.n_layers

        batch_size, seq_len = input_ids.size() if input_embeddings is None else input_embeddings.size()[:2]
        if past_key_values is None:
            past_length = 0
        else:
            past_length = past_key_values[0][0].size(-2)

        max_doc_len: Optional[int] = None
        cu_doc_lens: Optional[torch.Tensor] = None
        if doc_lens is not None and max_doc_lens is not None:
            max_doc_len = max(max_doc_lens)
            cu_doc_lens = get_cumulative_document_lengths(doc_lens)

        # Get embeddings of input.
        # shape: (batch_size, seq_len, d_model)
        x = self.transformer.wte(input_ids) if input_embeddings is None else input_embeddings  # type: ignore

        # Apply embedding layer norm.
        if self.config.embedding_layer_norm:
            x = self.transformer.emb_norm(x)
        
        if self.config.pos_emb and not (self.config.alibi or self.config.rope):
            # Get positional embeddings.
            # shape: (1, seq_len)
            pos = torch.arange(past_length, past_length + seq_len, dtype=torch.long, device=x.device).unsqueeze(0)
            # shape: (1, seq_len, d_model)
            pos_emb = self.transformer.wpe(pos)  # type: ignore
            x = pos_emb + x
        
        # Apply dropout.
        # shape: (batch_size, seq_len, d_model)
        x = self.transformer.emb_drop(x)  # type: ignore

        # Transform the attention mask into what the blocks expect.
        if attention_mask is not None:
            # shape: (batch_size, 1, 1, seq_len)
            attention_mask = attention_mask.to(dtype=torch.float).view(batch_size, -1)[:, None, None, :]
            attention_mask = (1.0 - attention_mask) * torch.finfo(attention_mask.dtype).min

        # Merge attention mask with attention bias.
        if (
            attention_bias is not None
            or attention_mask is not None
            or self.config.alibi
            # NOTE (epwalsh): we need to initialize the attn bias in order for attn to work properly
            # with key+value cache. Otherwise `F.scaled_dot_product_attention()` doesn't seem to compute
            # scores correctly.
            or past_key_values is not None
        ):
            if attention_bias is None and self.config.pos_emb and self.config.alibi:
                attention_bias = get_causal_attention_bias(
                    self.__cache, past_length + seq_len, x.device
                ) + self.get_alibi_attention_bias(past_length + seq_len, x.device)
            elif attention_bias is None:
                attention_bias = get_causal_attention_bias(self.__cache, past_length + seq_len, x.device)
            elif attention_bias.dtype in (torch.int8, torch.bool):
                attention_bias = attention_bias.to(dtype=torch.float)
                attention_bias.masked_fill_(attention_bias == 0.0, torch.finfo(attention_bias.dtype).min)

            # Transform to the right shape and data type.
            mask_len = seq_len
            if attention_mask is not None:
                mask_len = attention_mask.shape[-1]
            elif past_key_values is not None:
                mask_len = past_key_values[0][0].shape[-2] + seq_len
            attention_bias = attention_bias[:, :, :mask_len, :mask_len].to(dtype=torch.float)

            # Add in the masking bias.
            if attention_mask is not None:
                attention_bias = attention_bias + attention_mask
                # Might get -infs after adding attention mask, since dtype.min + dtype.min = -inf.
                # `F.scaled_dot_product_attention()` doesn't handle -inf like you'd expect, instead
                # it can produce NaNs.
                ensure_finite_(attention_bias, check_neg_inf=True, check_pos_inf=False)

        attn_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = [] if use_cache else None

        # decoder layers
        all_hidden_states = []

        # Apply blocks one-by-one.
        if self.config.block_group_size == 1:
            for block_idx, block in enumerate(self.transformer.blocks):
                if output_hidden_states:
                    # add hidden states
                    all_hidden_states.append(x)

                layer_past = None if past_key_values is None else past_key_values[block_idx]
                if should_checkpoint_block(self.activation_checkpointing_strategy, block_idx):
                    # shape: (batch_size, seq_len, d_model)
                    x, cache = self._activation_checkpoint_fn(
                        block,
                        x,
                        attention_bias=attention_bias,
                        layer_past=layer_past,
                        use_cache=use_cache,
                        max_doc_len=max_doc_len,
                        cu_doc_lens=cu_doc_lens,
                        layer_idx=block_idx,
                    )
                else:
                    # shape: (batch_size, seq_len, d_model)
                    x, cache = block(
                        x,
                        attention_bias=attention_bias,
                        layer_past=layer_past,
                        use_cache=use_cache,
                        use_rope_cache=use_rope_cache,
                        max_doc_len=max_doc_len,
                        cu_doc_lens=cu_doc_lens,
                        layer_idx=block_idx,
                    )

                if attn_key_values is not None:
                    assert cache is not None
                    attn_key_values.append(cache)
        else:
            for group_idx, block_group in enumerate(self.transformer.block_groups):
                if output_hidden_states:
                    # add hidden states
                    all_hidden_states.append(x)

                layers_past = (
                    None
                    if past_key_values is None
                    else past_key_values[
                        group_idx * self.config.block_group_size : (group_idx + 1) * self.config.block_group_size
                    ]
                )
                x, cache = block_group(
                    x,
                    attention_bias=attention_bias,
                    layers_past=layers_past,
                    use_cache=use_cache,
                    max_doc_len=max_doc_len,
                    cu_doc_lens=cu_doc_lens,
                )
                if attn_key_values is not None:
                    assert cache is not None
                    attn_key_values.extend(cache)

        if last_logits_only:
            # shape: (batch_size, 1, d_model)
            x = x[:, -1, :].unsqueeze(1)

        # Apply final layer norm.
        # shape: (batch_size, seq_len or 1, d_model)
        x = self.transformer.ln_f(x)  # type: ignore
        if output_hidden_states:
            # add final hidden state post-final-layernorm, following HuggingFace's convention
            all_hidden_states.append(x)

        # Get logits.
        # shape: (batch_size, seq_len or 1, vocab_size)
        if self.config.weight_tying:
            logits = F.linear(x, self.transformer.wte.weight, None)  # type: ignore
        else:
            logits = self.transformer.ff_out(x)  # type: ignore
        if self.config.scale_logits:
            logits.mul_(1 / math.sqrt(self.config.d_model))

        return OLMoOutput(logits=logits, attn_key_values=attn_key_values, hidden_states=tuple(all_hidden_states) if output_hidden_states else None)  # type: ignore[arg-type]

    def get_fsdp_wrap_policy(self, wrap_strategy: Optional[FSDPWrapStrategy] = None):
        if wrap_strategy is None:
            return None

        # The 'recurse' mode for the wrap function does not behave like you'd expect.
        # Even if we return False, it may still recurse because PyTorch does what it wants,
        # not what you want. This causes issues when, for example, we want to wrap 'ff_out' (a linear layer)
        # but not other linear layers within a block.
        # So we have to explicitly tell PyTorch which linear layers to wrap, and we also just
        # return True in 'recurse' mode for simplicity.
        size_based_module_to_wrap = {self.transformer.wte}
        if hasattr(self.transformer, "ff_out"):
            size_based_module_to_wrap.add(self.transformer.ff_out)

        if wrap_strategy == FSDPWrapStrategy.by_block:

            def fsdp_wrap_fn(module, recurse: bool = True, nonwrapped_numel: int = 0):
                del nonwrapped_numel
                wrap = isinstance(module, OLMoBlock)
                if recurse:
                    return True
                else:
                    return wrap

            return fsdp_wrap_fn
        elif wrap_strategy == FSDPWrapStrategy.by_block_and_size:

            def fsdp_wrap_fn(module, recurse: bool = True, nonwrapped_numel: int = 0):
                del nonwrapped_numel
                wrap = isinstance(module, (OLMoBlock,)) or module in size_based_module_to_wrap
                if recurse:
                    return True
                else:
                    return wrap

            return fsdp_wrap_fn
        elif wrap_strategy == FSDPWrapStrategy.by_block_group:
            if self.config.block_group_size <= 1:
                raise OLMoConfigurationError(
                    "'by_block_group' FSDP wrapping strategy requires block group size greater than 1"
                )

            def fsdp_wrap_fn(module, recurse: bool = True, nonwrapped_numel: int = 0):
                del nonwrapped_numel
                wrap = isinstance(module, OLMoBlockGroup)
                if recurse:
                    return True
                else:
                    return wrap

            return fsdp_wrap_fn
        elif wrap_strategy == FSDPWrapStrategy.by_block_group_and_size:
            if self.config.block_group_size <= 1:
                raise OLMoConfigurationError(
                    "'by_block_group_and_size' FSDP wrapping strategy requires block group size greater than 1"
                )

            def fsdp_wrap_fn(module, recurse: bool = True, nonwrapped_numel: int = 0):
                del nonwrapped_numel
                wrap = isinstance(module, (OLMoBlockGroup,)) or module in size_based_module_to_wrap
                if recurse:
                    return True
                else:
                    return wrap

            return fsdp_wrap_fn
        elif wrap_strategy == FSDPWrapStrategy.size_based:
            from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy

            return size_based_auto_wrap_policy
        elif wrap_strategy in {
            FSDPWrapStrategy.one_in_two,
            FSDPWrapStrategy.one_in_three,
            FSDPWrapStrategy.one_in_four,
            FSDPWrapStrategy.one_in_five,
        }:
            c = {
                FSDPWrapStrategy.one_in_two: 2,
                FSDPWrapStrategy.one_in_three: 3,
                FSDPWrapStrategy.one_in_four: 4,
                FSDPWrapStrategy.one_in_five: 5,
            }[wrap_strategy]

            def fsdp_wrap_fn(module, recurse: bool = True, nonwrapped_numel: int = 0):
                del nonwrapped_numel
                wrap = isinstance(module, OLMoBlock) and module.layer_id % c == 0
                if recurse:
                    return True
                else:
                    return wrap

            return fsdp_wrap_fn
        else:
            raise NotImplementedError(wrap_strategy)

    def num_params(self, include_embedding: bool = True) -> int:
        """
        Get the total number of parameters.
        """
        params = (np for np in self.named_parameters())
        if not include_embedding:
            params = filter(  # type: ignore
                lambda np: ".wte." not in np[0] and ".wpe." not in np[0],
                params,
            )
        return sum(p.numel() for _, p in params)

    @property
    def num_fwd_flops(self):
        if self.__num_fwd_flops:
            return self.__num_fwd_flops

        # embedding table is just a lookup in the forward pass
        n_params = self.num_params(include_embedding=False)
        # the number of parameters is approximately the number of multiply-accumulates (MAC) in the network
        # each MAC has 2 FLOPs - we multiply by 2 ie 2 * n_param
        # this gets us FLOPs / token
        params_flops_per_token = 2 * n_params
        # there are 2 FLOPS per mac; there is A=Q*K^T and out=A*V ops (ie mult by 2)
        attn_flops_per_token = (
            self.config.n_layers * 2 * 2 * (self.config.d_model * self.config.max_sequence_length)
        )
        self.__num_fwd_flops = params_flops_per_token + attn_flops_per_token
        return self.__num_fwd_flops

    @property
    def num_bck_flops(self):
        if self.__num_bck_flops:
            return self.__num_bck_flops

        n_params = self.num_params()
        params_flops_per_token = 4 * n_params
        attn_flops_per_token = self.config.n_layers * 8 * (self.config.d_model * self.config.max_sequence_length)
        self.__num_bck_flops = params_flops_per_token + attn_flops_per_token
        return self.__num_bck_flops

    def generate(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        attention_bias: Optional[torch.Tensor] = None,
        max_steps: int = 10,
        beam_size: int = 1,
        per_node_beam_size: Optional[int] = None,
        sampler: Optional[Sampler] = None,
        min_steps: Optional[int] = None,
        final_sequence_scorer: Optional[FinalSequenceScorer] = None,
        constraints: Optional[List[Constraint]] = None,
        # eos_token_id: Optional[Union[int, List[int]]] = None,
    ) -> OLMoGenerateOutput:
        """
        Generate token IDs using beam search.

        Note that by default ``beam_size`` is set to 1, which is greedy decoding.

        :param input_ids: A tensor of shape `(batch_size, seq_len)`.
        :param attention_mask: A optional tensor of shape `(batch_size, seq_len)`, the same
            as for the forward method.
        :param attention_bias: A tensor of shape
            `(batch_size, 1, seq_len + tokens_to_generate, seq_len + tokens_to_generate)`,
            the same as for the forward method except only one shape is excepted here.

        For an explanation of the other arguments, see :class:`BeamSearch`.
        """
        beam_search = BeamSearch(
            self.config.eos_token_id,
            # end_index=eos_token_id or self.config.eos_token_id,
            max_steps=max_steps,
            beam_size=beam_size,
            per_node_beam_size=per_node_beam_size,
            sampler=sampler,
            min_steps=min_steps,
            final_sequence_scorer=final_sequence_scorer,
            constraints=constraints,
        )

        # Validate inputs.
        batch_size, seq_len = input_ids.shape
        if attention_mask is not None:
            assert attention_mask.shape == (batch_size, seq_len)
        if attention_bias is not None:
            assert len(attention_bias.shape) == 4
            assert attention_bias.shape[:2] == (batch_size, 1)
            assert (
                seq_len + beam_search.max_steps
                <= attention_bias.shape[2]
                == attention_bias.shape[3]
                <= self.config.max_sequence_length
            )

        tokens_generated = 0

        def flatten_past_key_values(
            past_key_values: List[Tuple[torch.Tensor, torch.Tensor]],
        ) -> Dict[str, torch.Tensor]:
            out = {}
            for i, (key, value) in enumerate(past_key_values):
                out[f"past_key_{i}"] = key
                out[f"past_value_{i}"] = value
            return out

        def unflatten_past_key_values(
            past_key_values: Dict[str, torch.Tensor],
        ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
            out = []
            for i in range(self.config.n_layers):
                past_key = past_key_values[f"past_key_{i}"]
                past_value = past_key_values[f"past_value_{i}"]
                out.append((past_key, past_value))
            return out

        def step(
            last_predictions: torch.Tensor, state: dict[str, torch.Tensor]
        ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
            nonlocal tokens_generated

            attention_mask = state.get("attention_mask")
            attention_bias = state.get("attention_bias")

            if tokens_generated > 0:
                past_key_values = unflatten_past_key_values(state)
                input_ids = last_predictions.unsqueeze(1)
                if attention_mask is not None:
                    group_size = input_ids.shape[0]
                    attention_mask = torch.cat((attention_mask, attention_mask.new_ones((group_size, 1))), dim=-1)
            else:
                past_key_values = None
                input_ids = state["input_ids"]

            tokens_generated += 1

            # Run forward pass of model to get logits, then normalize to get log probs.
            output = self(
                input_ids,
                attention_mask=attention_mask,
                attention_bias=attention_bias,
                past_key_values=past_key_values,
                use_cache=True,
                last_logits_only=True,
            )
            log_probs = F.log_softmax(output.logits[:, -1, :], dim=-1)

            # Create new state.
            state = flatten_past_key_values(output.attn_key_values)
            if attention_mask is not None:
                state["attention_mask"] = attention_mask
            if attention_bias is not None:
                state["attention_bias"] = attention_bias

            return log_probs, state

        initial_preds = input_ids.new_zeros((batch_size,))  # This is arbitrary, we won't use this.
        state: dict[str, torch.Tensor] = {"input_ids": input_ids}
        if attention_mask is not None:
            state["attention_mask"] = attention_mask
        if attention_bias is not None:
            state["attention_bias"] = attention_bias
        with torch.no_grad():
            token_ids, scores = beam_search.search(initial_preds, state, step)

        return OLMoGenerateOutput(
            token_ids=token_ids,  # type: ignore[arg-type]
            scores=scores,  # type: ignore[arg-type]
        )

    @classmethod
    def from_checkpoint(
        cls, checkpoint_dir: PathOrStr, device: str = "cpu", checkpoint_type: Optional[CheckpointType] = None
    ) -> OLMo:
        """
        Load an OLMo model from a checkpoint.
        """
        from .util import resource_path

        # Guess checkpoint type.
        if checkpoint_type is None:
            try:
                if resource_path(checkpoint_dir, "model.pt").is_file():
                    checkpoint_type = CheckpointType.unsharded
                else:
                    checkpoint_type = CheckpointType.sharded
            except FileNotFoundError:
                checkpoint_type = CheckpointType.sharded

        # Load config.
        config_path = resource_path(checkpoint_dir, "config.yaml")
        model_config = ModelConfig.load(config_path, key="model", validate_paths=False)

        if checkpoint_type == CheckpointType.unsharded:
            # Initialize model (always on CPU to start with so we don't run out of GPU memory).
            model_config.init_device = "cpu"
            model = OLMo(model_config)

            # Load state dict directly to target device.
            state_dict_path = resource_path(checkpoint_dir, "model.pt")
            state_dict = torch.load(state_dict_path, map_location="cpu")
            model.load_state_dict(model._make_state_dict_compatible(state_dict)[0])
            model = model.to(torch.device(device))
        else:
            train_config = TrainConfig.load(config_path)
            if train_config.sharded_checkpointer == ShardedCheckpointerType.olmo_core:
                from olmo_core.distributed.checkpoint import (  # type: ignore
                    load_model_and_optim_state,
                )

                model_config.init_device = device
                model = OLMo(model_config)
                load_model_and_optim_state(checkpoint_dir, model)
            else:
                # train_config.sharded_checkpointer == ShardedCheckpointerType.torch_new
                from .checkpoint import load_model_state

                # Initialize model on target device. In this case the state dict is loaded in-place
                # so it's not necessary to start on CPU if the target device is a GPU.
                model_config.init_device = device
                model = OLMo(model_config)

                # Load state dict in place.
                load_model_state(checkpoint_dir, model)

        return model.eval()

    def _make_state_dict_compatible(
        self, state_dict: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Set[str]]]:
        """
        Handles some cases where the state dict is valid yet may need to be transformed in order to
        be loaded.

        This modifies the state dict in-place and also returns it, along with a mapping of original key
        names to new key names in cases where the keys were simply renamed. That mapping can be used
        to make a corresponding optimizer state dict compatible as well.
        """
        import re
        from fnmatch import fnmatch

        new_keys_to_og_keys: Dict[str, str] = {}

        # Remove "_fsdp_wrapped_module." prefix from all keys. We don't want this prefix when the model is
        # not wrapped in FSDP. And when the model is wrapped in FSDP, loading this state dict will still work
        # fine without the prefixes. This also simplifies the other steps below.
        for key in list(state_dict.keys()):
            state_dict[(new_key := key.replace("_fsdp_wrapped_module.", ""))] = state_dict.pop(key)
            new_keys_to_og_keys[new_key] = key

        # For backwards compatibility prior to fixing https://github.com/allenai/LLM/issues/222
        if self.config.block_type == BlockType.sequential:
            for key in list(state_dict.keys()):
                if fnmatch(key, "transformer.*.norm.weight"):
                    tensor = state_dict.pop(key)
                    state_dict[(new_key := key.replace("norm.weight", "attn_norm.weight"))] = tensor
                    new_keys_to_og_keys[new_key] = new_keys_to_og_keys[key]
                    state_dict[(new_key := key.replace("norm.weight", "ff_norm.weight"))] = tensor.clone()
                    new_keys_to_og_keys[new_key] = new_keys_to_og_keys[key]
                    del new_keys_to_og_keys[key]
                elif fnmatch(key, "transformer.*.norm.bias"):
                    tensor = state_dict.pop(key)
                    state_dict[(new_key := key.replace("norm.bias", "attn_norm.bias"))] = tensor
                    new_keys_to_og_keys[new_key] = new_keys_to_og_keys[key]
                    state_dict[(new_key := key.replace("norm.bias", "ff_norm.bias"))] = tensor.clone()
                    new_keys_to_og_keys[new_key] = new_keys_to_og_keys[key]
                    del new_keys_to_og_keys[key]

        # For loading a state dict that was saved with a different `block_group_size`.
        if "transformer.block_groups.0.0.attn_out.weight" in state_dict.keys():
            state_dict_block_group_size = len(
                [k for k in state_dict.keys() if fnmatch(k, "transformer.block_groups.0.*.attn_out.weight")]
            )
        else:
            state_dict_block_group_size = 1
        if self.config.block_group_size != state_dict_block_group_size:
            log.info(
                f"Regrouping state dict blocks from group size {state_dict_block_group_size} to "
                f"group size {self.config.block_group_size}"
            )
            # For simplicity we're first going to flatten out the block groups in the state dict (if necessary)
            # and then (re-)group them into the right block sizes.
            if state_dict_block_group_size > 1:
                for key in list(state_dict.keys()):
                    if (m := re.match(r"transformer.block_groups\.(\d+)\.(\d+)\..*", key)) is not None:
                        group_idx, group_block_idx = int(m.group(1)), int(m.group(2))
                        block_idx = (group_idx * state_dict_block_group_size) + group_block_idx
                        state_dict[
                            (
                                new_key := key.replace(
                                    f"block_groups.{group_idx}.{group_block_idx}.", f"blocks.{block_idx}."
                                )
                            )
                        ] = state_dict.pop(key)
                        new_keys_to_og_keys[new_key] = new_keys_to_og_keys.pop(key)

            if self.config.block_group_size > 1:
                # Group the state dict blocks into the right block size.
                for key in list(state_dict.keys()):
                    if (m := re.match(r"transformer.blocks\.(\d+)\..*", key)) is not None:
                        block_idx = int(m.group(1))
                        group_idx, group_block_idx = (
                            block_idx // self.config.block_group_size,
                            block_idx % self.config.block_group_size,
                        )
                        state_dict[
                            (
                                new_key := key.replace(
                                    f"blocks.{block_idx}.", f"block_groups.{group_idx}.{group_block_idx}."
                                )
                            )
                        ] = state_dict.pop(key)
                        new_keys_to_og_keys[new_key] = new_keys_to_og_keys.pop(key)

        og_keys_to_new: Dict[str, Set[str]] = defaultdict(set)
        for new_key, og_key in new_keys_to_og_keys.items():
            og_keys_to_new[og_key].add(new_key)

        return state_dict, og_keys_to_new
    

class OLMoForCausalLM(OLMo):
    
    @classmethod
    def from_checkpoint(
        cls, 
        checkpoint_dir: PathOrStr, 
        device: str = "cpu", 
        checkpoint_type: Optional[CheckpointType] = None
    ) -> OLMoForCausalLM:
        """
        Load an OLMo model from a checkpoint.
        """
        from .util import resource_path

        # Guess checkpoint type.
        if checkpoint_type is None:
            try:
                if resource_path(checkpoint_dir, "model.pt").is_file():
                    checkpoint_type = CheckpointType.unsharded
                else:
                    checkpoint_type = CheckpointType.sharded
            except FileNotFoundError:
                checkpoint_type = CheckpointType.sharded

        # Load config.
        config_path = resource_path(checkpoint_dir, "config.yaml")
        model_config = ModelConfig.load(config_path, key="model", validate_paths=False)
        model_config.flash_attention = False # only for OLMoForCausalLM

        if checkpoint_type == CheckpointType.unsharded:
            # Initialize model (always on CPU to start with so we don't run out of GPU memory).
            model_config.init_device = device
            model = OLMoForCausalLM(model_config)

            # Load state dict directly to target device.
            state_dict_path = resource_path(checkpoint_dir, "model.pt")
            state_dict = torch.load(state_dict_path, map_location="cpu")
            model.load_state_dict(model._make_state_dict_compatible(state_dict)[0])
            model = model.to(torch.device(device))
        else:
            train_config = TrainConfig.load(config_path)
            if train_config.sharded_checkpointer == ShardedCheckpointerType.olmo_core:
                from olmo_core.distributed.checkpoint import (  # type: ignore
                    load_model_and_optim_state,
                )

                model_config.init_device = device
                model = OLMoForCausalLM(model_config)
                load_model_and_optim_state(checkpoint_dir, model)
            else:
                # train_config.sharded_checkpointer == ShardedCheckpointerType.torch_new
                from .checkpoint import load_model_state

                # Initialize model on target device. In this case the state dict is loaded in-place
                # so it's not necessary to start on CPU if the target device is a GPU.
                model_config.init_device = device
                model = OLMoForCausalLM(model_config)

                # Load state dict in place.
                load_model_state(checkpoint_dir, model)

        return model.eval()
        
    
    def generate(
        self, 
        input_ids,
        attention_mask = None,
        max_new_tokens = 10,
        num_beams = 1, 
        min_length = None,
        return_input_ids = True,
        eliminate_token_ids = [186, 187, 8004, 8044], # 186: \t, 187: \n
        eliminate_tolerance = 1,
        **kwargs
        # do_sample = False,
        # temperature = 1.0,
        # eos_token_id = None,
        
    ):
        outputs = super().generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_steps=max_new_tokens,
            beam_size=num_beams,
            per_node_beam_size=None,
            sampler=None,
            min_steps=min_length,
            final_sequence_scorer=None,
            constraints=None,
        ).token_ids.squeeze(-2)
        
        if eliminate_token_ids:
            output_ids = []
            for token_id in eliminate_token_ids:
                positions = torch.where(outputs == token_id)
                
                if len(positions[-1]) > 0:
                    
                    for i in range(len(outputs)):
                        position = positions[-1][positions[0]==i]
                        
                        if position[0] > eliminate_tolerance: # 避免一开始就截断
                            output_ids.append(outputs[i, :position[0]])
                        elif len(position) > 1:
                            output_ids.append(outputs[i, :position[1]])
            
            if len(output_ids) == 0:
                    output_ids = outputs
        else:
            output_ids = outputs
        
        if return_input_ids:
            return [torch.cat([input_id, output_id], dim=-1) for input_id, output_id in zip(input_ids, output_ids)]
        else:
            return output_ids