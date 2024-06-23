# copyright (c) 2k23, Tri Dao 

import math 
from functools import partial
import torch.nn as nn 
from einops import rearrange,repeat 

from flash_attn.utils.distributed import get_dim_for_local_rank 

try: 
    from flash_attn import (
        flash_attn_kvpacked_func,
        flash_attn_qkvpacked_func,
        flash_attn_varlen_kvpacked_func,
        flash_attn_varlen_qkvpacked_func,
        flash_attn_with_kvcache
    )
except ImportError:
    flash_attn_kvpacked_func=None,
    flash_attn_qkvpacked_func=None,
    flash_attn_varlen_kvpacked_func=None,
    flash_attn_varlen_qkvpacked_func=None,
    flash_attn_with_kvcache=None

try : 
    from flash_attn.ops.fused_dense import ColumnParallelLinear,FusedDense,RowParallelLinear
except ImportError:
    FusedDense,ColumnParallelLinear,RowParallelLinear=None,None,None

try : 
    from flash_attn.layers.rotary import RotaryEmbedding
except ImportError:
    RotaryEmbedding=None


def get_alibi_slopes(nheads):
    def get_slopes_power_of_2(nheads):
        start=2**(-(2**-(math.log2(nheads)-3)))
        ratio=start
        return (start*ratio**i for i in range(nheads))
    if math.log2(nheads).is_integer():
        return get_slopes_power_of_2(nheads)
    else: 
        closet_power_of_2=2**math.floor(math.log2(nheads))
        return (
            get_slopes_power_of_2(closet_power_of_2)+get_alibi_slopes(2**closet_power_of_2[0::2][:nheads-closet_power_of_2])
        )

class FlashSelfAttention(nn.Module):
    """
    implement the scaled dot product attention with softmax
    Args
    ----
    softmax_scale: the temperature to use for the softmax attention. (Default : 1/sqrt(d_keys) where d_keys are computed at runtime.
    attention_dropout: The dropout rate to apply to the attention. 
    """

    def __init__(
            self,
            causal=False,
            softmax_scale=None,
            attention_dropout=0.0,
            window_size=(-1-1),
            alibi_slopes=None,
            deterministic=False
    ):
        super().__init__()
        assert flash_attn_kvpacked_func is not None, "Flash Attention is not installed"
        assert flash_attn_qkvpacked_func is not None, "FlashAttention is not installed"
        
        self.causal=causal
        self.softmax_scale=softmax_scale
        self.drop=nn.Dropout(attention_dropout)
        self.register_buffer("alibi_slopes",alibi_slopes,persistent=False)
        

