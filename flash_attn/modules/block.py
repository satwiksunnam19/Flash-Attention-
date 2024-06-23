# copyright 2024, Tri Dao.

from functools import partial 
from typing import Optional

import torch 
import torch.nn as nn
import torch.nn.functional as F 
from torch import Tensor 

from tochvision.ops import StochasticDepth

from flash_attn.modules.mha import MHA 
from flash_attn.modules.mlp import Mlp