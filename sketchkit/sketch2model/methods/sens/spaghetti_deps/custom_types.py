# import open3d
import enum
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as nnf
from constants import DEBUG
from typing import Tuple, List, Union, Callable, Type, Iterator, Dict, Set, Optional, Any, Sized, Iterable
from types import DynamicClassAttribute
from enum import Enum, unique
import torch.optim.optimizer
import torch.utils.data

if DEBUG or True:
    seed = 99
    torch.manual_seed(seed)
    np.random.seed(seed)

N = type(None)
V = np.array
ARRAY = np.ndarray
ARRAYS = Union[Tuple[ARRAY, ...], List[ARRAY]]
VN = Optional[ARRAY]
VNS = Optional[ARRAYS]
T = torch.Tensor
TS = Union[Tuple[T, ...], List[T]]
TN = Optional[T]
TNS = Union[Tuple[TN, ...], List[TN]]
TSN = Optional[TS]
TA = Union[T, ARRAY]

V_Mesh = Tuple[ARRAY, ARRAY]
T_Mesh = Tuple[T, Optional[T]]
T_Mesh_T = Union[T_Mesh, T]
COLORS = Union[T, ARRAY, Tuple[int, int, int]]

D = torch.device
CPU = torch.device('cpu')


def get_device(device_id: int) -> D:
    if not torch.cuda.is_available():
        if device_id >= 0:
            print("warning GPU is not available")
        return CPU
    device_id = min(torch.cuda.device_count() - 1, device_id)
    return torch.device(f'cuda:{device_id}')


CUDA = get_device
Optimizer = torch.optim.Adam
Dataset = torch.utils.data.Dataset
DataLoader = torch.utils.data.DataLoader
Subset = torch.utils.data.Subset


@unique
class Padding(Enum):
    ZERO = 0
    REFLECTIVE = 1


class ModelType(enum.Enum):

    @DynamicClassAttribute
    def value(self) -> str:
        return super(ModelType, self).value

    PPE3 = 'ppe3'
    PPE2 = 'ppe2'
    EXPLICIT = 'exp'
    PE = 'pe'
    ReLU = 'relu'
    PPE = 'ppe'
    SIREN = 'siren'
    HYBRID = 'hybrid'


class LossType(enum.Enum):

    @DynamicClassAttribute
    def value(self) -> str:
        return super(LossType, self).value

    CROSS = 'cross'
    HINGE = 'hinge'
    IN_OUT = 'in_out'