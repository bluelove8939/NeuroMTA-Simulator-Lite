import math
import functools
from typing import Sequence

from neuromta.common import *


__all__ = [
    "MemoryDescriptor",
    "VariableDescriptor",
    "BufferDescriptor",
]


class MemoryDescriptor:
    def __init__(self, mem_type: MemoryType, mem_bank_id: int):
        self.mem_type       = mem_type
        self.mem_bank_id    = mem_bank_id
        
        if not isinstance(self.mem_bank_id, int):
            raise Exception(f"[ERROR] The memory bank ID should be an integer, not '{type(self.mem_bank_id).__name__}'")
        
    @classmethod
    def l1(cls, core_id: int):
        return cls(MemoryType.L1, mem_bank_id=core_id)
    
    @classmethod
    def l2(cls):
        return cls(MemoryType.L2, 0)
        
    def __eq__(self, value):
        if not isinstance(value, MemoryDescriptor):
            return False
        return self.mem_type == value.mem_type and self.mem_bank_id == value.mem_bank_id

    def __ne__(self, value):
        return not self.__eq__(value)
    
    def __hash__(self):
        return hash((self.mem_type, self.mem_bank_id))\
            
    def __str__(self):
        return f"Memory(mem_type={self.mem_type.name:2s}, bank_id={self.mem_bank_id})"
        

class VariableDescriptor:
    def __init__(self, var_id: str, shape: Sequence[int], tile_shape: Sequence[int], word_size: int):
        self.var_id     = var_id
        self.shape      = shape
        self.tile_shape = tile_shape
        self.word_size  = word_size
        
        if len(shape) != 2:
            raise Exception(f"[ERROR] The shape of the variable should be 2D, not {len(shape)}D")
        if len(tile_shape) != 2:
            raise Exception(f"[ERROR] The shape of the tile should be 2D, not {len(tile_shape)}D")
        
    def create_buffer(self, mem_desc: MemoryDescriptor, tile_idx: int):
        if not isinstance(tile_idx, int):
            raise Exception(f"[ERROR] The tile index should be an integer, not '{type(tile_idx).__name__}'")
        
        return BufferDescriptor(var_desc=self, mem_desc=mem_desc, tile_idx=tile_idx,)
        
    def tile_id(self, r, c):
        rn = math.ceil(self.shape[0] / self.tile_shape[0])
        cn = math.ceil(self.shape[1] / self.tile_shape[1])
        
        if r >= rn:
            raise Exception(f"[ERROR] Invalid row index '{r}' since it exceeds its limit '{rn}'")
        if c >= cn:
            raise Exception(f"[ERROR] Invalid column index '{c}' since it exceeds its limit '{cn}'")
        
        return r * cn + c
        
    @property
    def size(self) -> int:
        return functools.reduce(lambda a, b: a * b, self.shape, 1) * self.word_size

    @property
    def tile_size(self) -> int:
        return functools.reduce(lambda a, b: a * b, self.tile_shape, 1) * self.word_size
        
    def __eq__(self, value):
        if not isinstance(value, VariableDescriptor):
            raise Exception(f"[ERROR] Cannot compare variable with '{type(value).__name__}'")
        
        return self.var_id == value.var_id

    def __ne__(self, value):
        return not self.__eq__(value)
    
    def __hash__(self):
        return hash((self.var_id, self.shape, self.tile_shape, self.word_size))
    
    def __str__(self):
        return f"Variable(id={self.var_id:6s}, shape={str(self.shape):16s}, tile_shape{str(self.tile_shape):16s})"

    
class BufferDescriptor:
    def __init__(
        self,
        var_desc: VariableDescriptor,
        mem_desc: MemoryDescriptor,
        tile_idx: int,
    ):  
        self.var_desc   = var_desc
        self.mem_desc   = mem_desc
        self.tile_idx   = tile_idx
    
    @property
    def size(self) -> int:
        return self.var_desc.tile_size
        
    def __eq__(self, value):
        if not isinstance(value, BufferDescriptor):
            return False
        return self.var_desc == value.var_desc and self.tile_idx == value.tile_idx and self.mem_desc == value.mem_desc
    
    def __ne__(self, value):
        return not self.__eq__(value)
        
    def __hash__(self):
        return hash((hash(self.var_desc), hash(self.mem_desc), self.tile_idx))
        
    def __str__(self):
        return f"Buffer(var={self.var_desc.var_id:6s}, mem_desc={self.mem_desc}, tile_idx={self.tile_idx})"