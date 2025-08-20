import math
import torch

from neuromta.framework import *

from neuromta.hardware.context.mem_context import *
from neuromta.hardware.context.icnt_context import *
from neuromta.hardware.context.mxu_context import *
from neuromta.hardware.context.vpu_context import *

from neuromta.hardware.core.npu_core import *
from neuromta.hardware.core.dma_core import *
from neuromta.hardware.core.icnt_core import *


__all__ = [
    "MTAccelerator"
]


class MTAccelerator(Device):
    def __init__(
        self, 
        
        icnt_config: IcntConfig, 
        mem_config: MemConfig,
        mxu_config: MXUConfig,
        vpu_config: VPUConfig,
    ):
        super().__init__()
        
        self.icnt_context = IcntContext(**icnt_config)
        self.mem_context  = MemContext(**mem_config)
        
        self.mxu_config = mxu_config
        self.vpu_config = vpu_config
        
        self.npu_core_coords = self.icnt_context.npu_core_coords
        self.dma_core_coords = self.icnt_context.dma_core_coords

        self.npu_coord_to_core_idx_mappings = {coord: idx for idx, coord in enumerate(self.npu_core_coords)}
        self.dma_coord_to_core_idx_mappings = {coord: idx for idx, coord in enumerate(self.dma_core_coords)}

        self.npu_cores: list[NPUCore] = [
            NPUCore(coord=coord, mem_context=self.mem_context, icnt_context=self.icnt_context, mxu_config=self.mxu_config, vpu_config=self.vpu_config)
            for coord in self.npu_core_coords
        ]
        
        self.dma_cores: list[DMACore] = [
            DMACore(coord=coord, mem_context=self.mem_context, icnt_context=self.icnt_context)
            for coord in self.dma_core_coords
        ]
        
        self.icnt_core = IcntCore(icnt_context=self.icnt_context)

    def create_circular_buffer_to_cores(self, cb_id: str, page_size: int, n_pages: int, coords: list[tuple[int, int]]=None) -> BufferPointer | list[BufferPointer]:
        if coords is None:
            coords = self.npu_core_coords
        if len(coords) == 2 and isinstance(coords[0], int) and isinstance(coords[1], int):
            coords = [coords]
            
        ptrs: list[Pointer] = []

        for coord in coords:
            core_idx = self.npu_coord_to_core_idx_mappings[coord]
            core = self.npu_cores[core_idx]
            # ptr = Pointer(ptr_id=f"cb_{cb_id}_{core_idx}")
            # core.cb_create(ptr, page_size=page_size, n_pages=n_pages)
            ptr = create_buffer_ptr(mem_handle=core.mem_handle, page_size=page_size, n_pages=n_pages, is_circular=True)
            ptrs.append(ptr)

        if len(coords) == 1:
            return ptrs[0]
        return ptrs
    
    def create_l1_buffer_to_cores(self, bf_id: str, page_size: int, n_pages: int, coords: list[tuple[int, int]]=None) -> BufferPointer | list[BufferPointer]:
        if coords is None:
            coords = self.npu_core_coords
        if len(coords) == 2 and isinstance(coords[0], int) and isinstance(coords[1], int):
            coords = [coords]
            
        ptrs: list[Pointer] = []

        for coord in coords:
            core_idx = self.npu_coord_to_core_idx_mappings[coord]
            core = self.npu_cores[core_idx]
            ptr = create_buffer_ptr(mem_handle=core.mem_handle, page_size=page_size, n_pages=n_pages, is_circular=False)
            ptrs.append(ptr)
        
        if len(coords) == 1:
            return ptrs[0]
        return ptrs
    
    def create_sharded_main_memory_buffer(self, bf_id: str, page_size: int, n_pages: int, coords: list[tuple[int, int]]=None) -> BufferPointer:
        if coords is None:
            coords = self.icnt_context.core_map.core_coord(IcntCoreType.DMA)
        if len(coords) == 2 and isinstance(coords[0], int) and isinstance(coords[1], int):
            coords = [coords]
        
        mem_handles = [self.dma_cores[self.dma_coord_to_core_idx_mappings[coord]].mem_handle for coord in coords]
        ptr = create_sharded_buffer_ptr(mem_handles=mem_handles, page_size=page_size, n_pages=n_pages)

        return ptr
