import torch
from typing import Any, Sequence

from neuromta.framework import *

from neuromta.hardware.multi_tile.context.mem_context import *
from neuromta.hardware.multi_tile.context.icnt_context import *
from neuromta.hardware.multi_tile.context.cmap_context import *
from neuromta.hardware.multi_tile.context.mxu_context import *
from neuromta.hardware.multi_tile.context.vpu_context import *

from neuromta.hardware.multi_tile.core.npu_core import *
from neuromta.hardware.multi_tile.core.dma_core import *
from neuromta.hardware.multi_tile.core.icnt_core import *
from neuromta.hardware.multi_tile.core.main_mem_core import *


__all__ = [
    "MTAccelerator"
]


class MTAccelerator(Device):
    def __init__(
        self, 
        
        cmap_config: CmapConfig, 
        icnt_config: IcntConfig,
        mem_config: MemConfig,
        mxu_config: MXUConfig,
        vpu_config: VPUConfig,
    ):
        super().__init__()
        
        self.cmap_context = CmapContext(config=cmap_config)
        self.icnt_context = IcntContext(**icnt_config)
        self.mem_context  = MemContext(**mem_config)
        
        self.mxu_config = mxu_config
        self.vpu_config = vpu_config
        
        self.npu_core_coords = self.cmap_context.npu_core_coords
        self.dma_core_coords = self.cmap_context.dma_core_coords

        self.npu_coord_to_core_idx_mappings = {coord: idx for idx, coord in enumerate(self.npu_core_coords)}
        self.dma_coord_to_core_idx_mappings = {coord: idx for idx, coord in enumerate(self.dma_core_coords)}

        self.npu_cores: list[NPUCore] = [
            NPUCore(coord=coord, mem_context=self.mem_context, cmap_context=self.cmap_context, mxu_config=self.mxu_config, vpu_config=self.vpu_config)
            for coord in self.npu_core_coords
        ]
        
        self.dma_cores: list[DMACore] = [
            DMACore(coord=coord, mem_context=self.mem_context, cmap_context=self.cmap_context)
            for coord in self.dma_core_coords
        ]
        
        self.icnt_core = IcntCore(cmap_context=self.cmap_context, icnt_context=self.icnt_context)
        self.main_mem_core = MainMemoryCore(mem_context=self.mem_context, cmap_context=self.cmap_context)
        
    def get_core_from_coord(self, coord: tuple[int, int]) -> NPUCore | DMACore:
        if coord in self.npu_coord_to_core_idx_mappings:
            return self.npu_cores[self.npu_coord_to_core_idx_mappings[coord]]
        elif coord in self.dma_coord_to_core_idx_mappings:
            return self.dma_cores[self.dma_coord_to_core_idx_mappings[coord]]
        else:
            raise ValueError(f"[ERROR] No core found for coordinate {coord}.")
    
    def get_npu_core(self, key: tuple[int, int] | int) -> NPUCore:
        if isinstance(key, int):
            coord = self.cmap_context.get_coord_from_address(addr=key)
        elif isinstance(key, tuple):
            coord = key
        
        core_idx = self.npu_coord_to_core_idx_mappings.get(coord, None)
        
        if core_idx is None:
            raise Exception(f"[ERROR] No NPU core found for coordinate {coord}.")

        return self.npu_cores[core_idx]
    
    def get_l1_mem_handle(self, key: tuple[int, int] | int) -> MemoryHandle:
        core = self.get_npu_core(key)
        return core.mem_handle
    
    def get_main_mem_handle(self) -> MemoryHandle:
        return self.main_mem_core.mem_handle

    def create_local_l1_circular_buffer(self, page_size: int, n_pages: int, coords: list[tuple[int, int]]=None) -> Reference | list[Reference]:
        if coords is None:
            coords = self.npu_core_coords
        if len(coords) == 2 and isinstance(coords[0], int) and isinstance(coords[1], int):
            coords = [coords]
            
        ptrs: list[BufferHandle] = []

        for coord in coords:
            mem_handle = self.get_l1_mem_handle(coord)
            ptr = create_uniform_buffer(mem_handle=mem_handle, page_size=page_size, n_pages=n_pages, is_circular=True)
            ptrs.append(ptr)

        if len(coords) == 1:
            return ptrs[0]
        return ptrs
    
    def create_local_l1_buffer(self, page_size: int, n_pages: int, coords: list[tuple[int, int]]=None) -> Reference | list[Reference]:
        if coords is None:
            coords = self.npu_core_coords
        if len(coords) == 2 and isinstance(coords[0], int) and isinstance(coords[1], int):
            coords = [coords]
            
        ptrs: list[BufferHandle] = []

        for coord in coords:
            mem_handle = self.get_l1_mem_handle(coord)
            ptr = create_uniform_buffer(mem_handle=mem_handle, page_size=page_size, n_pages=n_pages, is_circular=False)
            ptrs.append(ptr)
        
        if len(coords) == 1:
            return ptrs[0]
        return ptrs

    def create_sharded_l1_buffer(self, page_size: int, n_pages: int, coords: list[tuple[int, int]]=None) -> Reference:
        if coords is None:
            coords = self.cmap_context.config.core_coord(CmapCoreType.NPU)
        if len(coords) == 2 and isinstance(coords[0], int) and isinstance(coords[1], int):
            coords = [coords]

        mem_handles = [self.get_l1_mem_handle(coord) for coord in coords]
        ptr = create_distributed_buffer(mem_handles=mem_handles, page_size=page_size, n_pages=n_pages)

        return ptr

    def create_sharded_main_buffer(self, page_size: int, n_pages: int, channel_id: int | Sequence[int]=None) -> Reference:
        # if coords is None:
        #     coords = self.cmap_context.config.core_coord(CmapCoreType.DMA)
        # if len(coords) == 2 and isinstance(coords[0], int) and isinstance(coords[1], int):
        #     coords = [coords]
            
        # channel_id = [self.get_core_from_coord(coord).channel_id for coord in coords]
        
        if channel_id is None:
            channel_id = list(range(self.cmap_context.config.n_main_mem_channels))
        
        mem_handle = self.get_main_mem_handle()
        
        ptr = create_uniform_buffer(mem_handle=mem_handle, page_size=page_size, n_pages=n_pages, is_circular=False, channel_id=channel_id)
        return ptr
    
    def set_ptr_content(self, ptr: Reference | Pointer | BufferHandle, content: torch.Tensor):
        if isinstance(ptr, Reference):
            ptr = ptr.resolve(is_read=False)
        
        if isinstance(ptr, Pointer):
            if ptr.ptr_type == PointerType.PAGE:
                self._set_page_var_ptr_content(ptr, content)
            else:
                self._set_page_var_ptr_content(ptr, content)
        elif isinstance(ptr, BufferHandle):
            page_size = ptr.page_size
            n_pages = ptr.n_pages
            
            if content.numel() * content.element_size() != page_size * n_pages:
                raise ValueError(f"[ERROR] Content size {content.numel() * content.element_size()} does not match buffer size {page_size * n_pages}.")
            
            content = content.view(dtype=torch.uint8).reshape((n_pages, page_size))
            
            for page_ptr, page_content in zip(ptr.page_ptrs, content):
                self._set_page_var_ptr_content(page_ptr, page_content)
        else:
            raise Exception(f"[ERROR] Unsupported pointer type: {type(ptr)}. Expected BufferPointer or Pointer.")

    def _set_page_var_ptr_content(self, ptr: Pointer, content: Any):
        if ptr.ptr_type == PointerType.PAGE:
            if not isinstance(content, torch.Tensor):
                raise ValueError(f"[ERROR] Content must be a torch.Tensor for PAGE pointer, got {type(content)}.")
            
            content = content.view(dtype=torch.uint8)
            
        if self.cmap_context.config.check_main_mem_addr(ptr.addr):
            mem_handle = self.get_main_mem_handle()
        elif self.cmap_context.config.check_l1_mem_addr(ptr.addr):
            mem_handle = self.get_l1_mem_handle(ptr.addr)
        else:
            raise Exception(f"[ERROR] Unsupported address: {ptr.addr}")

        mem_handle.set_content(ptr, content)

    def get_ptr_content(self, ptr: Reference | Pointer | BufferHandle, shape: tuple[int, ...]=None, dtype: torch.dtype=None) -> torch.Tensor:
        if isinstance(ptr, Reference):
            ptr = ptr.resolve(is_read=True)
            
        if isinstance(ptr, Pointer):
            if ptr.ptr_type == PointerType.PAGE:
                return self._get_page_var_ptr_content(ptr, shape, dtype)
            else:
                return self._get_page_var_ptr_content(ptr)
        elif isinstance(ptr, BufferHandle):
            page_size = ptr.page_size
            n_pages = ptr.n_pages
            
            content = torch.empty((n_pages, page_size), dtype=torch.uint8).contiguous()
            
            for i, page_ptr in enumerate(ptr.page_ptrs):
                content[i, :] = self._get_page_var_ptr_content(page_ptr, shape=(-1,), dtype=torch.uint8)
                
            if dtype is not None:
                content = content.view(dtype=dtype)
            if shape is not None:
                content = content.reshape(shape)

            return content
        else:
            raise Exception(f"[ERROR] Unsupported pointer type: {type(ptr)}. Expected BufferPointer or Pointer.")

    def _get_page_var_ptr_content(self, ptr: Pointer, shape: tuple[int, ...]=None, dtype: torch.dtype=None) -> torch.Tensor:
        if self.cmap_context.config.check_main_mem_addr(ptr.addr):
            mem_handle = self.get_main_mem_handle()
        elif self.cmap_context.config.check_l1_mem_addr(ptr.addr):
            mem_handle = self.get_l1_mem_handle(ptr.addr)
        else:
            raise Exception(f"[ERROR] Unsupported address: {ptr.addr}")
        
        content = mem_handle.get_content(ptr, shape=shape, dtype=dtype)

        return content