import math
from typing import Sequence

from neuromta.common.core import *

from neuromta.ip.hardware.common.custom_types import MemoryType
from neuromta.ip.hardware.common.buffer_handle import CircularBufferHandle, BufferHandle

from neuromta.ip.hardware.core.memory import MemoryContext, MemoryOwnerCore
from neuromta.ip.hardware.core.interconnect import IcntNetworkContext, RouterCore


__all__ = ['DMACore']


class DMACore(MemoryOwnerCore, RouterCore):
    def __init__(
        self,
        coord: tuple[int, int],
        mem_context: MemoryContext,
        icnt_context: IcntNetworkContext,
    ):
        super().__init__(
            core_id=DEFAULT_CORE_ID, 
            cycle_model=DMACoreCycleModel(self),
            functional_model=DMACoreFunctionalModel(self),
            coord=coord,
            mem_context=mem_context,
            mem_seg_type=MemoryType.MAIN,
            icnt_context=icnt_context
        )
        
    #######################################
    # Memory Lock Commands
    #######################################
    
    @core_command_method
    def acquire_mem_lock(self, handle: BufferHandle | CircularBufferHandle | str):
        pid = get_global_context()
        if not handle.lock.is_acquired_with(key=pid):
            handle.lock.acquire(key=pid)
        
    @core_command_method
    def release_mem_lock(self, handle: BufferHandle | CircularBufferHandle | str):
        pid = get_global_context()
        if handle.lock.is_locked_with(key=pid):
            handle.lock.release(key=pid)

    #######################################
    # Memory Access Commands
    #######################################
        
    @core_command_method
    def _atom_dma_read(self, src_handle: BufferHandle, src_page_idx: int, tmp_dst_handle: BufferHandle, tmp_dst_page_idx: int, n_pages: int):
        pid = get_global_context()
        if not src_handle.lock.is_locked_with(key=pid):
            return False
    
    @core_command_method
    def _atom_dma_write(self, dst_handle: BufferHandle, dst_page_idx: int, tmp_src_handle: BufferHandle, tmp_src_page_idx: int, n_pages: int):
        pid = get_global_context()
        if not dst_handle.lock.is_locked_with(key=pid):
            return False
        
    @core_kernel_method
    def dma_read(self, src_handle: BufferHandle, src_page_idx: int, tmp_dst_handle: BufferHandle, tmp_dst_page_idx: int, n_pages: int):
        self.acquire_mem_lock(src_handle)
        self._atom_dma_read(src_handle, src_page_idx, tmp_dst_handle, tmp_dst_page_idx, n_pages)
        self.release_mem_lock(src_handle)
        
    @core_kernel_method
    def dma_write(self, dst_handle: BufferHandle, dst_page_idx: int, tmp_src_handle: BufferHandle, tmp_src_page_idx: int, n_pages: int):
        self.acquire_mem_lock(dst_handle)
        self._atom_dma_write(dst_handle, dst_page_idx, tmp_src_handle, tmp_src_page_idx, n_pages)
        self.release_mem_lock(dst_handle)

class DMACoreCycleModel(CoreCycleModel):
    def __init__(self, core: 'DMACore'):
        super().__init__()

        self.core = core
        
    def _atom_dma_read(self, src_handle: BufferHandle, src_page_idx: int, tmp_dst_handle: BufferHandle, tmp_dst_page_idx: int, n_pages: int):
        return self.core.mem_context.get_main_mem_rd_latency(n_pages * src_handle.page_size)
        
    def _atom_dma_write(self, dst_handle: BufferHandle, dst_page_idx: int, tmp_src_handle: BufferHandle, tmp_src_page_idx: int, n_pages: int):
        return self.core.mem_context.get_main_mem_wr_latency(n_pages * dst_handle.page_size)
        
class DMACoreFunctionalModel(CoreFunctionalModel):
    def __init__(self, core: 'DMACore'):
        super().__init__()

        self.core = core
        
    def _atom_dma_read(self, src_handle: BufferHandle, src_page_idx: int, tmp_dst_handle: BufferHandle, tmp_dst_page_idx: int, n_pages: int):
        pages = src_handle.data_get_page_burst(src_page_idx, n_pages)
        tmp_dst_handle.data_set_page_burst(tmp_dst_page_idx, pages)
        
    def _atom_dma_write(self, dst_handle: BufferHandle, dst_page_idx: int, tmp_src_handle: BufferHandle, tmp_src_page_idx: int, n_pages: int):
        pages = tmp_src_handle.data_get_page_burst(tmp_src_page_idx, n_pages)
        dst_handle.data_set_page_burst(dst_page_idx, pages)