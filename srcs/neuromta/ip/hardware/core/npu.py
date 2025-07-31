from neuromta.common.core import *

from neuromta.ip.hardware.common.custom_types import MemoryType
from neuromta.ip.hardware.common.buffer_handle import CircularBufferHandle, BufferHandle, TemporaryBufferHandle
from neuromta.ip.hardware.common.tensor_processor_config import TensorProcessorConfig

from neuromta.ip.hardware.core.memory import MemoryContext, MemoryOwnerCore
from neuromta.ip.hardware.core.interconnect import IcntNetworkContext, IcntNetworkCore, RouterCore
from neuromta.ip.hardware.core.dma import DMACore


__all__ = [
    "TensorProcessorContext",
    "NPUCore",
] 


class TensorProcessorContext:
    def __init__(
        self,
        config: TensorProcessorConfig = TensorProcessorConfig(),
    ):
        self.config = config
        
    
class NPUCore(MemoryOwnerCore, RouterCore):
    def __init__(
        self,
        coord: tuple[int, int],
        mem_context: MemoryContext,
        icnt_context: IcntNetworkContext,
    ):
        super().__init__(
            core_id=DEFAULT_CORE_ID, 
            cycle_model=NPUCoreCycleModel(core=self),
            functional_model=NPUCoreFunctionalModel(core=self),
            coord=coord,
            mem_context=mem_context,
            mem_seg_type=MemoryType.L1,
            icnt_context=icnt_context
        )
        
    #######################################
    # Memory Lock Commands
    #######################################
    
    @core_command_method
    def _atom_acquire_mem_lock(self, handle: BufferHandle | CircularBufferHandle | str):
        pid = get_global_context()
        if not handle.lock.is_acquired_with(key=pid):
            handle.lock.acquire(key=pid)

    @core_command_method
    def _atom_release_mem_lock(self, handle: BufferHandle | CircularBufferHandle | str):
        pid = get_global_context()
        if handle.lock.is_locked_with(key=pid):
            handle.lock.release(key=pid)
        
    #######################################
    # Memory Access Commands
    #######################################
    
    @core_command_method
    def _atom_read_l1_mem(self, src_handle: BufferHandle, src_page_idx: int, tmp_dst_handle: BufferHandle, tmp_dst_page_idx: int, n_pages: int):
        pid = get_global_context()
        if not src_handle.lock.is_locked_with(key=pid): 
            return False
    
    @core_command_method
    def _atom_write_l1_mem(self, dst_handle: BufferHandle, dst_page_idx: int, tmp_src_handle: BufferHandle, tmp_src_page_idx: int, n_pages: int):
        pid = get_global_context()
        if not dst_handle.lock.is_locked_with(key=pid): 
            return False
        
    @core_kernel_method
    def read_l1_mem(self, src_handle: BufferHandle, src_page_idx: int, tmp_dst_handle: BufferHandle, tmp_dst_page_idx: int, n_pages: int):
        self._atom_acquire_mem_lock(src_handle)
        self._atom_read_l1_mem(src_handle, src_page_idx, tmp_dst_handle, tmp_dst_page_idx, n_pages)
        self._atom_release_mem_lock(src_handle)
        
    @core_kernel_method
    def write_l1_mem(self, dst_handle: BufferHandle, dst_page_idx: int, tmp_src_handle: BufferHandle, tmp_src_page_idx: int, n_pages: int):
        self._atom_acquire_mem_lock(dst_handle)
        self._atom_write_l1_mem(dst_handle, dst_page_idx, tmp_src_handle, tmp_src_page_idx, n_pages)
        self._atom_release_mem_lock(dst_handle)
            
    @core_kernel_method
    def copy_page(self, src_handle: BufferHandle, src_page_idx: int, dst_handle: BufferHandle, dst_page_idx: int, n_pages: int):
        if self.mem_context.check_l1_mem_addr(src_handle.addr):
            src_core_id, _, _ = self.mem_context.parse_l1_mem_addr(src_handle.addr)
            src_core: NPUCore = self.mem_context.get_owner(MemoryType.L1, src_core_id)
            src_coord = src_core.coord
        elif self.mem_context.check_main_mem_addr(src_handle.addr):
            src_core_id, _, _ = self.mem_context.parse_main_mem_addr(src_handle.addr)
            src_core: DMACore = self.mem_context.get_owner(MemoryType.MAIN, src_core_id)
            src_coord = src_core.coord
            
        if self.mem_context.check_l1_mem_addr(dst_handle.addr):
            dst_core_id, _, _ = self.mem_context.parse_l1_mem_addr(dst_handle.addr)
            dst_core: NPUCore = self.mem_context.get_owner(MemoryType.L1, dst_core_id)
            dst_coord = dst_core.coord
        elif self.mem_context.check_main_mem_addr(dst_handle.addr):
            dst_core_id, _, _ = self.mem_context.parse_main_mem_addr(dst_handle.addr)
            dst_core: DMACore = self.mem_context.get_owner(MemoryType.MAIN, dst_core_id)
            dst_coord = dst_core.coord

        tmp_handle = TemporaryBufferHandle(page_size=src_handle.page_size, n_pages=n_pages)
        
        # start parallel
        for i in range(n_pages):
            self.create_new_parallel_kernel()   
            
            if isinstance(src_core, NPUCore):
                src_core.read_l1_mem(src_handle, src_page_idx + i, tmp_handle, i, 1)
            elif isinstance(src_core, DMACore):
                src_core.dma_read(src_handle, src_page_idx + i, tmp_handle, i, 1)
            
            if src_coord != dst_coord:
                self.icnt_context.icnt_core.request_data_transfer(consumer_coord=dst_coord, producer_coord=src_coord, data_size=tmp_handle.page_size)
                
            if isinstance(dst_core, NPUCore):
                dst_core.write_l1_mem(dst_handle, dst_page_idx + i, tmp_handle, i, 1)
            elif isinstance(dst_core, DMACore):
                dst_core.dma_write(dst_handle, dst_page_idx + i, tmp_handle, i, 1)
                
        self.merge_parallel_kernels()
        # end parallel

    ########################################
    # Circular Buffer Management Commands
    ########################################
    
    @core_command_method
    def cb_reserve_back(self, handle: CircularBufferHandle, n_pages: int):
        if not handle.check_vacancy(n_pages): return False
        handle.allocate_cb_space(n_pages=n_pages)
    
    @core_command_method
    def cb_push_back(self, handle: CircularBufferHandle, n_pages: int):
        handle.occupy_cb_space(n_pages=n_pages)
        
    @core_command_method
    def cb_wait_front(self, handle: CircularBufferHandle, n_pages: int):
        if not handle.check_occupancy(n_pages): return False
        handle.evacuate_cb_space(n_pages=n_pages)
        
    @core_command_method
    def cb_pop_front(self, handle: CircularBufferHandle, n_pages: int):
        handle.deallocate_cb_space(n_pages=n_pages)
        
class NPUCoreCycleModel(CoreCycleModel):
    def __init__(self, core: 'NPUCore'):
        super().__init__()
        
        self.core = core
    
    def _atom_read_l1_mem(self, src_handle: BufferHandle, src_page_idx: int, tmp_dst_handle: BufferHandle, tmp_dst_page_idx: int, n_pages: int):
        return self.core.mem_context.get_l1_mem_rd_latency(n_pages * src_handle.page_size)
    
    def _atom_write_l1_mem(self, dst_handle: BufferHandle, dst_page_idx: int, tmp_src_handle: BufferHandle, tmp_src_page_idx: int, n_pages: int):
        return self.core.mem_context.get_l1_mem_wr_latency(n_pages * dst_handle.page_size)

class NPUCoreFunctionalModel(CoreFunctionalModel):
    def __init__(self, core: 'NPUCore'):
        super().__init__()
        
        self.core = core
        
    def _atom_read_l1_mem(self, src_handle: BufferHandle, src_page_idx: int, tmp_dst_handle: BufferHandle, tmp_dst_page_idx: int, n_pages: int):
        pages = src_handle.data_get_page_burst(src_page_idx, n_pages)
        tmp_dst_handle.data_set_page_burst(tmp_dst_page_idx, pages)
        
    def _atom_write_l1_mem(self, dst_handle: BufferHandle, dst_page_idx: int, tmp_src_handle: BufferHandle, tmp_src_page_idx: int, n_pages: int):
        pages = tmp_src_handle.data_get_page_burst(tmp_src_page_idx, n_pages)
        dst_handle.data_set_page_burst(dst_page_idx, pages)
        
        
if __name__ == "__main__":
    from neuromta.common.device import Device
    
    class MyDevice(Device):
        def __init__(self):
            super().__init__()
            
            self.mem_context = MemoryContext()
            self.icnt_context = IcntNetworkContext(grid_shape=(4, 4))
            
            self.npu_core = NPUCore(coord=(0, 0), mem_context=self.mem_context, icnt_context=self.icnt_context)
            self.dma_core = DMACore(coord=(0, 1), mem_context=self.mem_context, icnt_context=self.icnt_context)
            self.icnt_core = IcntNetworkCore(icnt_context=self.icnt_context)
            
    device = MyDevice()
    device.initialize(create_trace=False)
    
    bf_handle = device.mem_context.create_buffer_handle("buffer1", addr=device.mem_context.get_main_mem_addr(0, 0, 0), page_size=32*32*4, n_pages=8)
    cb_handle = device.mem_context.create_circular_buffer_handle("circular_buffer1", addr=bf_handle.addr + bf_handle.size, page_size=32*32*4, n_pages=8)
    
    bf_handle.data_set_page(0, "DATA 1")
    bf_handle.data_set_page(1, "DATA 2")
    bf_handle.data_set_page(2, "DATA 3")
    bf_handle.data_set_page(3, "DATA 4")
    
    @core_kernel_method
    def reader_kernel(core: NPUCore, bf_handle: BufferHandle, cb_handle: CircularBufferHandle) -> int:
        core.cb_reserve_back(cb_handle, 2)
        core.copy_page(src_handle=bf_handle, src_page_idx=0, dst_handle=cb_handle, dst_page_idx=0, n_pages=2)
        core.cb_push_back(cb_handle, 2)
    
    @core_kernel_method
    def writer_kernel(core: NPUCore, bf_handle: BufferHandle, cb_handle: CircularBufferHandle) -> int:
        core.cb_wait_front(cb_handle, 2)
        core.copy_page(src_handle=cb_handle, src_page_idx=0, dst_handle=bf_handle, dst_page_idx=4, n_pages=2)
        core.cb_pop_front(cb_handle, 2)
    
    reader_kernel(device.npu_core, bf_handle, cb_handle)
    writer_kernel(device.npu_core, bf_handle, cb_handle)
    
    device.verbose = True   # print debug messages
    device.run_kernels()
    
    for i in range(bf_handle.n_pages):
        print(f"Buffer Page {i}: {bf_handle.data_get_page(i)}")