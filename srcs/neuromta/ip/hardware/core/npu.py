import math
from neuromta.common.core import *

from neuromta.common.custom_types import MemoryType
from neuromta.common.buffer_handle import CircularBufferHandle, BufferHandle, TemporaryBufferHandle, PageHandle
from neuromta.common.synchronizer import TicketLock

from neuromta.ip.hardware.core.memory import MemoryContext, MemoryOwnerCore
from neuromta.ip.hardware.core.interconnect import IcntNetworkContext, IcntNetworkCore, RouterCore
from neuromta.ip.hardware.core.dma import DMACore
from neuromta.ip.hardware.core.vector_unit import VPUContext, VPUConfig, VPUOperator
from neuromta.ip.hardware.core.matrix_unit import MXUContext, MXUConfig, MXUDataflow


__all__ = [
    "NPUCore",
] 
        
    
class NPUCore(MemoryOwnerCore, RouterCore):
    def __init__(
        self,
        coord: tuple[int, int],
        mem_context: MemoryContext,
        icnt_context: IcntNetworkContext,
        vpu_config: VPUConfig = VPUConfig(),
        mxu_config: MXUConfig = MXUConfig(),
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
        
        self.mxu_context = mxu_config.create_context()
        self.vpu_context = vpu_config.create_context()
        
        self.mxu_lock = TicketLock()
        self.vpu_lock = TicketLock()
        
    #######################################
    # Lock Commands
    #######################################
    
    def _acquire_ambiguous_lock(self, lock: TicketLock):
        pid = get_global_context()
        if not lock.is_acquired_with(key=pid):
            lock.acquire(key=pid)
            
    def _release_ambiguous_lock(self, lock: TicketLock):
        pid = get_global_context()
        if lock.is_locked_with(key=pid):
            lock.release(key=pid)
            
    def _check_pid_is_locked_with(self, lock: TicketLock):
        pid = get_global_context()
        return lock.is_locked_with(key=pid)
    
    @core_command_method
    def _atom_acquire_buffer_lock(self, handle: BufferHandle):
        self._acquire_ambiguous_lock(handle.lock)

    @core_command_method
    def _atom_release_buffer_lock(self, handle: BufferHandle):
        self._release_ambiguous_lock(handle.lock)
            
    @core_command_method
    def _atom_acquire_mxu_lock(self):
        self._acquire_ambiguous_lock(self.mxu_lock)
            
    @core_command_method
    def _atom_release_mxu_lock(self):
        self._release_ambiguous_lock(self.mxu_lock)
    
    @core_command_method
    def _atom_acquire_vpu_lock(self):
        self._acquire_ambiguous_lock(self.vpu_lock)

    @core_command_method
    def _atom_release_vpu_lock(self):
        self._release_ambiguous_lock(self.vpu_lock)
        
    #######################################
    # Memory Access Commands
    #######################################
    
    @core_command_method
    def _atom_read_l1_mem(self, src_handle: BufferHandle, src_page_idx: int, tmp_dst_handle: TemporaryBufferHandle, tmp_dst_page_idx: int, n_pages: int):
        return self._check_pid_is_locked_with(src_handle.lock)

    @core_command_method
    def _atom_write_l1_mem(self, dst_handle: BufferHandle, dst_page_idx: int, tmp_src_handle: TemporaryBufferHandle, tmp_src_page_idx: int, n_pages: int):
        return self._check_pid_is_locked_with(dst_handle.lock)
        
    @core_kernel_method
    def read_l1_mem(self, src_handle: BufferHandle, src_page_idx: int, tmp_dst_handle: BufferHandle, tmp_dst_page_idx: int, n_pages: int):
        self._atom_acquire_buffer_lock(src_handle)
        self._atom_read_l1_mem(src_handle, src_page_idx, tmp_dst_handle, tmp_dst_page_idx, n_pages)
        self._atom_release_buffer_lock(src_handle)
        
    @core_kernel_method
    def write_l1_mem(self, dst_handle: BufferHandle, dst_page_idx: int, tmp_src_handle: BufferHandle, tmp_src_page_idx: int, n_pages: int):
        self._atom_acquire_buffer_lock(dst_handle)
        self._atom_write_l1_mem(dst_handle, dst_page_idx, tmp_src_handle, tmp_src_page_idx, n_pages)
        self._atom_release_buffer_lock(dst_handle)
            
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
        if not handle.check_vacancy(n_pages): 
            return False
        handle.allocate_cb_space(n_pages=n_pages)
    
    @core_command_method
    def cb_push_back(self, handle: CircularBufferHandle, n_pages: int):
        handle.occupy_cb_space(n_pages=n_pages)
        
    @core_command_method
    def cb_wait_front(self, handle: CircularBufferHandle, n_pages: int):
        if not handle.check_occupancy(n_pages): 
            return False
        
    @core_command_method
    def cb_pop_front(self, handle: CircularBufferHandle, n_pages: int):
        handle.deallocate_cb_space(n_pages=n_pages)
        
    #########################################
    # MXU: Matrix Multiplication Unit
    #########################################
    
    @core_command_method
    def _atom_mxu_tiled_gemm(
        self, 
        ifm_handle:  TemporaryBufferHandle, ifm_page_idx:  int, 
        wgt_handle:  TemporaryBufferHandle, wgt_page_idx:  int, 
        psum_handle: TemporaryBufferHandle, psum_page_idx: int,
        ofm_handle:  TemporaryBufferHandle, ofm_page_idx:  int,
        streaming_n_tiles: int  = 1,
        skip_wgt_preload:  bool = False,
        skip_psum_preload: bool = False,
        skip_ofm_flush:    bool = False,
    ):
        if not self._check_pid_is_locked_with(self.mxu_lock):
            return False
        
    #########################################
    # VPU: Vector Processing Unit
    #########################################
    
    @core_command_method
    def _atom_vpu_set_vector_reg(self, handle: TemporaryBufferHandle, page_idx: int, page_offset: int, vreg_idx: int, burst_len: int=1):
        if not self._check_pid_is_locked_with(self.vpu_lock):
            return False
        
    @core_command_method
    def _atom_vpu_get_vector_reg(self, handle: TemporaryBufferHandle, page_idx: int, page_offset: int, vreg_idx: int, burst_len: int=1):
        if not self._check_pid_is_locked_with(self.vpu_lock):
            return False
        
    @core_command_method
    def _atom_vpu_execute_vector_reg(self, opcode: VPUOperator, vreg_a: int, vreg_b: int=None, vreg_dest: int=None, inplace: bool=False, burst_len: int=1):
        if not self._check_pid_is_locked_with(self.vpu_lock):
            return False
        
class NPUCoreCycleModel(CoreCycleModel):
    def __init__(self, core: 'NPUCore'):
        super().__init__()
        
        self.core = core
    
    def _atom_read_l1_mem(self, src_handle: BufferHandle, src_page_idx: int, tmp_dst_handle: BufferHandle, tmp_dst_page_idx: int, n_pages: int):
        return self.core.mem_context.get_l1_mem_rd_latency(n_pages * src_handle.page_size)
    
    def _atom_write_l1_mem(self, dst_handle: BufferHandle, dst_page_idx: int, tmp_src_handle: BufferHandle, tmp_src_page_idx: int, n_pages: int):
        return self.core.mem_context.get_l1_mem_wr_latency(n_pages * dst_handle.page_size)
    
    def _atom_mxu_tiled_gemm(
        self, 
        ifm_handle:  TemporaryBufferHandle, ifm_page_idx:  int, 
        wgt_handle:  TemporaryBufferHandle, wgt_page_idx:  int, 
        psum_handle: TemporaryBufferHandle, psum_page_idx: int,
        ofm_handle:  TemporaryBufferHandle, ofm_page_idx:  int,
        streaming_n_tiles: int  = 1,
        skip_wgt_preload:  bool = False,
        skip_psum_preload: bool = False,
        skip_ofm_flush:    bool = False,
    ):
        total_cycles = 0
        
        if not skip_psum_preload:
            if self.core.mxu_context.dataflow == MXUDataflow.OS:
                total_cycles += self.core.mxu_context.get_preload_pe_arr_cycles()
            elif self.core.mxu_context.dataflow == MXUDataflow.WS:
                raise Exception(f"[ERROR] PSUM preload is not supported in WS dataflow")
        
        if not skip_wgt_preload:
            if self.core.mxu_context.dataflow == MXUDataflow.OS:
                raise Exception("[ERROR] WGT preload is not supported in OS dataflow.")
            elif self.core.mxu_context.dataflow == MXUDataflow.WS:
                total_cycles += self.core.mxu_context.get_preload_pe_arr_cycles()
                
        total_cycles += (self.core.mxu_context.get_execute_cycles() * streaming_n_tiles)
            
        if not skip_ofm_flush:
            if self.core.mxu_context.dataflow == MXUDataflow.OS:
                total_cycles += self.core.mxu_context.get_flush_pe_arr_cycles()
            elif self.core.mxu_context.dataflow == MXUDataflow.WS:
                total_cycles += self.core.mxu_context.get_flush_acc_regs_cycles() * streaming_n_tiles
                    
        return total_cycles
    
    def _atom_vpu_execute_vector_reg(self, opcode: VPUOperator, vreg_a: int, vreg_b: int=None, vreg_dest: int=None, inplace: bool=False, burst_len: int=1):
        if opcode.is_unary:
            return self.core.vpu_context.unary_op_latency * burst_len
        else:
            return self.core.vpu_context.arith_op_latency * burst_len

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
        
    def _atom_mxu_tiled_gemm(
        self, 
        ifm_handle:  TemporaryBufferHandle, ifm_page_idx:  int, 
        wgt_handle:  TemporaryBufferHandle, wgt_page_idx:  int, 
        psum_handle: TemporaryBufferHandle, psum_page_idx: int,
        ofm_handle:  TemporaryBufferHandle, ofm_page_idx:  int,
        streaming_n_tiles: int  = 1,
        skip_wgt_preload:  bool = False,
        skip_psum_preload: bool = False,
        skip_ofm_flush:    bool = False,
    ):  
        if not skip_psum_preload:
            if self.core.mxu_context.dataflow == MXUDataflow.OS:
                self.core.mxu_context.load_tile_pe_arr(psum_handle.data_get_page(psum_page_idx).content_view(shape=(32, 32), dtype=self.core.mxu_context.acc_dtype))
            elif self.core.mxu_context.dataflow == MXUDataflow.WS:
                raise Exception(f"[ERROR] PSUM preload is not supported in WS dataflow")
            
        if not skip_wgt_preload:
            if self.core.mxu_context.dataflow == MXUDataflow.OS:
                raise Exception("[ERROR] WGT preload is not supported in OS dataflow.")
            elif self.core.mxu_context.dataflow == MXUDataflow.WS:
                self.core.mxu_context.load_tile_pe_arr(wgt_handle.data_get_page(wgt_page_idx).content_view(shape=(32, 32), dtype=self.core.mxu_context.dtype))
            
        if self.core.mxu_context.dataflow == MXUDataflow.OS:
            for i in range(streaming_n_tiles):
                if self.core.mxu_context.dataflow == MXUDataflow.OS:
                    ifm_tile = ifm_handle.data_get_page(ifm_page_idx + i).content_view(shape=(32, 32), dtype=self.core.mxu_context.dtype)
                    wgt_tile = wgt_handle.data_get_page(wgt_page_idx + i).content_view(shape=(32, 32), dtype=self.core.mxu_context.dtype)
                    self.core.mxu_context.execute_gemm(ifm_tile=ifm_tile, wgt_tile=wgt_tile, psum_tile=None)
            
            if not skip_ofm_flush:
                ofm_tile = self.core.mxu_context.get_pe_arr_regs()
                ofm_handle.data_set_page(ofm_page_idx, ofm_tile)
        elif self.core.mxu_context.dataflow == MXUDataflow.WS:
            for i in range(streaming_n_tiles):
                ifm_tile = ifm_handle.data_get_page(ifm_page_idx + i).content_view(shape=(32, 32), dtype=self.core.mxu_context.dtype)
                psum_tile = psum_handle.data_get_page(psum_page_idx + i).content_view(shape=(32, 32), dtype=self.core.mxu_context.acc_dtype)
                self.core.mxu_context.execute_gemm(ifm_tile=ifm_tile, wgt_tile=None, psum_tile=psum_tile)
                
                if not skip_ofm_flush:
                    if streaming_n_tiles > 1:
                        raise Exception(f"[ERROR] It is impossible to skip OFM flush with WS dataflow when the number of streaming tiles is larger than 1")
                
                    ofm_tile = self.core.mxu_context.get_acc_regs()
                    ofm_handle.data_set_page(ofm_page_idx + i, ofm_tile)
                    
    def _atom_vpu_set_vector_reg(self, handle: TemporaryBufferHandle, page_idx: int, page_offset: int, vreg_idx: int, burst_len: int=1):
        for i in range(burst_len):
            st = page_offset + i * self.core.vpu_context.vlen
            ed = page_offset + (i + 1) * self.core.vpu_context.vlen
            vreg_data = handle.data_get_page(page_idx).content_view(shape=(-1,), dtype=self.core.vpu_context.vdtype)[st:ed]
            self.core.vpu_context.set_vector_reg(vreg_idx + i, vreg_data)
        
    def _atom_vpu_get_vector_reg(self, handle: TemporaryBufferHandle, page_idx: int, page_offset: int, vreg_idx: int, burst_len: int=1):
        for i in range(burst_len):
            st = page_offset + i * self.core.vpu_context.vlen
            ed = page_offset + (i + 1) * self.core.vpu_context.vlen
            vreg_data = self.core.vpu_context.get_vector_reg(vreg_idx + i)
            page = handle.data_get_page(page_idx)
            page.content_view(shape=(-1,), dtype=self.core.vpu_context.vdtype)[st:ed] = vreg_data

    def _atom_vpu_execute_vector_reg(self, opcode: VPUOperator, vreg_a: int, vreg_b: int=None, vreg_dest: int=None, inplace: bool=False, burst_len: int=1):
        for i in range(burst_len):
            vra = vreg_a + i
            vrb = vreg_b + i if vreg_b is not None else None
            vrd = vreg_dest + i if vreg_dest is not None else None
            self.core.vpu_context.execute_vector_op(opcode, vra, vrb, vrd, inplace=inplace)
        
if __name__ == "__main__":
    import numpy as np
    
    from neuromta.common.parser_utils import parse_mem_cap_str
    from neuromta.common.device import Device
    
    class MyDevice(Device):
        def __init__(self):
            super().__init__()
            
            self.mem_context = MemoryContext()
            self.icnt_context = IcntNetworkContext(grid_shape=(4, 4))
            self.mxu_config = MXUConfig(pe_arr_height=32, pe_arr_width=32, seq_len=32, dtype=np.int32, acc_dtype=np.int32, dataflow=MXUDataflow.OS, op_latency_per_byte=1)
            self.vpu_config = VPUConfig(vreg_len=parse_mem_cap_str("128B"), vreg_num=32, vdtype=np.int32, vlen_max=1024, vlen_min=32)

            self.npu_core = NPUCore(coord=(0, 0), mem_context=self.mem_context, icnt_context=self.icnt_context, mxu_config=self.mxu_config, vpu_config=self.vpu_config)
            self.dma_core = DMACore(coord=(0, 1), mem_context=self.mem_context, icnt_context=self.icnt_context)
            self.icnt_core = IcntNetworkCore(icnt_context=self.icnt_context)
            
    device = MyDevice()
    device.initialize(create_trace=False)
    device.change_sim_model_options(use_cycle_model=True, use_functional_model=True)
    
    device.npu_core.vpu_context.reconfigure_vector_reg_file(vlen=32, vdtype=np.int32)
    
    ifm = np.arange(0, 32*32, 1).astype(np.int32).reshape((32, 32))
    wgt = np.arange(0, 32*32, 1).astype(np.int32).reshape((32, 32))
    psum = np.arange(0, 32*32, 1).astype(np.int32).reshape((32, 32))
    
    ifm_handle = TemporaryBufferHandle(32*32*4, 1, [PageHandle(content=ifm),])
    wgt_handle = TemporaryBufferHandle(32*32*4, 1, [PageHandle(content=wgt),])
    psum_handle = TemporaryBufferHandle(32*32*4, 1, [PageHandle(content=psum),])
    ofm_handle = TemporaryBufferHandle(32*32*4, 1, [PageHandle(page_size=32*32*4),])
    
    @core_kernel_method
    def example_tiled_gemm_kernel(
        core: NPUCore,
        ifm_handle:  TemporaryBufferHandle, ifm_page_idx:  int, 
        wgt_handle:  TemporaryBufferHandle, wgt_page_idx:  int, 
        psum_handle: TemporaryBufferHandle, psum_page_idx: int,
        ofm_handle:  TemporaryBufferHandle, ofm_page_idx:  int,
    ):
        core._atom_acquire_mxu_lock()
        core._atom_mxu_tiled_gemm(
            ifm_handle, ifm_page_idx, 
            wgt_handle, wgt_page_idx, 
            psum_handle, psum_page_idx, 
            ofm_handle, ofm_page_idx,
            1, True, False, False
        )
        core._atom_release_mxu_lock()
        
        core._atom_acquire_vpu_lock()
        core._atom_vpu_set_vector_reg(ifm_handle, 0, 0, 0, 1)
        core._atom_vpu_set_vector_reg(wgt_handle, 0, 0, 1, 1)
        core._atom_vpu_execute_vector_reg(VPUOperator.ADD, 0, 1, 2, inplace=False, burst_len=1)
        core._atom_vpu_get_vector_reg(ofm_handle, 0, 0, 2, burst_len=1)
        core._atom_release_vpu_lock()
        
    example_tiled_gemm_kernel(device.npu_core, ifm_handle, 0, wgt_handle, 0, psum_handle, 0, ofm_handle, 0)
    
    device.verbose = True   # print debug messages
    device.run_kernels()
    
    reference = (ifm @ wgt) + psum
    reference[0, :] = ifm[0, :] + wgt[0, :]  # Simulate the VPU operation
    
    simulated = ofm_handle.data_get_page(0).content
    
    print("\n=== REFERENCE RESULT ===")
    print(reference)
    
    print("\n=== FUNCTIONAL SIMULATION RESULT ===")
    print(simulated)
    
    print(f"\nfunctional simulation validated: {np.allclose(reference, simulated)}")
    
    
        
# if __name__ == "__main__":
#     import numpy as np
    
#     from neuromta.common.parser_utils import parse_mem_cap_str
#     from neuromta.common.device import Device
    
#     class MyDevice(Device):
#         def __init__(self):
#             super().__init__()
            
#             self.mem_context = MemoryContext()
#             self.icnt_context = IcntNetworkContext(grid_shape=(4, 4))
#             self.mxu_config = MXUConfig(pe_arr_height=32, pe_arr_width=32, seq_len=256, acc_dtype=np.float32, dataflow=MXUDataflow.OS, op_latency_per_byte=1)
#             self.vpu_config = VPUConfig(vreg_len=parse_mem_cap_str("128B"), vreg_num=32, vlen_max=1024, vlen_min=32)

#             self.npu_core = NPUCore(coord=(0, 0), mem_context=self.mem_context, icnt_context=self.icnt_context, mxu_config=self.mxu_config, vpu_config=self.vpu_config)
#             self.dma_core = DMACore(coord=(0, 1), mem_context=self.mem_context, icnt_context=self.icnt_context)
#             self.icnt_core = IcntNetworkCore(icnt_context=self.icnt_context)
            
#     device = MyDevice()
#     device.initialize(create_trace=False)
#     device.change_sim_model_options(use_cycle_model=True, use_functional_model=False)
    
#     bf_handle = device.mem_context.create_buffer_handle("buffer1", addr=device.mem_context.get_main_mem_addr(0, 0, 0), page_size=32*32*4, n_pages=8)
#     cb_handle = device.mem_context.create_circular_buffer_handle("circular_buffer1", addr=bf_handle.addr + bf_handle.size, page_size=32*32*4, n_pages=8)
    
#     bf_handle.data_set_page(0, "DATA 1")
#     bf_handle.data_set_page(1, "DATA 2")
#     bf_handle.data_set_page(2, "DATA 3")
#     bf_handle.data_set_page(3, "DATA 4")
    
#     @core_kernel_method
#     def reader_kernel(core: NPUCore, bf_handle: BufferHandle, cb_handle: CircularBufferHandle) -> int:
#         core.cb_reserve_back(cb_handle, 2)
#         core.copy_page(src_handle=bf_handle, src_page_idx=0, dst_handle=cb_handle, dst_page_idx=0, n_pages=2)
#         core.cb_push_back(cb_handle, 2)
    
#     @core_kernel_method
#     def writer_kernel(core: NPUCore, bf_handle: BufferHandle, cb_handle: CircularBufferHandle) -> int:
#         core.cb_wait_front(cb_handle, 2)
#         core.copy_page(src_handle=cb_handle, src_page_idx=0, dst_handle=bf_handle, dst_page_idx=4, n_pages=2)
#         core.cb_pop_front(cb_handle, 2)
    
#     reader_kernel(device.npu_core, bf_handle, cb_handle)
#     writer_kernel(device.npu_core, bf_handle, cb_handle)
    
#     device.verbose = True   # print debug messages
#     device.run_kernels()
    
#     for i in range(bf_handle.n_pages):
#         print(f"Buffer Page {i}: {bf_handle.data_get_page(i)}")