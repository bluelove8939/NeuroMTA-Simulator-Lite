import math
import enum
from typing import Any, Sequence

from neuromta.common import *

from neuromta.hardware.mem_context import MemContext
from neuromta.hardware.icnt_context import IcntContext
from neuromta.hardware.vpu_context import VPUContext, VPUConfig, VPUOperator
from neuromta.hardware.mxu_context import MXUContext, MXUConfig, MXUDataflow
import torch


__all__ = [
    "NPUCore",
] 


class NPUCore(Core):
    def __init__(
        self,
        core_id: str,
        coord: tuple[int, int],
        mem_context: MemContext, 
        icnt_context: IcntContext,
        vpu_config: VPUConfig = VPUConfig(),
        mxu_config: MXUConfig = MXUConfig(),
    ):
        super().__init__(
            # core_id=DEFAULT_CORE_ID, 
            core_id=core_id,
            cycle_model=NPUCoreCycleModel(core=self),
        )
        
        self.coord = coord
        self.mem_context = mem_context
        self.icnt_context = icnt_context
        
        self.mem_handle = MemoryHandle(mem_id=self.coord.__str__(), base_addr=0, size=self.icnt_context._l1_mem_bank_size)
        
        self.mxu_context = mxu_config.create_context()
        self.vpu_context = vpu_config.create_context()
        
        self.mxu_lock = TicketLock()
        self.vpu_lock = TicketLock()

    #############################################################
    # Semaphore Management (Inter-Core Control Mechanism)
    #############################################################
        
    @core_command_method
    def sem_create(self, ptr: Pointer, initial_value: int=0):
        ptr.sem = Semaphore(initial_value)
    
    @core_command_method
    def sem_remove(self, ptr: Pointer):
        ptr.clear()
        
    @core_command_method
    def sem_set_value(self, ptr: Pointer, value: int):
        ptr.sem.value = value

    @core_command_method
    def sem_increase(self, ptr: Pointer, value: int=1):
        ptr.sem.value += value

    @core_command_method
    def sem_decrease(self, ptr: Pointer, value: int=1):
        ptr.sem.value -= value

    @core_command_method
    def sem_wait(self, ptr: Pointer, value: int=0):
        return ptr.sem.value == value

    #############################################################
    # Circular Buffer Management (Intra-Core Control Management)
    #############################################################
    
    @core_command_method
    def cb_create(self, ptr: Pointer, page_size: int, n_pages: int):
        pages = self.mem_handle.allocate_page_handles(page_size=page_size, n_pages=n_pages)
        ptr.handle = CircularBufferHandle(page_size=page_size, pages=pages)
    
    @core_command_method
    def cb_remove(self, ptr: Pointer):
        cb_handle: CircularBufferHandle = ptr.handle
        self.mem_handle.deallocate_page_handles(cb_handle.pages)
        ptr.clear()

    @core_command_method
    def cb_reserve_back(self, ptr: Pointer, n_pages: int):
        cb_handle: CircularBufferHandle = ptr.handle
        if not cb_handle.check_vacancy(n_pages):
            return False
        cb_handle.allocate_cb_space(n_pages)
        
    @core_command_method
    def cb_push_back(self, ptr: Pointer, n_pages: int):
        cb_handle: CircularBufferHandle = ptr.handle
        cb_handle.occupy_cb_space(n_pages)
        
    @core_command_method
    def cb_wait_front(self, ptr: Pointer, n_pages: int):
        cb_handle: CircularBufferHandle = ptr.handle
        return cb_handle.check_occupancy(n_pages)
    
    @core_command_method
    def cb_pop_front(self, ptr: Pointer, n_pages: int):
        cb_handle: CircularBufferHandle = ptr.handle
        cb_handle.deallocate_cb_space(n_pages)
        
    #############################################################
    # Buffer Management (Main Memory Management)
    #############################################################
    
    @core_command_method
    def l1_buff_create(self, ptr: Pointer, page_size: int, n_pages: int):
        pages = self.mem_handle.allocate_page_handles(page_size=page_size, n_pages=n_pages)
        ptr.handle = BufferHandle(page_size=page_size, pages=pages)
        
    @core_command_method
    def l1_buff_remove(self, ptr: Pointer):
        buff_handle: BufferHandle = ptr.handle
        self.mem_handle.deallocate_page_handles(buff_handle.pages)
        ptr.clear()
        
    @core_command_method
    def copy_page(self, src_ptr: Pointer, dst_ptr: Pointer):
        if src_ptr.ptr_type != PointerType.PAGE or dst_ptr.ptr_type != PointerType.PAGE:
            raise Exception("[ERROR] copy_pages can only be used with PAGE pointers.")
        
        if not self.use_functional_model:
            return  # Terminate the command to reduce the simulation time without actual MXU functional unit (do not return anything to make sure that the command is executed only once)

        src_page: PageHandle = src_ptr.handle
        dst_page: PageHandle = dst_ptr.handle
        dst_page.copy_from(src_page)
        
    @core_kernel_method
    def copy_buffer(self, src_ptr: Pointer, dst_ptr: Pointer, src_offset_idx: int, dst_offset_idx: int, n_pages: int, parallel: bool = False):
        if src_ptr.ptr_type != PointerType.BUFFER or dst_ptr.ptr_type != PointerType.BUFFER:
            raise Exception("[ERROR] copy_buffers can only be used with BUFFER pointers.")

        if not self.use_functional_model:
            return  # Terminate the command to reduce the simulation time without actual MXU functional unit (do not return anything to make sure that the command is executed only once)

        for i in range(n_pages):
            if parallel:
                start_parallel_thread()
                
            self.copy_page(src_ptr[src_offset_idx + i], dst_ptr[dst_offset_idx + i])
            
            if parallel:
                end_parallel_thread()

    #############################################################
    # Lock Management
    #############################################################
    
    def _acquire_ambiguous_lock(self, lock: TicketLock):
        pid = get_global_pid()
        if not lock.is_acquired_with(key=pid):
            lock.acquire(key=pid)
            
    def _release_ambiguous_lock(self, lock: TicketLock):
        pid = get_global_pid()
        if lock.is_locked_with(key=pid):
            lock.release(key=pid)
            
    def _check_pid_is_locked_with(self, lock: TicketLock):
        pid = get_global_pid()
        return lock.is_locked_with(key=pid)
        
    #############################################################
    # MXU Commands
    #############################################################
    
    @core_command_method
    def mxu_acquire_lock(self):
        self._acquire_ambiguous_lock(self.mxu_lock)
            
    @core_command_method
    def mxu_release_lock(self):
        self._release_ambiguous_lock(self.mxu_lock)
        
    @core_command_method
    def mxu_reconfigure(self, dtype: torch.dtype, acc_dtype: torch.dtype):
        if not self._check_pid_is_locked_with(self.mxu_lock):
            return False
        
        self.mxu_context.reconfigure_dtype(dtype=dtype, acc_dtype=acc_dtype)
    
    @core_command_method
    def mxu_tiled_gemm(
        self, 
        ifm_ptr:  Pointer,
        wgt_ptr:  Pointer,
        psum_ptr: Pointer,
        ofm_ptr:  Pointer,
        preload_wgt:   bool,
        preload_psum:  bool,
        flush_ofm:     bool,
    ):
        if not self._check_pid_is_locked_with(self.mxu_lock):
            return False
        
        if not self.use_functional_model:
            return  # Terminate the command to reduce the simulation time without actual MXU functional unit (do not return anything to make sure that the command is executed only once)
        
        if preload_psum:
            if self.mxu_context.dataflow == MXUDataflow.OS:
                self.mxu_context.load_tile_pe_arr(psum_ptr.handle.content_view(shape=self.mxu_context.ofm_tile_shape, dtype=self.mxu_context.acc_dtype))       
            elif self.mxu_context.dataflow == MXUDataflow.WS:
                raise Exception(f"[ERROR] PSUM preload is not supported in WS dataflow")    
        
        if preload_wgt:
            if self.mxu_context.dataflow == MXUDataflow.OS:
                raise Exception("[ERROR] WGT preload is not supported in OS dataflow.")
            elif self.mxu_context.dataflow == MXUDataflow.WS:
                self.mxu_context.load_tile_pe_arr(wgt_ptr.handle.content_view(shape=self.mxu_context.wgt_tile_shape, dtype=self.mxu_context.dtype))
                
        if self.mxu_context.dataflow == MXUDataflow.OS:
            ifm_tile = ifm_ptr.handle.content_view(shape=self.mxu_context.ifm_tile_shape, dtype=self.mxu_context.dtype)
            wgt_tile = wgt_ptr.handle.content_view(shape=self.mxu_context.wgt_tile_shape, dtype=self.mxu_context.dtype)

            self.mxu_context.execute_gemm(ifm_tile=ifm_tile, wgt_tile=wgt_tile)
        elif self.mxu_context.dataflow == MXUDataflow.WS:
            ifm_tile = ifm_ptr.handle.content_view(shape=self.mxu_context.ifm_tile_shape, dtype=self.mxu_context.dtype)
            psum_tile = psum_ptr.handle.content_view(shape=self.mxu_context.ofm_tile_shape, dtype=self.mxu_context.acc_dtype)

            self.mxu_context.execute_gemm(ifm_tile=ifm_tile, psum_tile=psum_tile)
            
        if flush_ofm:
            if self.mxu_context.dataflow == MXUDataflow.OS:
                psum_tile = self.mxu_context.get_pe_arr_regs()   
            elif self.mxu_context.dataflow == MXUDataflow.WS:
                psum_tile = self.mxu_context.get_acc_regs() 
            
            ofm_page: PageHandle | BufferHandle = ofm_ptr.handle
            ofm_page.set_content(psum_tile)
            
    #############################################################
    # VPU Commands
    #############################################################
    
    @core_command_method
    def vpu_acquire_lock(self):
        self._acquire_ambiguous_lock(self.vpu_lock)

    @core_command_method
    def vpu_release_lock(self):
        self._release_ambiguous_lock(self.vpu_lock)
        
    @core_command_method
    def vpu_reconfigure(self, vlen: int, vdtype: torch.dtype):
        if not self._check_pid_is_locked_with(self.vpu_lock):
            return False
        
        self.vpu_context.reconfigure_vector_reg_file(vlen=vlen, vdtype=vdtype)
        
    @core_command_method
    def vpu_load_reg(self, ptr: Pointer, ptr_offset: int, vreg_idx: int, burst_len: int=1):
        if not self._check_pid_is_locked_with(self.vpu_lock):
            return False
        if not self.use_functional_model:
            return  # Terminate the command to reduce the simulation time without actual VPU functional unit (do not return anything to make sure that the command is executed only once)
        
        for i in range(burst_len):
            st = ptr_offset + i * self.vpu_context.vlen
            ed = ptr_offset + (i + 1) * self.vpu_context.vlen
            vreg_data = ptr.handle.content_view(shape=(-1,), dtype=self.vpu_context.vdtype)[st:ed]
            self.vpu_context.set_vector_reg(vreg_idx + i, vreg_data)
        
    @core_command_method
    def vpu_store_reg(self, ptr: Pointer, ptr_offset: int, vreg_idx: int, burst_len: int=1):
        if not self._check_pid_is_locked_with(self.vpu_lock):
            return False
        if not self.use_functional_model:
            return  # Terminate the command to reduce the simulation time without actual VPU functional unit (do not return anything to make sure that the command is executed only once)

        for i in range(burst_len):
            offset = (ptr_offset + i * self.vpu_context.vlen) * self.vpu_context.vdtype.itemsize
            vreg_data = self.vpu_context.get_vector_reg(vreg_idx + i)
            ptr.handle.set_content(vreg_data, offset=offset)

    @core_command_method
    def vpu_execute(self, opcode: VPUOperator, vreg_a: int, vreg_b: int=None, vreg_dest: int=None, inplace: bool=False, burst_len: int=1):
        if not self._check_pid_is_locked_with(self.vpu_lock):
            return False
        if not self.use_functional_model:
            return  # Terminate the command to reduce the simulation time without actual VPU functional unit (do not return anything to make sure that the command is executed only once)
        
        for i in range(burst_len):
            vra = vreg_a + i
            vrb = vreg_b + i if vreg_b is not None else None
            vrd = vreg_dest + i if vreg_dest is not None else None
            self.vpu_context.execute_vector_op(opcode, vra, vrb, vrd, inplace=inplace)

class NPUCoreCycleModel(CoreCycleModel):
    def __init__(self, core: 'NPUCore'):
        super().__init__()
        
        self.core = core
        
    def copy_page(self, src_ptr: Pointer, dst_ptr: Pointer):
        if src_ptr.ptr_type != PointerType.PAGE or dst_ptr.ptr_type != PointerType.PAGE:
            raise Exception("[ERROR] copy_pages can only be used with PAGE pointers.")
        
        src_page: PageHandle = src_ptr.handle
        dst_page: PageHandle = dst_ptr.handle
        
        if src_page.size != dst_page.size:
            raise Exception(f"[ERROR] Source page size ({src_page.size}) does not match destination page size ({dst_page.size}).")
        
        src_coord = self.core.icnt_context.get_coord_with_mem_base_addr(src_ptr.get_base_addr())
        dst_coord = self.core.icnt_context.get_coord_with_mem_base_addr(dst_ptr.get_base_addr())

        icnt_cycles = 0
        mem_cycles = 0

        if src_coord != dst_coord:
            icnt_cycles += self.core.icnt_context.get_control_packet_latency(dst_coord, src_coord)
            icnt_cycles += self.core.icnt_context.get_data_packet_latency(src_coord, dst_coord, data_size=src_page.size)

        if self.core.icnt_context.check_mem_handle_is_l1(src_ptr.get_mem_handle()):
            mem_cycles += self.core.mem_context.l1_config.get_cycles(size=src_page.size)
        else:
            mem_cycles += self.core.mem_context.main_config.get_cycles(size=src_page.size)
        
        if self.core.icnt_context.check_mem_handle_is_l1(dst_ptr.get_mem_handle()):
            mem_cycles += self.core.mem_context.l1_config.get_cycles(size=dst_page.size)
        else:
            mem_cycles += self.core.mem_context.main_config.get_cycles(size=dst_page.size)

        return icnt_cycles + mem_cycles
    
    def mxu_tiled_gemm(
        self, 
        ifm_ptr:  Pointer,
        wgt_ptr:  Pointer,
        psum_ptr: Pointer,
        ofm_ptr:  Pointer,
        preload_wgt:   bool,
        preload_psum:  bool,
        flush_ofm:     bool,
    ):
        total_cycles = 0
        
        if preload_psum:
            if self.core.mxu_context.dataflow == MXUDataflow.OS:
                total_cycles += self.core.mxu_context.get_preload_pe_arr_cycles()
            elif self.core.mxu_context.dataflow == MXUDataflow.WS:
                raise Exception(f"[ERROR] PSUM preload is not supported in WS dataflow")

        if preload_wgt:
            if self.core.mxu_context.dataflow == MXUDataflow.OS:
                raise Exception("[ERROR] WGT preload is not supported in OS dataflow.")
            elif self.core.mxu_context.dataflow == MXUDataflow.WS:
                total_cycles += self.core.mxu_context.get_preload_pe_arr_cycles()
                
        total_cycles += self.core.mxu_context.get_execute_cycles()

        if flush_ofm:
            if self.core.mxu_context.dataflow == MXUDataflow.OS:
                total_cycles += self.core.mxu_context.get_flush_pe_arr_cycles()
            elif self.core.mxu_context.dataflow == MXUDataflow.WS:
                total_cycles += self.core.mxu_context.get_flush_acc_regs_cycles()
                    
        return total_cycles
    
    def vpu_execute(self, opcode: VPUOperator, vreg_a: int, vreg_b: int=None, vreg_dest: int=None, inplace: bool=False, burst_len: int=1):
        if opcode.is_unary:
            return self.core.vpu_context.unary_op_latency * burst_len
        else:
            return self.core.vpu_context.arith_op_latency * burst_len

