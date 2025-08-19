import math
import enum
from typing import Any, Sequence

from neuromta.framework import *

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
        coord: tuple[int, int],
        mem_context: MemContext, 
        icnt_context: IcntContext,
        vpu_config: VPUConfig = VPUConfig(),
        mxu_config: MXUConfig = MXUConfig(),
    ):
        super().__init__(
            core_id=coord,  # coordinate is the core ID (it is guaranteed that each core is assigned with a unique coordinate in the given core grid!)
            cycle_model=NPUCoreCycleModel(core=self),
        )
        
        self.coord = coord
        self.mem_context = mem_context
        self.icnt_context = icnt_context
        
        self.local_mem_handle = MemoryHandle(
            mem_id=self.coord.__str__(), 
            base_addr=self.icnt_context.get_base_addr_from_coord(self.coord), 
            size=self.icnt_context._l1_mem_bank_size
        )
        
        self.mxu_context = mxu_config.create_context()
        self.vpu_context = vpu_config.create_context()
        
        self.mxu_lock = TicketLock()
        self.vpu_lock = TicketLock()

    #############################################################
    # Semaphore Management (Inter-Core Control Mechanism)
    #############################################################
        
    @core_command_method
    def sem_create(self, ptr: Pointer, initial_value: int=0):
        sem = self.local_mem_handle.allocate_variable_handle(initial_value=initial_value)
        ptr.var = sem
    
    @core_command_method
    def sem_remove(self, ptr: Pointer):
        self.local_mem_handle.deallocate_variable_handle(ptr.var)
        ptr.clear()
        
    @core_command_method
    def sem_set_value(self, ptr: Pointer, value: int):
        ptr.var.value = value

    @core_command_method
    def sem_increase(self, ptr: Pointer, value: int=1):
        ptr.var.value += value

    @core_command_method
    def sem_decrease(self, ptr: Pointer, value: int=1):
        ptr.var.value -= value

    @core_command_method
    def sem_wait(self, ptr: Pointer, value: int=0):
        return ptr.var.value == value

    #############################################################
    # Circular Buffer Management (Intra-Core Control Management)
    #############################################################
    
    @core_command_method
    def cb_create(self, ptr: Pointer, page_size: int, n_pages: int):
        pages = self.local_mem_handle.allocate_page_handles(page_size=page_size, n_pages=n_pages)
        ptr.handle = CircularBufferHandle(page_size=page_size, pages=pages)
    
    @core_command_method
    def cb_remove(self, ptr: Pointer):
        cb_handle: CircularBufferHandle = ptr.handle
        self.local_mem_handle.deallocate_page_handles(cb_handle.pages)
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
    # NoC
    #############################################################
    
    @core_command_method
    def noc_control_packet_transfer(self, dst_core_coord: tuple[int, int]):
        pass  # TODO: currently, NoC simulator is not implemented, BookSim2 will be integrated later.
    
    @core_command_method
    def noc_data_packet_transfer(self, dst_core_coord: tuple[int, int], data_size: int):
        pass  # TODO: currently, NoC simulator is not implemented, BookSim2 will be integrated later.
        
    #############################################################
    # Buffer Management
    #############################################################
    
    @core_command_method
    def local_buffer_allocate(self, ptr: Pointer, page_size: int, n_pages: int):
        pages = self.local_mem_handle.allocate_page_handles(page_size=page_size, n_pages=n_pages)
        ptr.handle = BufferHandle(page_size=page_size, pages=pages)
        
    @core_command_method
    def local_buffer_deallocate(self, ptr: Pointer):
        buff_handle: BufferHandle = ptr.handle
        self.local_mem_handle.deallocate_page_handles(buff_handle.pages)
        ptr.clear()
        
    @core_command_method
    def local_memcopy_page(self, dst_ptr: Pointer, src_ptr: Pointer):
        if dst_ptr.ptr_type != PointerType.PAGE or src_ptr.ptr_type != PointerType.PAGE:
            raise ValueError("[ERROR] Memory copy requires page pointers.")
        
        dst_handle: PageHandle = dst_ptr.handle
        src_handle: PageHandle = src_ptr.handle

        if dst_handle.size != src_handle.size:
            raise ValueError(f"[ERROR] Page sizes do not match: {dst_handle.size} != {src_handle.size}")

        dst_handle.copy_from(src_handle)
        
    @core_kernel_method
    def local_memcopy_buffer(self, dst_ptr: Pointer, dst_offset_page_idx: int, src_ptr: Pointer, src_offset_page_idx: int, n_pages: int):
        if dst_ptr.ptr_type != PointerType.BUFFER or src_ptr.ptr_type != PointerType.BUFFER:
            raise ValueError("[ERROR] Memory copy requires buffer pointers.")
        
        dst_st = dst_offset_page_idx
        dst_ed = dst_st + n_pages
        
        src_st = src_offset_page_idx
        src_ed = src_st + n_pages
        
        dst_ptrs = dst_ptr[dst_st:dst_ed]
        src_ptrs = src_ptr[src_st:src_ed]

        for dst_ptr, src_ptr in zip(dst_ptrs, src_ptrs):
            self.local_memcopy_page(dst_ptr, src_ptr)
            
    @core_kernel_method
    def _local_memcopy_page_and_transfer(self, dst_ptr: Pointer, src_ptr: Pointer):
        if dst_ptr.ptr_type != PointerType.PAGE or src_ptr.ptr_type != PointerType.PAGE:
            raise ValueError("[ERROR] Memory copy requires page pointers.")

        dst_owner_id = self.icnt_context.get_coord_from_address(dst_ptr.handle.addr)

        self.local_memcopy_page(dst_ptr, src_ptr)
        self.noc_data_packet_transfer(dst_core_coord=dst_owner_id, data_size=src_ptr.handle.size)

    #############################################################
    # Remote
    #############################################################

    @core_kernel_method
    def remote_memcopy_page(self, dst_ptr: Pointer, src_ptr: Pointer):
        if dst_ptr.ptr_type != PointerType.PAGE or src_ptr.ptr_type != PointerType.PAGE:
            raise ValueError("[ERROR] Memory copy requires page pointers.")
        
        src_owner_id = self.icnt_context.get_coord_from_address(src_ptr.handle.addr)
        
        if src_owner_id is None:
            raise Exception(f"[ERROR] Invalid destination core ID: {src_owner_id}. The address {src_ptr.handle.addr} does not map to a valid core.")

        msg = RPCMessage(
            msg_type=0,
            src_core_id=self.core_id,
            dst_core_id=src_owner_id,    # the destination of the RPC message is the source of the data 
            kernel_id=get_global_kernel_context().kernel_id,
            cmd_id="_local_memcopy_page_and_transfer",  # the remote core will read its local memory and transfer the data packet to myself!
            src_ptr=src_ptr,
            dst_ptr=dst_ptr,
        )
        
        self.noc_control_packet_transfer(dst_core_coord=src_owner_id)
        self.remote_rpc_send_request_msg(msg)
        
    @core_kernel_method
    def remote_memcopy_buffer(self, dst_ptr: Pointer, dst_offset_page_idx: int, src_ptr: Pointer, src_offset_page_idx: int, n_pages: int):
        if dst_ptr.ptr_type != PointerType.BUFFER or src_ptr.ptr_type != PointerType.BUFFER:
            raise ValueError("[ERROR] Memory copy requires buffer pointers.")
        
        dst_st = dst_offset_page_idx
        dst_ed = dst_st + n_pages
        
        src_st = src_offset_page_idx
        src_ed = src_st + n_pages
        
        dst_ptrs = dst_ptr[dst_st:dst_ed]
        src_ptrs = src_ptr[src_st:src_ed]

        for dst_ptr, src_ptr in zip(dst_ptrs, src_ptrs):
            self.remote_memcopy_page(dst_ptr, src_ptr)
            
    @core_kernel_method
    def remote_sem_set_value(self, ptr: Pointer, value: int):
        if ptr.ptr_type != PointerType.VARIABLE:
            raise ValueError("[ERROR] Semaphore set value requires semaphore pointer.")
        
        msg = RPCMessage(
            msg_type=0,
            src_core_id=self.core_id,
            dst_core_id=self.icnt_context.get_coord_from_address(ptr.handle.addr),
            kernel_id=get_global_kernel_context().kernel_id,
            cmd_id="sem_set_value",
            ptr=ptr,
            value=value
        )
        
        self.noc_control_packet_transfer(dst_core_coord=msg.dst_core_id)
        self.remote_rpc_send_request_msg(msg)
        
    @core_kernel_method
    def remote_sem_increase(self, ptr: Pointer, value: int=1):
        if ptr.ptr_type != PointerType.VARIABLE:
            raise ValueError("[ERROR] Semaphore increase requires semaphore pointer.")
        
        msg = RPCMessage(
            msg_type=0,
            src_core_id=self.core_id,
            dst_core_id=self.icnt_context.get_coord_from_address(ptr.handle.addr),
            kernel_id=get_global_kernel_context().kernel_id,
            cmd_id="sem_increase",
            ptr=ptr,
            value=value
        )
        
        self.noc_control_packet_transfer(dst_core_coord=msg.dst_core_id)
        self.remote_rpc_send_request_msg(msg)
        
    @core_kernel_method
    def remote_sem_decrease(self, ptr: Pointer, value: int=1):
        if ptr.ptr_type != PointerType.VARIABLE:
            raise ValueError("[ERROR] Semaphore decrease requires semaphore pointer.")

        msg = RPCMessage(
            msg_type=0,
            src_core_id=self.core_id,
            dst_core_id=self.icnt_context.get_coord_from_address(ptr.handle.addr),
            kernel_id=get_global_kernel_context().kernel_id,
            cmd_id="sem_decrease",
            ptr=ptr,
            value=value
        )

        self.noc_control_packet_transfer(dst_core_coord=msg.dst_core_id)
        self.remote_rpc_send_request_msg(msg)

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
        
    def noc_control_packet_transfer(self, dst_core_coord: tuple[int, int]):
        return self.core.icnt_context.get_control_packet_latency(src_coord=self.core.coord, dst_coord=dst_core_coord)   # TODO: simple NoC latency model, further be refined by BookSim2 
    
    def noc_data_packet_transfer(self, dst_core_coord: tuple[int, int], data_size: int):
        return self.core.icnt_context.get_data_packet_latency(src_coord=self.core.coord, dst_coord=dst_core_coord, data_size=data_size)   # TODO: simple NoC latency model, further be refined by BookSim2

    def local_memcopy_page(self, dst_ptr: Pointer, src_ptr: Pointer):
        if dst_ptr.ptr_type != PointerType.PAGE or src_ptr.ptr_type != PointerType.PAGE:
            raise ValueError("[ERROR] Memory copy requires page pointers.")
        
        src_handle: PageHandle = src_ptr.handle
        
        return self.core.mem_context.l1_config.get_cycles(size=src_handle.size)
    
    def remote_memcopy_page(self, dst_ptr: Pointer, src_ptr: Pointer):
        if dst_ptr.ptr_type != PointerType.PAGE or src_ptr.ptr_type != PointerType.PAGE:
            raise ValueError("[ERROR] Memory copy requires page pointers.")
        
        src_handle: PageHandle = src_ptr.handle
        src_owner_coord = self.core.icnt_context.get_coord_from_address(src_handle.addr)

        return self.core.icnt_context.get_control_packet_latency(src_coord=self.core.coord, dst_coord=src_owner_coord, data_size=src_handle.size)

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

