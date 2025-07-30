import math
from typing import Sequence

from neuromta.common.core import *
from neuromta.common.custom_types import MemoryType
from neuromta.common.parser_utils import parse_mem_cap_str

from neuromta.ip.hardware.synchronizer import TicketLock
from neuromta.ip.hardware.buffer_handle import CircularBufferHandle, BufferHandle, TemporaryBufferHandle
from neuromta.ip.hardware.main_memory_config import MainMemoryConfig, HBM2Config


__all__ = [
    "MemoryContext",
    "NPUCore",
]


################################################
# Hardware Context
################################################  

class MemoryContext:
    def __init__(
        self,
        
        l1_mem_base:        int = 0x00000000,
        l1_mem_bank_num:    int = 8,
        l1_mem_bank_size:   int = parse_mem_cap_str("128KB"),
        l1_mem_access_gran: int = parse_mem_cap_str("32B"),
        
        main_mem_base:      int = 0x10000000,
        main_mem_bank_num:  int = 8,
        main_mem_bank_size: int = parse_mem_cap_str("1GB"),
        main_mem_config:    MainMemoryConfig = HBM2Config(),
        
        l1_mem_rd_latency_per_block: int = 1,
        l1_mem_wr_latency_per_block: int = 2,
    ):
        self._l1_mem_base           = l1_mem_base
        self._l1_mem_seg_num        = 0
        self._l1_mem_bank_num       = l1_mem_bank_num
        self._l1_mem_bank_size      = l1_mem_bank_size
        self._l1_mem_access_gran    = l1_mem_access_gran
        
        self._main_mem_base         = main_mem_base
        self._main_mem_seg_num      = 0
        self._main_mem_bank_num     = main_mem_bank_num
        self._main_mem_bank_size    = main_mem_bank_size
        self._main_mem_config       = main_mem_config

        self._l1_mem_rd_latency_per_block   = l1_mem_rd_latency_per_block
        self._l1_mem_wr_latency_per_block   = l1_mem_wr_latency_per_block
        
        self._buffer_handles: dict[str, BufferHandle | CircularBufferHandle] = {}
        
    ###############################################
    # Memory Context Manipulation
    ###############################################
    
    def add_new_l1_mem_segment(self) -> int:
        seg_id = self._l1_mem_seg_num
        self._l1_mem_seg_num += 1
        return seg_id
    
    def add_new_main_mem_segment(self) -> int:
        seg_id = self._main_mem_seg_num
        self._main_mem_seg_num += 1
        return seg_id
        
    @property
    def _l1_mem_seg_size(self) -> int:
        return self._l1_mem_bank_size * self._l1_mem_bank_num
    
    @property
    def _l1_mem_total_size(self) -> int:
        return self._l1_mem_seg_size * self._l1_mem_seg_num
    
    @property
    def _main_mem_seg_size(self) -> int:
        return self._main_mem_bank_size * self._main_mem_bank_num

    @property
    def _main_mem_total_size(self) -> int:
        return self._main_mem_seg_size * self._main_mem_seg_num

    ###############################################
    # Address Checking and Parsing
    ###############################################
        
    def check_l1_mem_addr(self, addr: int) -> bool:
        return self._l1_mem_base <= addr < self._l1_mem_base + self._l1_mem_total_size
    
    def check_main_mem_addr(self, addr: int) -> bool:
        return self._main_mem_base <= addr < self._main_mem_base + self._main_mem_total_size

    def parse_l1_mem_addr(self, addr: int) -> tuple[int, int, int]:
        seg_id = (addr - self._l1_mem_base) // self._l1_mem_seg_size
        bank_id = (addr - self._l1_mem_base) % self._l1_mem_seg_size // self._l1_mem_bank_size
        offset = (addr - self._l1_mem_base) % self._l1_mem_seg_size % self._l1_mem_bank_size

        if addr < self._l1_mem_base or addr >= self._l1_mem_base + self._l1_mem_total_size:
            raise Exception(f"[ERROR] L1 memory address {addr:#x} out of bounds.")
        if seg_id < 0 or seg_id >= self._l1_mem_seg_num:
            raise Exception(f"[ERROR] L1 memory segment ID {seg_id} out of bounds. Must be in range [0, {self._l1_mem_seg_num - 1}].")
        if bank_id < 0 or bank_id >= self._l1_mem_bank_num:
            raise Exception(f"[ERROR] L1 memory bank ID {bank_id} out of bounds. Must be in range [0, {self._l1_mem_bank_num - 1}].")
        
        return seg_id, bank_id, offset
    
    def parse_main_mem_addr(self, addr: int) -> tuple[int, int, int]:
        seg_id = (addr - self._main_mem_base) // self._main_mem_seg_size
        bank_id = (addr - self._main_mem_base) % self._main_mem_seg_size // self._main_mem_bank_size
        offset = (addr - self._main_mem_base) % self._main_mem_seg_size % self._main_mem_bank_size

        if addr < self._main_mem_base or addr >= self._main_mem_base + self._main_mem_total_size:
            raise Exception(f"[ERROR] Main memory address {addr:#x} out of bounds.")
        if seg_id < 0 or seg_id >= self._main_mem_seg_num:
            raise Exception(f"[ERROR] Main memory segment ID {seg_id} out of bounds. Must be in range [0, {self._main_mem_seg_num - 1}].")
        if bank_id < 0 or bank_id >= self._main_mem_bank_num:
            raise Exception(f"[ERROR] Main memory bank ID {bank_id} out of bounds. Must be in range [0, {self._main_mem_bank_num - 1}].")
        
        return seg_id, bank_id, offset
    
    def get_l1_mem_addr(self, seg_id: int, bank_id: int, offset: int) -> int:
        if seg_id < 0 or seg_id >= self._l1_mem_seg_num:
            raise Exception(f"[ERROR] L1 memory segment ID {seg_id} out of bounds. Must be in range [0, {self._l1_mem_seg_num - 1}].")
        if bank_id < 0 or bank_id >= self._l1_mem_bank_num:
            raise Exception(f"[ERROR] L1 memory bank ID {bank_id} out of bounds. Must be in range [0, {self._l1_mem_bank_num - 1}].")
        if offset < 0 or offset >= self._l1_mem_bank_size:
            raise Exception(f"[ERROR] Offset {offset} out of bounds for L1 memory bank size {self._l1_mem_bank_size}.")
        
        return self._l1_mem_base + seg_id * self._l1_mem_seg_size + bank_id * self._l1_mem_bank_size + offset
    
    def get_main_mem_addr(self, seg_id: int, bank_id: int, offset: int) -> int:
        if seg_id < 0 or seg_id >= self._main_mem_seg_num:
            raise Exception(f"[ERROR] Main memory segment ID {seg_id} out of bounds. Must be in range [0, {self._main_mem_seg_num - 1}].")
        if bank_id < 0 or bank_id >= self._main_mem_bank_num:
            raise Exception(f"[ERROR] Main memory bank ID {bank_id} out of bounds. Must be in range [0, {self._main_mem_bank_num - 1}].")
        if offset < 0 or offset >= self._main_mem_bank_size:
            raise Exception(f"[ERROR] Offset {offset} out of bounds for main memory bank size {self._main_mem_bank_size}.")
        
        return self._main_mem_base + seg_id * self._main_mem_seg_size + bank_id * self._main_mem_bank_size + offset

    ###############################################
    # Memory Access Latency
    ###############################################
    
    def get_l1_mem_rd_latency(self, size: int) -> int:
        return self._l1_mem_rd_latency_per_block * math.ceil(size / self._l1_mem_bank_num / self._l1_mem_access_gran)
    
    def get_l1_mem_wr_latency(self, size: int) -> int:
        return self._l1_mem_wr_latency_per_block * math.ceil(size / self._l1_mem_bank_num / self._l1_mem_access_gran)
    
    def get_main_mem_rd_latency(self, size: int) -> int:
        return self._main_mem_config.get_cycles(size)
    
    def get_main_mem_wr_latency(self, size: int) -> int:
        return self._main_mem_config.get_cycles(size)

    ###############################################
    # Buffer Handle Management
    ###############################################
    
    def create_buffer_handle(self, buffer_id: str, addr: int, page_size: int, n_pages: int) -> BufferHandle:
        if buffer_id in self._buffer_handles:
            raise Exception(f"[ERROR] Buffer handle with ID '{buffer_id}' already exists.")
        if not (self.check_main_mem_addr(addr) or self.check_l1_mem_addr(addr)):
            raise Exception(f"[ERROR] Address {addr:#x} is out of bounds for buffer handle creation.")
        
        handle = BufferHandle(buffer_id, addr, page_size, n_pages)
        self._buffer_handles[buffer_id] = handle
        return handle
    
    def create_circular_buffer_handle(self, buffer_id: str, addr: int, page_size: int, n_pages: int) -> CircularBufferHandle:
        if buffer_id in self._buffer_handles:
            raise Exception(f"[ERROR] Circular buffer handle with ID '{buffer_id}' already exists.")
        if not (self.check_main_mem_addr(addr) or self.check_l1_mem_addr(addr)):
            raise Exception(f"[ERROR] Address {addr:#x} is out of bounds for circular buffer handle creation.")

        handle = CircularBufferHandle(buffer_id, addr, page_size, n_pages)
        self._buffer_handles[buffer_id] = handle
        return handle
    
    def remove_buffer_handle(self, buffer_id: str):
        if buffer_id not in self._buffer_handles:
            raise Exception(f"[ERROR] Buffer handle with ID '{buffer_id}' does not exist.")
        
        del self._buffer_handles[buffer_id]
        
    def remove_all_buffer_handles(self):
        self._buffer_handles.clear()
        
    def get_buffer_handle(self, buffer_id: str) -> BufferHandle | CircularBufferHandle:
        if buffer_id not in self._buffer_handles:
            raise Exception(f"[ERROR] Buffer handle with ID '{buffer_id}' does not exist.")
        
        return self._buffer_handles[buffer_id]
        
        
class InterconnectNetworkContext:
    def __init__(
        self,
        grid_shape: tuple[int, int] = (4, 4),
        flit_size: int = parse_mem_cap_str("32B"),
        control_packet_size: int = parse_mem_cap_str("64B"),
    ):
        super().__init__()
        
        self._grid_shape = tuple(grid_shape)
        self._flit_size  = flit_size
        self._control_packet_size = control_packet_size

        self._registered_nodes: dict[str, dict[int, tuple[tuple[int, int], Core]]] = {
            "NPUCore": {},
            "DMACore": {},
        }
        
        self._icnt_core = None
        
        self._router_locks: dict[tuple[int, int], TicketLock] = {
            (x, y): TicketLock() 
            for x in range(self._grid_shape[0]) 
            for y in range(self._grid_shape[1])
        }
        
    ################################################
    # Interconnection Network Core Management
    ################################################
    
    def register_core_as_global_control(self, core: 'IcntNetworkCore'):
        self._icnt_core = core
        
    @property
    def icnt_core(self) -> 'IcntNetworkCore':
        return self._icnt_core

    def register_core_as_node(self, core: Core, coord: tuple[int, int]):
        if coord[0] < 0 or coord[0] >= self._grid_shape[0]:
            raise Exception(f"[ERROR] X coordinate {coord[0]} is out of bounds for the interconnection network grid shape {self._grid_shape}.")
        if coord[1] < 0 or coord[1] >= self._grid_shape[1]:
            raise Exception(f"[ERROR] Y coordinate {coord[1]} is out of bounds for the interconnection network grid shape {self._grid_shape}.")
        
        if isinstance(core, NPUCore):
            core_type = "NPUCore"
            core_id = core._npu_core_id
        elif isinstance(core, DMACore):
            core_type = "DMACore"
            core_id = core._dma_core_id
        else:
            raise Exception(f"[ERROR] Core type {type(core).__name__} cannot be registered as a node for the interconnection network.")

        self._registered_nodes[core_type][core_id] = (coord, core)
        self._router_locks[coord] = TicketLock()
        
    def get_core_by_id(self, core_type: str | type, core_id: int) -> Core:
        if isinstance(core_type, type):
            core_type = core_type.__name__
            
        if core_type not in self._registered_nodes:
            raise Exception(f"[ERROR] Core type '{core_type}' is not registered in the interconnection network.")
        if core_id not in self._registered_nodes[core_type]:
            raise Exception(f"[ERROR] Core ID {core_id} not found in registered cores of type '{core_type}'.")
        
        return self._registered_nodes[core_type][core_id][1]
    
    def get_coord_by_id(self, core_type: str | type, core_id: int) -> tuple[int, int]:
        if isinstance(core_type, type):
            core_type = core_type.__name__
            
        if core_type not in self._registered_nodes:
            raise Exception(f"[ERROR] Core type '{core_type}' is not registered in the interconnection network.")
        if core_id not in self._registered_nodes[core_type]:
            raise Exception(f"[ERROR] Core ID {core_id} not found in registered cores of type '{core_type}'.")
        
        return self._registered_nodes[core_type][core_id][0]
    
    def get_router_lock(self, coord: tuple[int, int]) -> TicketLock:
        if coord[0] < 0 or coord[0] >= self._grid_shape[0]:
            raise Exception(f"[ERROR] X coordinate {coord[0]} is out of bounds for the interconnection network grid shape {self._grid_shape}.")
        if coord[1] < 0 or coord[1] >= self._grid_shape[1]:
            raise Exception(f"[ERROR] Y coordinate {coord[1]} is out of bounds for the interconnection network grid shape {self._grid_shape}.")
        
        return self._router_locks[coord]
    
    #################################################
    # Interconnection Network Latency
    #################################################
    
    def compute_hop_cnt(self, src_coord: tuple[int, int], dst_coord: tuple[int, int]) -> int:
        return abs(src_coord[0] - dst_coord[0]) + abs(src_coord[1] - dst_coord[1])
    
    def get_control_packet_latency(self, src_coord: tuple[int, int], dst_coord: tuple[int, int]) -> int:
        hop_cnt = self.compute_hop_cnt(src_coord, dst_coord)
        return hop_cnt + (self._control_packet_size // self._flit_size)
    
    def get_data_packet_latency(self, src_coord: tuple[int, int], dst_coord: tuple[int, int], data_size: int) -> int:
        hop_cnt = self.compute_hop_cnt(src_coord, dst_coord)
        return hop_cnt + (data_size // self._flit_size) + 1
    
    
class TensorProcessorContext:
    def __init__(self):
        pass
    
        
################################################
# Interconnect Network Core
################################################

class IcntNetworkCore(Core):
    def __init__(
        self,
        icnt_context: InterconnectNetworkContext,
    ):
        super().__init__(
            core_id=DEFAULT_CORE_ID, 
            cycle_model=IcntNetworkCoreCycleModel(self), 
            functional_model=IcntNetworkCoreFunctionalModel(self)
        )
        
        self.icnt_context = icnt_context
        self.icnt_context.register_core_as_global_control(self)
        
    @core_command_method
    def acquire_router_lock(self, coord: tuple[int, int]):
        pid = get_global_context()
        lock = self.icnt_context.get_router_lock(coord)

        if not lock.is_acquired_with(key=pid):
            lock.acquire(key=pid)
            
    @core_command_method
    def release_router_lock(self, coord: tuple[int, int]):
        pid = get_global_context()
        lock = self.icnt_context.get_router_lock(coord)
        
        if lock.is_locked_with(key=pid):
            lock.release(key=pid)
    
    @core_command_method
    def send_control_packet(self, src_coord: tuple[int, int], dst_coord: tuple[int, int]):
        pid = get_global_context()
        lock = self.icnt_context.get_router_lock(src_coord)
        
        if not lock.is_locked_with(key=pid):
            return False
        
    @core_command_method
    def send_data_packet(self, src_coord: tuple[int, int], dst_coord: tuple[int, int], data_size: int):
        pid = get_global_context()
        lock = self.icnt_context.get_router_lock(src_coord)
        
        if not lock.is_locked_with(key=pid):
            return False

    @core_kernel_method
    def request_data_transfer(self, consumer_coord: tuple[int, int], producer_coord: tuple[int, int], data_size: int):
        self.acquire_router_lock(consumer_coord)
        self.send_control_packet(consumer_coord, producer_coord)
        self.release_router_lock(consumer_coord)
        
        self.acquire_router_lock(producer_coord)
        self.send_data_packet(producer_coord, consumer_coord, data_size)
        self.release_router_lock(producer_coord)

class IcntNetworkCoreCycleModel(CoreCycleModel):
    def __init__(self, core: 'IcntNetworkCore'):
        super().__init__()
        
        self.core = core
        
    def send_control_packet(self, src_coord: tuple[int, int], dst_coord: tuple[int, int]):
        latency = self.core.icnt_context.get_control_packet_latency(src_coord, dst_coord)
        return latency
    
    def send_data_packet(self, src_coord: tuple[int, int], dst_coord: tuple[int, int], data_size: int):
        latency = self.core.icnt_context.get_data_packet_latency(src_coord, dst_coord, data_size)
        return latency
        
class IcntNetworkCoreFunctionalModel(CoreFunctionalModel):
    def __init__(self, core: 'IcntNetworkCore'):
        super().__init__()
        
        self.core = core
   

################################################
# DMA Core Implementation
################################################        

class DMACore(Core):
    def __init__(
        self,
        coord: tuple[int, int],
        mem_context: MemoryContext,
        icnt_context: InterconnectNetworkContext,
    ):
        super().__init__(
            core_id=DEFAULT_CORE_ID, 
            cycle_model=DMACoreCycleModel(self),
            functional_model=DMACoreFunctionalModel(self),
        )
        
        self.coord = coord
        self.mem_context = mem_context
        self.icnt_context = icnt_context
        
        # Initialization
        self._dma_core_id = self.mem_context.add_new_main_mem_segment()
        self.icnt_context.register_core_as_node(self, self.coord)
        
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

################################################
# NPU Core Implementation
################################################
    
class NPUCore(Core):
    def __init__(
        self,
        coord: tuple[int, int],
        mem_context: MemoryContext,
        icnt_context: InterconnectNetworkContext,
    ):
        super().__init__(
            core_id=DEFAULT_CORE_ID, 
            cycle_model=NPUCoreCycleModel(core=self),
            functional_model=NPUCoreFunctionalModel(core=self),
        )
        
        self.coord = coord
        self.mem_context = mem_context
        self.icnt_context = icnt_context
        # self.reg_context = RegisterContext(n_gp_regs=32, n_mxu_regs=4, n_vpu_regs=8)

        # Initialization
        self._npu_core_id = self.mem_context.add_new_l1_mem_segment()
        self.icnt_context.register_core_as_node(self, self.coord)
        
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
            src_coord = self.icnt_context.get_coord_by_id(NPUCore, src_core_id)
            src_core = self.icnt_context.get_core_by_id(NPUCore, src_core_id)
        elif self.mem_context.check_main_mem_addr(src_handle.addr):
            src_core_id, _, _ = self.mem_context.parse_main_mem_addr(src_handle.addr)
            src_coord = self.icnt_context.get_coord_by_id(DMACore, src_core_id)
            src_core = self.icnt_context.get_core_by_id(DMACore, src_core_id)
            
        if self.mem_context.check_l1_mem_addr(dst_handle.addr):
            dst_core_id, _, _ = self.mem_context.parse_l1_mem_addr(dst_handle.addr)
            dst_coord = self.icnt_context.get_coord_by_id(NPUCore, dst_core_id)
            dst_core = self.icnt_context.get_core_by_id(NPUCore, dst_core_id)
        elif self.mem_context.check_main_mem_addr(dst_handle.addr):
            dst_core_id, _, _ = self.mem_context.parse_main_mem_addr(dst_handle.addr)
            dst_coord = self.icnt_context.get_coord_by_id(DMACore, dst_core_id)
            dst_core = self.icnt_context.get_core_by_id(DMACore, dst_core_id)

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
            self.icnt_context = InterconnectNetworkContext(grid_shape=(4, 4))
            
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