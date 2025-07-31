import math

from neuromta.common.core import *
from neuromta.common.parser_utils import parse_mem_cap_str

from neuromta.ip.hardware.common.context import Context
from neuromta.ip.hardware.common.custom_types import MemoryType
from neuromta.ip.hardware.common.buffer_handle import BufferHandle, CircularBufferHandle
from neuromta.ip.hardware.common.main_memory_config import MainMemoryConfig, HBM2Config


__all__ = ["MemoryContext", "MemoryOwnerCore"]


class MemoryOwnerCore(Core):
    def __init__(self, mem_seg_type: MemoryType, mem_context: 'MemoryContext', **kwargs):
        super().__init__(**kwargs)
        
        self.mem_seg_type = mem_seg_type
        self.mem_context = mem_context
        self.mem_seg_id = self.mem_context.add_new_mem_segment(self)


class MemoryContext(Context):
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
        super().__init__(n_owners_per_res=-1)
        
        self._l1_mem_base           = l1_mem_base
        self._l1_mem_bank_num       = l1_mem_bank_num
        self._l1_mem_bank_size      = l1_mem_bank_size
        self._l1_mem_access_gran    = l1_mem_access_gran
        
        self._main_mem_base         = main_mem_base
        self._main_mem_bank_num     = main_mem_bank_num
        self._main_mem_bank_size    = main_mem_bank_size
        self._main_mem_config       = main_mem_config

        self._l1_mem_rd_latency_per_block   = l1_mem_rd_latency_per_block
        self._l1_mem_wr_latency_per_block   = l1_mem_wr_latency_per_block
        
        self._buffer_handles: dict[str, BufferHandle | CircularBufferHandle] = {}
        
    ###############################################
    # Memory Owner Management
    ###############################################
    
    def add_new_mem_segment(self, core: MemoryOwnerCore) -> int:
        if core.mem_seg_type == MemoryType.L1:
            return self.register_owner(MemoryType.L1, core)
        elif core.mem_seg_type == MemoryType.MAIN:
            return self.register_owner(MemoryType.MAIN, core)
        else:
            raise Exception(f"[ERROR] Unsupported memory segment type: {core.mem_seg_type}.")
    
    @property
    def _l1_mem_seg_num(self) -> int:
        return len(self._registered_owners[MemoryType.L1])
    
    @property
    def _main_mem_seg_num(self) -> int:
        return len(self._registered_owners[MemoryType.MAIN])
        
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