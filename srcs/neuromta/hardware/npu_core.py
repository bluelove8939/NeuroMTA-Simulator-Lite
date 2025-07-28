import enum
import math

from neuromta.common.core import *
from neuromta.common.parser_utils import parse_mem_cap_str

from neuromta.hardware.npu_global_controller import NPUGlobalController
from neuromta.hardware.synchronizer import CircularBufferHandle


__all__ = [
    "MXUDataflow",
    "VPUOpMetadata",
    "VPUOp",
    "NPUCore",
]


class MXUDataflow(enum.Enum):
    OS = enum.auto()
    WS = enum.auto()
    

class VPUOpMetadata:
    class Type(enum.Enum):
        UNARY = enum.auto()
        ARITH = enum.auto()
    
    def __init__(self, op_name: str, op_type: Type):
        self.op_name = op_name
        self.op_type = op_type

class VPUOp:
    ADD     = VPUOpMetadata("add", VPUOpMetadata.Type.ARITH)
    SUB     = VPUOpMetadata("sub", VPUOpMetadata.Type.ARITH)
    MUL     = VPUOpMetadata("mul", VPUOpMetadata.Type.ARITH)
    DIV     = VPUOpMetadata("div", VPUOpMetadata.Type.ARITH)
    
    NEG     = VPUOpMetadata("neg", VPUOpMetadata.Type.UNARY)
    ABS     = VPUOpMetadata("abs", VPUOpMetadata.Type.UNARY)
    RELU    = VPUOpMetadata("relu", VPUOpMetadata.Type.UNARY)


class NPUCore(Core):
    def __init__(
        self,
        core_id: str,
        
        # Global context
        global_controller: NPUGlobalController,
        
        # MXU configuration
        mxu_m_tile: int = 32,
        mxu_n_tile: int = 32,
        mxu_k_tile: int = 32,
        mxu_dataflow: MXUDataflow = MXUDataflow.OS,
        
        # VPU configuration
        vpu_max_vlen: int = 256,
        vpu_unary_op_latency: int = 2,
        vpu_arith_op_latency: int = 4,
        
        # L1 memory configuration
        l1_capacity: int = parse_mem_cap_str("1MB"),
        l1_block_size: int = parse_mem_cap_str("32B"),
        l1_read_cycle_per_block: int = 1,
        l1_write_cycle_per_block: int = 4,
        
        # Circular buffer configuration
        local_cb_sem_access_latency: int = 1,
        local_cb_access_latency_per_entry: int = 1,
    ):
        super().__init__(core_id=core_id)
        
        self.mxu_m_tile = mxu_m_tile
        self.mxu_n_tile = mxu_n_tile
        self.mxu_k_tile = mxu_k_tile
        self.mxu_dataflow = mxu_dataflow
        
        if self.mxu_dataflow == MXUDataflow.OS:
            self.mxu_pe_arr_height = self.mxu_m_tile
            self.mxu_pe_arr_width  = self.mxu_n_tile
            self.mxu_tile_seq_len  = self.mxu_k_tile
        elif self.mxu_dataflow == MXUDataflow.WS:
            self.mxu_pe_arr_height = self.mxu_k_tile
            self.mxu_pe_arr_width  = self.mxu_n_tile
            self.mxu_tile_seq_len  = self.mxu_m_tile
            
        self.vpu_max_vlen = vpu_max_vlen
        self.vpu_unary_op_latency = vpu_unary_op_latency
        self.vpu_arith_op_latency = vpu_arith_op_latency
        
        self.l1_capacity = l1_capacity
        self.l1_block_size = l1_block_size
        self.l1_read_cycle_per_block = l1_read_cycle_per_block
        self.l1_write_cycle_per_block = l1_write_cycle_per_block
        self.l1_num_blocks = self.l1_capacity // self.l1_block_size
        
        self.local_cb_sem_access_latency = local_cb_sem_access_latency
        self.local_cb_access_latency_per_entry = local_cb_access_latency_per_entry
        
        self.cb_handles: dict[str, CircularBufferHandle] = {}
        self.global_controller = global_controller
    
    # MXU commands
    @core_command_method("mxu_pe_arr_width")
    def mxu_preload(self):
        pass
    
    @core_command_method("mxu_tile_seq_len")
    def mxu_execute(self):
        pass

    @core_command_method("mxu_pe_arr_width")
    def mxu_flush(self):
        pass
    
    # VPU commands
    def _vpu_op_cycles(self, op: VPUOpMetadata, vlen: int) -> int:
        if op.op_type == VPUOpMetadata.Type.UNARY:
            return self.vpu_unary_op_latency * math.ceil(vlen / self.vpu_max_vlen)
        elif op.op_type == VPUOpMetadata.Type.ARITH:
            return self.vpu_arith_op_latency * math.ceil(vlen / self.vpu_max_vlen)
        else:
            raise ValueError(f"Unsupported VPU operation type: {op.op_type}")
    
    @core_command_method("_vpu_op_cycles")
    def vpu_op(self, op: VPUOpMetadata, vlen: int):
        pass
    
    # L1 memory access commands
    def _l1_read_cycles(self, addr: int, size: int) -> int:
        return self.l1_read_cycle_per_block * math.ceil(size / self.l1_block_size)

    def _l1_write_cycles(self, addr: int, size: int) -> int:
        return self.l1_write_cycle_per_block * math.ceil(size / self.l1_block_size)
    
    @core_command_method("_l1_read_cycles")
    def l1_read(self, addr: int, size):
        if addr < 0 or addr >= self.l1_capacity:
            raise ValueError(f"Address {addr} is out of L1 memory bounds (0-{self.l1_capacity-1})")

    @core_command_method("_l1_write_cycles")
    def l1_write(self, addr: int, size):
        if addr < 0 or addr >= self.l1_capacity:
            raise ValueError(f"Address {addr} is out of L1 memory bounds (0-{self.l1_capacity-1})")
        
    # Intra-core communication commands
    def local_cb_access_latency(self, name: str, entry_num: int):
        return self.local_cb_access_latency_per_entry * entry_num
    
    def local_cb_create_buffer_handle(self, name: str, entry_num: int):
        if name in self.cb_handles:
            raise Exception(f"[ERROR] Circular buffer with name '{name}' already exists.")
        if entry_num <= 0:
            raise Exception("[ERROR] Entry number must be greater than 0.")
        
        self.cb_handles[name] = CircularBufferHandle(entry_num)

    def local_cb_remove_buffer_handle(self, name: str):
        if name not in self.cb_handles:
            raise Exception(f"[ERROR] Circular buffer with name '{name}' does not exist.")
        
        del self.cb_handles[name]
        
    @core_command_method("local_cb_sem_access_latency")
    def local_cb_reserve_back(self, name: str, entry_num: int):
        if name not in self.cb_handles:
            raise Exception(f"[ERROR] Circular buffer with name '{name}' does not exist.")
        
        handle = self.cb_handles[name]
        
        if not handle.check_vacancy(entry_num): return False
        if handle.is_locked: return False
        
        handle.lock(entry_num)
        
    @core_command_method("local_cb_access_latency")
    def local_cb_push_back(self, name: str, entry_num: int):
        if name not in self.cb_handles:
            raise Exception(f"[ERROR] Circular buffer with name '{name}' does not exist.")
        
        handle = self.cb_handles[name]
        handle.push_back(entry_num)
        handle.unlock(entry_num)
        
    @core_command_method("local_cb_sem_access_latency")
    def local_cb_wait_front(self, name: str, entry_num: int):
        if name not in self.cb_handles:
            raise Exception(f"[ERROR] Circular buffer with name '{name}' does not exist.")
        
        handle = self.cb_handles[name]
        
        if not handle.check_occupancy(entry_num): return False
        if handle.is_locked: return False
        
        handle.lock(entry_num)
        
    @core_command_method("local_cb_access_latency")
    def local_cb_pop_front(self, name: str, entry_num: int=1):
        if name not in self.cb_handles:
            raise Exception(f"[ERROR] Circular buffer with name '{name}' does not exist.")
        
        handle = self.cb_handles[name]        
        handle.pop_front(entry_num)
        handle.unlock(entry_num)


# if __name__ == "__main__":
#     global_controller = NPUGlobalController(core_id="global_controller")
    
#     core = NPUCore(
#         core_id="npu_core",

#         global_controller=global_controller,

#         mxu_m_tile=32,
#         mxu_n_tile=32,
#         mxu_k_tile=256,
#         mxu_dataflow=MXUDataflow.OS,
        
#         l1_capacity=parse_mem_cap_str("1MB"),
#         l1_block_size=parse_mem_cap_str("16B"),
#         l1_read_cycle_per_block=1,
#         l1_write_cycle_per_block=4,
        
#         global_controller=global_controller
#     )
    
#     core.global_controller.cb_create_buffer_handle("buffer", 16)
    
#     @core_kernel_method
#     def reader_kernel(core: NPUCore):
#         core.global_controller.cb_reserve_back("buffer", 4)
#         core.global_controller.cb_push_back("buffer", 4)
        
#     @core_kernel_method
#     def writer_kernel(core: NPUCore):
#         core.global_controller.cb_wait_front("buffer", 4)
#         core.global_controller.cb_pop_front("buffer", 4)

#     core.dispatch_kernel(kernel=reader_kernel(core))
#     core.dispatch_kernel(kernel=writer_kernel(core))

#     total_cycles = 0
    
#     def debug_hook(cmd: Command):
#         global total_cycles
#         print(f"[DEBUG] #{total_cycles:<5d} | {cmd}")
        
#     core.register_command_debug_hook(debug_hook)
#     global_controller.register_command_debug_hook(debug_hook)
    
#     while not core.is_idle:
#         cycles = core.get_remaining_cycles()
#         core.update_cycle_time(cycles)
        
#         total_cycles += cycles
        
#     core.global_controller.cb_remove_buffer_handle("buffer")
        
#     print(f"Total cycles executed: {total_cycles}")
        

if __name__ == "__main__":
    global_controller = NPUGlobalController(core_id="global_controller")
    
    core = NPUCore(
        core_id="npu_core",
        
        global_controller=global_controller,
        
        mxu_m_tile=32,
        mxu_n_tile=32,
        mxu_k_tile=256,
        mxu_dataflow=MXUDataflow.OS,

        vpu_max_vlen=256,
        vpu_unary_op_latency=2,
        vpu_arith_op_latency=4,

        l1_capacity=parse_mem_cap_str("1MB"),
        l1_block_size=parse_mem_cap_str("16B"),
        l1_read_cycle_per_block=1,
        l1_write_cycle_per_block=4,
    )
    
    core.local_cb_create_buffer_handle("buffer", 16)
    core.local_cb_create_buffer_handle("result", 4)
    
    @core_kernel_method
    def read_kernel(core: NPUCore):
        core.local_cb_reserve_back("buffer", 4)
        
        for _ in range(4):
            core.l1_read(0, core.l1_block_size)
            core.local_cb_push_back("buffer", 1)
        
    @core_kernel_method
    def compute_kernel(core: NPUCore):
        core.local_cb_wait_front("buffer", 4)
        for _ in range(4):
            core.local_cb_pop_front("buffer", 1)
        
        core.mxu_preload()
        for _ in range(4):
            core.mxu_execute()
        core.mxu_flush()
        
        core.vpu_op(VPUOp.RELU, 1024)
        
        core.local_cb_reserve_back("result", 1)
        core.local_cb_push_back("result", 1)
        
    @core_kernel_method
    def write_kernel(core: NPUCore):
        core.local_cb_wait_front("result", 1)
        core.local_cb_pop_front("result", 1)        
        core.l1_write(0, core.l1_block_size)

    core.dispatch_kernel(kernel=read_kernel(core))
    core.dispatch_kernel(kernel=compute_kernel(core))
    core.dispatch_kernel(kernel=write_kernel(core))
    
    total_cycles = 0
    
    def debug_hook(cmd: Command):
        global total_cycles
        print(f"[DEBUG] #{total_cycles:<5d} | {cmd}")
        
    core.register_command_debug_hook(debug_hook)
    global_controller.register_command_debug_hook(debug_hook)
    
    while not core.is_idle:
        cycles = core.get_remaining_cycles()
        core.update_cycle_time(cycles)
        
        total_cycles += cycles
        
    core.local_cb_remove_buffer_handle("buffer")
    core.local_cb_remove_buffer_handle("result")
        
    print(f"Total cycles executed: {total_cycles}")
