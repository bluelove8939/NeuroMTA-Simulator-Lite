from neuromta.common.device import Device
from neuromta.common.parser_utils import parse_mem_cap_str

from neuromta.hardware.npu_core import NPUCore, NPUGlobalController, MXUDataflow


__all__ = [
    "NPUCoreConfig",
    "NPUDevice",
]


class NPUCoreConfig(dict):
    def __init__(self, mxu_m_tile=32, mxu_n_tile=32, mxu_k_tile=32, mxu_dataflow=MXUDataflow.OS,
                 l1_capacity=parse_mem_cap_str("1MB"), l1_block_size=parse_mem_cap_str("32B"),
                 l1_read_cycle_per_block=1, l1_write_cycle_per_block=4):
        super().__init__({
            "mxu_m_tile": mxu_m_tile,
            "mxu_n_tile": mxu_n_tile,
            "mxu_k_tile": mxu_k_tile,
            "mxu_dataflow": mxu_dataflow,
            "l1_capacity": l1_capacity,
            "l1_block_size": l1_block_size,
            "l1_read_cycle_per_block": l1_read_cycle_per_block,
            "l1_write_cycle_per_block": l1_write_cycle_per_block
        })


class NPUDevice(Device):
    def __init__(
        self,
        npu_core_config: NPUCoreConfig = NPUCoreConfig(),
        npu_core_grid_x: int = 2,
        npu_core_grid_y: int = 2,
    ):
        super().__init__()
        
        self.npu_core_config = npu_core_config
        self.npu_core_grid_x = npu_core_grid_x
        self.npu_core_grid_y = npu_core_grid_y
        
        self.npu_global_controller = NPUGlobalController()
        
        self.npu_cores: list[list[NPUCore]] = [[NPUCore(**self.npu_core_config, global_controller=self.npu_global_controller) 
            for _ in range(self.npu_core_grid_x)] 
            for _ in range(self.npu_core_grid_y)]
        

if __name__ == "__main__":
    from neuromta.common.core import core_kernel_method
    
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
        
        core.local_cb_reserve_back("result", 1)
        core.local_cb_push_back("result", 1)
        
    @core_kernel_method
    def write_kernel(core: NPUCore):
        core.local_cb_wait_front("result", 1)
        core.local_cb_pop_front("result", 1)        
        core.l1_write(0, core.l1_block_size)
    
    
    device = NPUDevice()
    device.initialize()
    
    for core_row in device.npu_cores:
        for core in core_row:
            core.local_cb_create_buffer_handle("buffer", 16)
            core.local_cb_create_buffer_handle("result", 4)
        
            core.dispatch_kernel(kernel=read_kernel(core))
            core.dispatch_kernel(kernel=compute_kernel(core))
            core.dispatch_kernel(kernel=write_kernel(core))

    device.run_kernels()
    
    for core_row in device.npu_cores:
        for core in core_row:
            core.local_cb_remove_buffer_handle("buffer")
            core.local_cb_remove_buffer_handle("result")

    print("NPU Device simulation completed.")