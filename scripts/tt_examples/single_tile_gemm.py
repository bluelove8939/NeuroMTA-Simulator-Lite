from neuromta.common.core import *
from neuromta.common.parser_utils import parse_mem_cap_str

from neuromta.hardware.npu_core import *
from neuromta.hardware.npu_device import *


@core_kernel_method
def read_kernel(core: NPUCore):
    core.local_cb_reserve_back("ifm_buffer", 2)
    core.local_cb_reserve_back("wgt_buffer", 2)
    core.local_cb_reserve_back("psum_buffer", 1)
    
    for _ in range(2):
        core.l1_read(0, core.l1_block_size)
        core.local_cb_push_back("ifm_buffer", 1)
        
    for _ in range(2):
        core.l1_read(0, core.l1_block_size)
        core.local_cb_push_back("wgt_buffer", 1)
        
    core.l1_read(0, core.l1_block_size)
    core.local_cb_push_back("psum_buffer", 1)
    
@core_kernel_method
def compute_kernel(core: NPUCore):
    core.local_cb_wait_front("ifm_buffer", 2)
    core.local_cb_wait_front("wgt_buffer", 2)
    core.local_cb_wait_front("psum_buffer", 1)
    
    for _ in range(2):
        core.local_cb_pop_front("ifm_buffer", 1)
    for _ in range(2):
        core.local_cb_pop_front("wgt_buffer", 1)
    core.local_cb_pop_front("psum_buffer", 1)
    
    core.mxu_preload()
    for _ in range(2):
        core.mxu_execute()
    core.mxu_flush()
    
    core.local_cb_reserve_back("ofm_buffer", 1)
    core.local_cb_push_back("ofm_buffer", 1)
    
@core_kernel_method
def write_kernel(core: NPUCore):
    core.local_cb_wait_front("ofm_buffer", 1)
    core.local_cb_pop_front("ofm_buffer", 1)
    core.l1_write(0, core.l1_block_size)


if __name__ == "__main__":
    npu_core_config = NPUCoreConfig(
        mxu_m_tile=32,
        mxu_n_tile=32,
        mxu_k_tile=32,
        mxu_dataflow=MXUDataflow.OS,
        
        l1_capacity=parse_mem_cap_str("1MB"),
        l1_block_size=parse_mem_cap_str("32B"),
        l1_read_cycle_per_block=1,
        l1_write_cycle_per_block=4
    )
    
    device = NPUDevice(
        npu_core_config=npu_core_config,
        npu_core_grid_x=2,
        npu_core_grid_y=2
    )
    
    device.initialize()
    
    # Create circular buffers for each core
    for core_row in device.npu_cores:
        for core in core_row:
            core.local_cb_create_buffer_handle("ifm_buffer", 16)
            core.local_cb_create_buffer_handle("wgt_buffer", 16)
            core.local_cb_create_buffer_handle("psum_buffer", 16)
            core.local_cb_create_buffer_handle("ofm_buffer", 4)
    
    # Dispatch kernels to each core
    for core_row in device.npu_cores:
        for core in core_row:
            core.dispatch_kernel(kernel=read_kernel(core))
            core.dispatch_kernel(kernel=compute_kernel(core))
            core.dispatch_kernel(kernel=write_kernel(core))

    # Run the device to execute the kernels
    device.run_kernels()
    
    # Clean up circular buffers
    for core_row in device.npu_cores:
        for core in core_row:
            core.local_cb_remove_buffer_handle("ifm_buffer")
            core.local_cb_remove_buffer_handle("wgt_buffer")
            core.local_cb_remove_buffer_handle("psum_buffer")
            core.local_cb_remove_buffer_handle("ofm_buffer")

    print("NPU Device simulation completed.")