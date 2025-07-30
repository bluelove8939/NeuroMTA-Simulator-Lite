import os

from neuromta.common.core import *
from neuromta.common.parser_utils import parse_mem_cap_str

from neuromta.hardware.npu_core import *
from neuromta.hardware.npu_device import *


@core_kernel_method
def gemm1_read_kernel(core: NPUCore):
    core.local_cb_reserve_back("ifm_buffer", 1)
    core.l1_read(0, core.l1_block_size)
    core.local_cb_push_back("ifm_buffer", 1)
    
    core.local_cb_reserve_back("wgt_buffer", 1)
    core.l1_read(0, core.l1_block_size)
    core.local_cb_push_back("wgt_buffer", 1)
    
    core.local_cb_reserve_back("psum_buffer", 1)
    core.l1_read(0, core.l1_block_size)
    core.local_cb_push_back("psum_buffer", 1)
    
@core_kernel_method
def gemm1_compute_kernel(core: NPUCore):
    core.local_cb_wait_front("ifm_buffer", 1)
    core.local_cb_pop_front("ifm_buffer", 1)
    
    core.local_cb_wait_front("wgt_buffer", 1)
    core.local_cb_pop_front("wgt_buffer", 1)
    
    core.local_cb_wait_front("psum_buffer", 1)
    core.local_cb_pop_front("psum_buffer", 1)
    
    core.mxu_preload()
    core.mxu_execute()
    core.mxu_flush()
    
    core._global_controller.acquire_global_lock()
    core._global_controller.cb_reserve_back("gemm2_ifm_buffer", 1)
    core._global_controller.cb_push_back("gemm2_ifm_buffer", 1)
    core._global_controller.release_global_lock()
    
@core_kernel_method
def gemm1_write_kernel(core: NPUCore):
    pass
    
    
@core_kernel_method
def gemm2_read_kernel(core: NPUCore):
    core.local_cb_reserve_back("wgt_buffer", 1)
    core.local_cb_reserve_back("psum_buffer", 1)
    
    core.l1_read(0, core.l1_block_size)
    core.local_cb_push_back("wgt_buffer", 1)
        
    core.l1_read(0, core.l1_block_size)
    core.local_cb_push_back("psum_buffer", 1)
    
@core_kernel_method
def gemm2_compute_kernel(core: NPUCore):
    core._global_controller.acquire_global_lock()
    core._global_controller.cb_wait_front("gemm2_ifm_buffer", 1)
    core._global_controller.cb_pop_front("gemm2_ifm_buffer", 1)
    core._global_controller.release_global_lock()
    
    core.local_cb_wait_front("wgt_buffer", 1)
    core.local_cb_pop_front("wgt_buffer", 1)
    
    core.local_cb_wait_front("psum_buffer", 1)
    core.local_cb_pop_front("psum_buffer", 1)
    
    core.mxu_preload()
    core.mxu_execute()
    core.mxu_flush()
    
    core.local_cb_reserve_back("ofm_buffer", 1)
    core.local_cb_push_back("ofm_buffer", 1)
    
@core_kernel_method
def gemm2_write_kernel(core: NPUCore):
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
        npu_core_grid_y=1
    )
    
    device.initialize(create_trace=True)
    
    # Create circular buffers for each core
    for core_row in device.npu_cores:
        for core in core_row:
            core.local_cb_create_buffer_handle("ifm_buffer", 16)
            core.local_cb_create_buffer_handle("wgt_buffer", 16)
            core.local_cb_create_buffer_handle("psum_buffer", 16)
            core.local_cb_create_buffer_handle("ofm_buffer", 4)
    device.npu_global_controller.cb_create_buffer_handle("gemm2_ifm_buffer", 16)
    
    # Dispatch kernels to each core
    for i in range(1):
        for j in range(1):
            core = device.npu_cores[i][j]
            core.dispatch_kernel(kernel=gemm1_read_kernel(core))
            core.dispatch_kernel(kernel=gemm1_compute_kernel(core))
            core.dispatch_kernel(kernel=gemm1_write_kernel(core))
            
        for j in range(1, 2):
            core = device.npu_cores[i][j]
            core.dispatch_kernel(kernel=gemm2_read_kernel(core))
            core.dispatch_kernel(kernel=gemm2_compute_kernel(core))
            core.dispatch_kernel(kernel=gemm2_write_kernel(core))

    # Run the device to execute the kernels
    device.run_kernels()
    
    # Clean up circular buffers
    for core_row in device.npu_cores:
        for core in core_row:
            core.local_cb_remove_buffer_handle("ifm_buffer")
            core.local_cb_remove_buffer_handle("wgt_buffer")
            core.local_cb_remove_buffer_handle("psum_buffer")
            core.local_cb_remove_buffer_handle("ofm_buffer")
    device.npu_global_controller.cb_remove_buffer_handle("gemm2_ifm_buffer")

    print("NPU Device simulation completed.")
    
    # Save traces to a file
    trace_dirname   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "traces")
    trace_filename  = os.path.splitext(os.path.basename(__file__))[0] + "_traces.csv"
    trace_filepath  = os.path.join(trace_dirname, trace_filename)

    os.makedirs(trace_dirname, exist_ok=True)
    device.save_traces(trace_filepath)
    
    print(f"Traces saved to {trace_filepath}")