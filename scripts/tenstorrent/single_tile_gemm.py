import os

from neuromta.common import *
from neuromta.hardware import *
from neuromta.ip.tenstorrent.architecture import TenstorrentConfig, TenstorrentDevice


@core_kernel_method
def read_kernel(
    core:           NPUCore, 
    
    ifm_ptr:        Pointer, 
    wgt_ptr:        Pointer,
    psum_ptr:       Pointer,
    
    ifm_cb_ptr:     Pointer,
    wgt_cb_ptr:     Pointer,
    psum_cb_ptr:    Pointer,

    n_pages_input:  int,
    n_pages_psum:   int,
):
    pass


if __name__ == "__main__":
    config = TenstorrentConfig.BLACKHOLE()
    device = TenstorrentDevice(**config)
    
    device.initialize(create_trace=True)
    device.change_sim_model_options(use_cycle_model=True, use_functional_model=True)
    

    # Run the device to execute the kernels
    device.run_kernels()
    

    print("NPU Device simulation completed.")
    
    # Save traces to a file
    trace_dirname   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "traces")
    trace_filename  = os.path.splitext(os.path.basename(__file__))[0] + "_traces.csv"
    trace_filepath  = os.path.join(trace_dirname, trace_filename)

    os.makedirs(trace_dirname, exist_ok=True)
    device.save_traces(trace_filepath)
    
    print(f"Traces saved to {trace_filepath}")