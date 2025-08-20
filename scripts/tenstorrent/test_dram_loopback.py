import os
import time
import torch

from neuromta.framework import *
from neuromta.hardware import *
from neuromta.ip.tenstorrent.architecture import TenstorrentConfig, TenstorrentDevice


FILENAME = os.path.splitext(os.path.basename(__file__))[0]
TRACE_DIR = os.path.join(os.path.dirname(__file__), ".logs", FILENAME)


@core_kernel_method
def kernel_main(core: NPUCore, main_in_ptr: Pointer, l1_ptr: Pointer, main_out_ptr: Pointer, n_pages: int):
    for i in range(n_pages):
        core.async_noc_page_read(l1_ptr[0], main_in_ptr[i])
        core.async_rpc_barrier()
         
        core.async_noc_page_write(main_out_ptr[i], l1_ptr[0])
        core.async_rpc_barrier()


if __name__ == "__main__":
    config = TenstorrentConfig.BLACKHOLE()
    
    device = TenstorrentDevice(**config)
    device.initialize()
    device.change_sim_model_options(use_cycle_model=True, use_functional_model=True)
    
    page_size = 256
    n_pages = 4
    dtype = torch.int8
    
    npu_core = device.npu_cores[0]
    
    main_in_ptr     = device.create_sharded_main_memory_buffer("main_buffer", page_size, n_pages, coords=[device.dma_core_coords[0],])
    main_out_ptr    = device.create_sharded_main_memory_buffer("main_buffer", page_size, n_pages, coords=[device.dma_core_coords[0],])
    l1_ptr          = device.create_l1_buffer_to_cores("l1_buffer", page_size, 1, coords=[npu_core.coord,])


    for i in range(n_pages):
        content = torch.zeros(page_size // dtype.itemsize, dtype=dtype).fill_(i+1)
        device.dma_cores[0].mem_handle.set_content(main_in_ptr[i], content)
    
    kernel_main(npu_core, main_in_ptr, l1_ptr, main_out_ptr, n_pages=n_pages)
    
    st = time.time()
    device.run_kernels(verbose=True, max_steps=-1, save_trace=True, save_trace_dir=TRACE_DIR)
    ed = time.time()
    
    print(f"kernel simulation time: {(ed - st)*1000:.2f}ms")
    print(f"simulation terminated with {device.timestamp}")
    print("simulation ouput")
    print(device.dma_cores[0].mem_handle.get_content(main_in_ptr[0], shape=(-1, page_size // dtype.itemsize), dtype=dtype))
    print(device.dma_cores[0].mem_handle.get_content(main_out_ptr[0], shape=(-1, page_size // dtype.itemsize), dtype=dtype))
