import os
import torch

from neuromta.framework import *
from neuromta.hardware import *
from neuromta.ip.tenstorrent.architecture import TenstorrentConfig, TenstorrentDevice


FILENAME = os.path.splitext(os.path.basename(__file__))[0]
TRACE_DIR = os.path.join(os.path.dirname(__file__), ".logs", FILENAME)


if __name__ == "__main__":
    config = TenstorrentConfig.BLACKHOLE()
    
    device = TenstorrentDevice(**config)
    device.initialize()
    device.change_sim_model_options(use_cycle_model=True, use_functional_model=True)
    
    M = 32
    N = 32
    K = 32
    dtype = torch.int32
    acc_dtype = torch.int32
    
    ifm  = torch.arange(0, M * K, dtype=dtype).reshape(M, K)
    wgt  = torch.arange(0, K * N, dtype=dtype).reshape(K, N)
    psum = torch.ones((M, N), dtype=acc_dtype)
    ofm  = torch.zeros((M, N), dtype=acc_dtype)
    
    ifm_size = ifm.numel() * ifm.element_size()
    wgt_size = wgt.numel() * wgt.element_size()
    psum_size = psum.numel() * psum.element_size()
    ofm_size = ofm.numel() * ofm.element_size()
    
    ifm_ptrs  = device.create_l1_buffer_to_cores("ifm",  page_size=ifm_size,  n_pages=1)
    wgt_ptrs  = device.create_l1_buffer_to_cores("wgt",  page_size=wgt_size,  n_pages=1)
    psum_ptrs = device.create_l1_buffer_to_cores("psum", page_size=psum_size, n_pages=1)
    ofm_ptrs  = device.create_l1_buffer_to_cores("ofm",  page_size=ofm_size,  n_pages=1)
    
    device.npu_cores[0].mem_handle.set_content(ifm_ptrs[0], ifm)
    device.npu_cores[0].mem_handle.set_content(wgt_ptrs[0], wgt)
    device.npu_cores[0].mem_handle.set_content(psum_ptrs[0], psum)
    device.npu_cores[1].mem_handle.set_content(ofm_ptrs[1], ofm)

    @core_kernel_method
    def gemm_kernel(
        core: NPUCore, 
        
        src_ifm_ptr: Pointer, 
        src_wgt_ptr: Pointer, 
        src_psum_ptr: Pointer,     
        
        tmp_ifm_ptr: Pointer,
        tmp_wgt_ptr: Pointer,
        tmp_psum_ptr: Pointer,
        tmp_ofm_ptr: Pointer
    ):  
        core.async_noc_buffer_read(tmp_ifm_ptr, 0, src_ifm_ptr, 0, 1)
        core.async_noc_buffer_read(tmp_wgt_ptr, 0, src_wgt_ptr, 0, 1)
        core.async_noc_buffer_read(tmp_psum_ptr, 0, src_psum_ptr, 0, 1)
        core.async_rpc_barrier()

        core.mxu_reconfigure(dtype=torch.int32, acc_dtype=torch.int32)
        core.mxu_tiled_gemm(
            ifm_ptr=tmp_ifm_ptr,
            wgt_ptr=tmp_wgt_ptr,
            psum_ptr=tmp_psum_ptr,
            ofm_ptr=tmp_ofm_ptr,
            preload_wgt=False,
            preload_psum=True,
            flush_ofm=True
        )
        
        core.vpu_reconfigure(vlen=32, vdtype=dtype)
        core.vpu_load_reg(tmp_ifm_ptr, 0, 0, 4)
        core.vpu_load_reg(tmp_wgt_ptr, 0, 4, 4)
        core.vpu_execute(VPUOperator.ADD, 0, 4, 8, inplace=False, burst_len=4)
        core.vpu_store_reg(tmp_ofm_ptr, 0, 8, 4)
        
    gemm_kernel(
        device.npu_cores[1],
        
        src_ifm_ptr=ifm_ptrs[0],
        src_wgt_ptr=wgt_ptrs[0],
        src_psum_ptr=psum_ptrs[0],
        
        tmp_ifm_ptr=ifm_ptrs[1],
        tmp_wgt_ptr=wgt_ptrs[1],
        tmp_psum_ptr=psum_ptrs[1],
        tmp_ofm_ptr=ofm_ptrs[1]
    )

    device.run_kernels(verbose=True, max_steps=-1, save_trace=True, save_trace_dir=TRACE_DIR)
    
    reference = torch.matmul(ifm, wgt) + psum
    reference[0:4, :] = ifm[0:4, :] + wgt[0:4, :]  # Simulate the effect of the VPU operation
    simulated = device.npu_cores[1].mem_handle.get_content(ofm_ptrs[1], shape=(M, N), dtype=acc_dtype)

    print(f"\n=== REFERENCE ===\n{reference}")
    print(f"\n=== SIMULATED ===\n{simulated}")
    print(f"\nsimulation terminated with valid result: {torch.allclose(reference, simulated)}")