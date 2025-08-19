import enum
import torch

from neuromta.framework import *
from neuromta.hardware import *


__all__ = [
    "TenstorrentConfig",
    "TenstorrentDevice",
]


class TenstorrentConfig(dict):
    def __init__(
        self,
        
        icnt_config: IcntConfig, 
        mem_config: MemConfig,
        mxu_config: MXUConfig,
        vpu_config: VPUConfig, 
    ):
        self["icnt_config"] = icnt_config
        self["mem_config"] = mem_config
        self["mxu_config"] = mxu_config
        self["vpu_config"] = vpu_config
        
    @classmethod
    def BLACKHOLE(cls) -> 'TenstorrentConfig':
        core_map = IcntCoreMap.from_shape((12, 16))
        core_map.grid[ :, 0   ] = IcntCoreType.DMA
        core_map.grid[ :, 8   ] = IcntCoreType.DMA
        core_map.grid[2:, 1:8 ] = IcntCoreType.NPU
        core_map.grid[2:, 9:16] = IcntCoreType.NPU
        
        icnt_config = IcntConfig(
            core_map=core_map,
            l1_mem_bank_size=parse_mem_cap_str("1.5MB"),
            main_mem_bank_size=parse_mem_cap_str("1.2GB"),
            flit_size=parse_mem_cap_str("16B"),
            control_packet_size=parse_mem_cap_str("32B"),
        )
        
        l1_mem_config = L1MemoryConfig(
            access_gran=parse_mem_cap_str("512B"),
        )
        
        main_mem_config = MainMemoryConfig(
            transfer_speed=9600,      # MT/s (DDR6 typical speed)
            ch_io_width=128,          # bits (DDR6 typical channel width)
            ch_num=2,                 # channels (example for DDR6)
            burst_len=256,            # bytes (typical burst length)
            is_ddr=True,
            processor_clock_freq=parse_freq_str("1.35GHz"),
        )
        
        mem_config = MemConfig(
            l1_config=l1_mem_config,
            main_config=main_mem_config,
        )
        
        mxu_config = MXUConfig(
            pe_arr_height=32,
            pe_arr_width=32,
            seq_len=32,
            dtype=torch.float32,
            acc_dtype=torch.float32,
            dataflow=MXUDataflow.OS,
            op_latency_per_byte=1,
        )
        
        vpu_config = VPUConfig(
            vreg_len=parse_mem_cap_str("128B"),
            vreg_num=32,
            vdtype=torch.float32,
            
            vlen_max=1024,
            vlen_min=32,
            
            unary_op_latency=1,
            arith_op_latency=2,
        )
        
        return cls(
            icnt_config=icnt_config,
            mem_config=mem_config,
            mxu_config=mxu_config,
            vpu_config=vpu_config,
        )
        
    @classmethod
    def WORMHOLE(cls) -> 'TenstorrentConfig':
        core_map = IcntCoreMap.from_shape((10, 10))
        core_map.grid[:, 0   ] = IcntCoreType.DMA
        core_map.grid[:, 5   ] = IcntCoreType.DMA
        core_map.grid[:, 1:5 ] = IcntCoreType.NPU
        core_map.grid[:, 6:10] = IcntCoreType.NPU
        
        icnt_config = IcntConfig(
            core_map=core_map,
            l1_mem_bank_size=parse_mem_cap_str("1MB"),
            main_mem_bank_size=parse_mem_cap_str("1.2GB"),
            flit_size=parse_mem_cap_str("16B"),
            control_packet_size=parse_mem_cap_str("32B"),
        )
        
        l1_mem_config = L1MemoryConfig(
            access_gran=parse_mem_cap_str("512B"),
        )
        
        main_mem_config = MainMemoryConfig(
            transfer_speed=9600,      # MT/s (DDR6 typical speed)
            ch_io_width=128,          # bits (DDR6 typical channel width)
            ch_num=2,                 # channels (example for DDR6)
            burst_len=256,            # bytes (typical burst length)
            is_ddr=True,
            processor_clock_freq=parse_freq_str("1GHz"),
        )
        
        mem_config = MemConfig(
            l1_config=l1_mem_config,
            main_config=main_mem_config,
        )
        
        mxu_config = MXUConfig(
            pe_arr_height=32,
            pe_arr_width=32,
            seq_len=32,
            dtype=torch.float32,
            acc_dtype=torch.float32,
            dataflow=MXUDataflow.OS,
            op_latency_per_byte=1,
        )
        
        vpu_config = VPUConfig(
            vreg_len=parse_mem_cap_str("128B"),
            vreg_num=32,
            vdtype=torch.float32,
            
            vlen_max=1024,
            vlen_min=32,
            
            unary_op_latency=1,
            arith_op_latency=2,
        )
        
        return cls(
            icnt_config=icnt_config,
            mem_config=mem_config,
            mxu_config=mxu_config,
            vpu_config=vpu_config,
        )
    
    @classmethod
    def EAGLE_N1(cls) -> 'TenstorrentConfig':
        core_map = IcntCoreMap.from_shape((6, 5))
        core_map.grid[:, 0 ] = IcntCoreType.DMA
        core_map.grid[:, 1:] = IcntCoreType.NPU
        
        icnt_config = IcntConfig(
            core_map=core_map,
            l1_mem_bank_size=parse_mem_cap_str("1500KB"),
            main_mem_bank_size=parse_mem_cap_str("11GB"),
            flit_size=parse_mem_cap_str("16B"),
            control_packet_size=parse_mem_cap_str("32B"),
        )
        
        l1_mem_config = L1MemoryConfig(
            access_gran=parse_mem_cap_str("512B"),
        )
        
        main_mem_config = MainMemoryConfig(
            transfer_speed=9600,      # MT/s (DDR6 typical speed)
            ch_io_width=128,          # bits (DDR6 typical channel width)
            ch_num=2,                 # channels (example for DDR6)
            burst_len=256,            # bytes (typical burst length)
            is_ddr=True,
            processor_clock_freq=parse_freq_str("500MHz"),
        )
        
        mem_config = MemConfig(
            l1_config=l1_mem_config,
            main_config=main_mem_config,
        )
        
        mxu_config = MXUConfig(
            pe_arr_height=32,
            pe_arr_width=32,
            seq_len=32,
            dtype=torch.float32,
            acc_dtype=torch.float32,
            dataflow=MXUDataflow.OS,
            op_latency_per_byte=1,
        )
        
        vpu_config = VPUConfig(
            vreg_len=parse_mem_cap_str("128B"),
            vreg_num=32,
            vdtype=torch.float32,
            
            vlen_max=1024,
            vlen_min=32,
            
            unary_op_latency=1,
            arith_op_latency=2,
        )
        
        return cls(
            icnt_config=icnt_config,
            mem_config=mem_config,
            mxu_config=mxu_config,
            vpu_config=vpu_config,
        )
        

class TenstorrentDevice(MTAccelerator):
    def __init__(self, icnt_config, mem_config, mxu_config, vpu_config):
        super().__init__(icnt_config, mem_config, mxu_config, vpu_config)
        
        
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

    ifm_ptrs[0].handle.set_content(ifm)
    wgt_ptrs[0].handle.set_content(wgt)
    psum_ptrs[0].handle.set_content(psum)
    ofm_ptrs[1].handle.set_content(ofm)

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
        core.remote_memcopy_buffer(tmp_ifm_ptr, 0, src_ifm_ptr, 0, 1)
        core.remote_memcopy_buffer(tmp_wgt_ptr, 0, src_wgt_ptr, 0, 1)
        core.remote_memcopy_buffer(tmp_psum_ptr, 0, src_psum_ptr, 0, 1)
        core.remote_rpc_barrier()

        core.mxu_acquire_lock()
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
        core.mxu_release_lock()
        
        core.vpu_acquire_lock()
        core.vpu_reconfigure(vlen=32, vdtype=dtype)
        core.vpu_load_reg(tmp_ifm_ptr, 0, 0, 4)
        core.vpu_load_reg(tmp_wgt_ptr, 0, 4, 4)
        core.vpu_execute(VPUOperator.ADD, 0, 4, 8, inplace=False, burst_len=4)
        core.vpu_store_reg(tmp_ofm_ptr, 0, 8, 4)
        core.vpu_release_lock()
        
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

    device.verbose = True   # print debug messages
    device.run_kernels(max_steps=-1)
    
    reference = torch.matmul(ifm, wgt) + psum
    reference[0:4, :] = ifm[0:4, :] + wgt[0:4, :]  # Simulate the effect of the VPU operation
    simulated = ofm_ptrs[1].handle.content_view((M, N), dtype=acc_dtype)

    print(f"\n=== REFERENCE ===\n{reference}")
    print(f"\n=== SIMULATED ===\n{simulated}")
    print(f"\nsimulation terminated with valid result: {torch.allclose(reference, simulated)}")