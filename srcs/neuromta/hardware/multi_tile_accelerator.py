import math
import torch

from neuromta.framework import *

from neuromta.hardware.mem_context import *
from neuromta.hardware.icnt_context import *
from neuromta.hardware.mxu_context import *
from neuromta.hardware.vpu_context import *
from neuromta.hardware.npu import *


__all__ = [
    "MTAccelerator"
]


class MTAccelerator(Device):
    def __init__(
        self, 
        
        icnt_config: IcntConfig, 
        mem_config: MemConfig,
        mxu_config: MXUConfig,
        vpu_config: VPUConfig,
    ):
        super().__init__()
        
        # self.mem_space = MemorySpace()
        self.icnt_context = IcntContext(**icnt_config)
        self.mem_context = MemContext(**mem_config)
        
        self.mxu_config = mxu_config
        self.vpu_config = vpu_config
        
        self.npu_core_coords: list[tuple[int, int]] = self.icnt_context.core_map.core_coord(IcntCoreType.NPU)
        self.dma_core_coords: list[tuple[int, int]] = self.icnt_context.core_map.core_coord(IcntCoreType.DMA)
        
        self.npu_coord_to_core_idx_mappings: dict[tuple[int, int], int] = {
            coord: idx for idx, coord in enumerate(self.npu_core_coords)
        }
        
        self.dma_coord_to_core_idx_mappings: dict[tuple[int, int], int] = {
            coord: idx for idx, coord in enumerate(self.dma_core_coords)
        }

        self.npu_cores: list[NPUCore] = [
            NPUCore(coord=coord, mem_context=self.mem_context, icnt_context=self.icnt_context, mxu_config=self.mxu_config, vpu_config=self.vpu_config)
            for coord in self.npu_core_coords
        ]
        
    def create_circular_buffer_to_cores(self, cb_id: str, page_size: int, n_pages: int, coords: list[tuple[int, int]]=None) -> Pointer | list[Pointer]:
        if coords is None:
            coords = self.npu_core_coords
        if len(coords) == 2 and isinstance(coords[0], int) and isinstance(coords[1], int):
            coords = [coords]
            
        ptrs: list[Pointer] = []

        for coord in coords:
            core_idx = self.npu_coord_to_core_idx_mappings[coord]
            core = self.npu_cores[core_idx]
            ptr = Pointer(ptr_id=f"cb_{cb_id}_{core_idx}")
            core.cb_create(ptr, page_size=page_size, n_pages=n_pages)
            ptrs.append(ptr)

        if len(coords) == 1:
            return ptrs[0]
        return ptrs
    
    def create_l1_buffer_to_cores(self, bf_id: str, page_size: int, n_pages: int, coords: list[tuple[int, int]]=None) -> Pointer | list[Pointer]:
        if coords is None:
            coords = self.npu_core_coords
        if len(coords) == 2 and isinstance(coords[0], int) and isinstance(coords[1], int):
            coords = [coords]
            
        ptrs: list[Pointer] = []

        for coord in coords:
            core_idx = self.npu_coord_to_core_idx_mappings[coord]
            core = self.npu_cores[core_idx]
            ptr = Pointer(ptr_id=f"bf_{bf_id}_{core_idx}")
            core.local_buffer_allocate(ptr, page_size=page_size, n_pages=n_pages)
            ptrs.append(ptr)
        
        if len(coords) == 1:
            return ptrs[0]
        return ptrs
    

if __name__ == "__main__":
    core_map = IcntCoreMap.from_shape((2, 2))
    core_map.grid[:, 0] = IcntCoreType.DMA
    core_map.grid[:, 1] = IcntCoreType.NPU
    
    M = 32
    N = 32
    K = 32
    dtype = torch.int32
    acc_dtype = torch.int32
    
    icnt_config = IcntConfig(core_map=core_map)
    mem_config = MemConfig()
    mxu_config = MXUConfig(pe_arr_height=32, pe_arr_width=32, seq_len=32, dtype=dtype, acc_dtype=acc_dtype, dataflow=MXUDataflow.OS)
    vpu_config = VPUConfig()
    
    device = MTAccelerator(icnt_config=icnt_config, mem_config=mem_config, mxu_config=mxu_config, vpu_config=vpu_config)
    device.initialize()
    device.change_sim_model_options(use_cycle_model=True, use_functional_model=True)

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


# if __name__ == "__main__":
#     core_map = IcntCoreMap.from_shape((10, 10))
#     core_map.grid[:, 0   ] = IcntCoreType.DMA
#     core_map.grid[:, 5   ] = IcntCoreType.DMA
#     core_map.grid[:, 1:5 ] = IcntCoreType.NPU
#     core_map.grid[:, 6:10] = IcntCoreType.NPU
    
#     M = 32
#     N = 32
#     K = 32
#     dtype = torch.int32
#     acc_dtype = torch.int32
    
#     icnt_config = IcntConfig(core_map=core_map)
#     mem_config = MemConfig()
#     mxu_config = MXUConfig(pe_arr_height=32, pe_arr_width=32, seq_len=32, dtype=dtype, acc_dtype=acc_dtype, dataflow=MXUDataflow.OS)
#     vpu_config = VPUConfig()
    
#     device = MTAccelerator(icnt_config=icnt_config, mem_config=mem_config, mxu_config=mxu_config, vpu_config=vpu_config)
#     device.initialize()
#     device.change_sim_model_options(use_cycle_model=True, use_functional_model=True)

#     ifm  = torch.arange(0, M * K, dtype=dtype).reshape(M, K)
#     wgt  = torch.arange(0, K * N, dtype=dtype).reshape(K, N)
#     psum = torch.ones((M, N), dtype=acc_dtype)
#     ofm  = torch.zeros((M, N), dtype=acc_dtype)
    
#     ifm_size = ifm.numel() * ifm.element_size()
#     wgt_size = wgt.numel() * wgt.element_size()
#     psum_size = psum.numel() * psum.element_size()
#     ofm_size = ofm.numel() * ofm.element_size()
    
#     npu_core = device.npu_cores[0]
#     npu_coord = npu_core.coord
    
#     ifm_ptr  = device.create_l1_buffer_to_cores("ifm",  page_size=ifm_size,  n_pages=1, coords=npu_coord)
#     wgt_ptr  = device.create_l1_buffer_to_cores("wgt",  page_size=wgt_size,  n_pages=1, coords=npu_coord)
#     psum_ptr = device.create_l1_buffer_to_cores("psum", page_size=psum_size, n_pages=1, coords=npu_coord)
#     ofm_ptr  = device.create_l1_buffer_to_cores("ofm",  page_size=ofm_size,  n_pages=1, coords=npu_coord)
    
#     ifm_ptr.handle.set_content(ifm)
#     wgt_ptr.handle.set_content(wgt)
#     psum_ptr.handle.set_content(psum)
#     ofm_ptr.handle.set_content(ofm)

#     @core_kernel_method
#     def gemm_kernel(core: NPUCore, ifm_ptr: Pointer, wgt_ptr: Pointer, psum_ptr: Pointer, ofm_ptr: Pointer):
#         core.mxu_acquire_lock()
#         core.mxu_reconfigure(dtype=torch.int32, acc_dtype=torch.int32)
#         core.mxu_tiled_gemm(
#             ifm_ptr=ifm_ptr,
#             wgt_ptr=wgt_ptr,
#             psum_ptr=psum_ptr,
#             ofm_ptr=ofm_ptr,
#             preload_wgt=False,
#             preload_psum=True,
#             flush_ofm=True
#         )
#         core.mxu_release_lock()
        
#         core.vpu_acquire_lock()
#         core.vpu_reconfigure(vlen=32, vdtype=dtype)
#         core.vpu_load_reg(ifm_ptr, 0, 0, 4)
#         core.vpu_load_reg(wgt_ptr, 0, 4, 4)
#         core.vpu_execute(VPUOperator.ADD, 0, 4, 8, inplace=False, burst_len=4)
#         core.vpu_store_reg(ofm_ptr, 0, 8, 4)
#         core.vpu_release_lock()
        
#     gemm_kernel(npu_core, ifm_ptr, wgt_ptr, psum_ptr, ofm_ptr)

#     device.verbose = True   # print debug messages
#     device.run_kernels(max_steps=100000)
    
#     reference = torch.matmul(ifm, wgt) + psum
#     reference[0:4, :] = ifm[0:4, :] + wgt[0:4, :]  # Simulate the effect of the VPU operation
#     simulated = ofm_ptr.handle.content_view((M, N), dtype=acc_dtype)

#     print(f"\n=== REFERENCE ===\n{reference}")
#     print(f"\n=== SIMULATED ===\n{simulated}")
#     print(f"\nsimulation terminated with valid result: {torch.allclose(reference, simulated)}")
