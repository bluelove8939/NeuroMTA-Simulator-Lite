import enum
import torch

from neuromta.common.device import Device

from neuromta.ip.hardware.npu import NPUCore, DMACore, IcntNetworkCore
from neuromta.ip.hardware.matrix_unit import MXUConfig, MXUDataflow
from neuromta.ip.hardware.vector_unit import VPUConfig, VPUOperator
from neuromta.ip.hardware.memory import MemoryContext, parse_mem_cap_str
from neuromta.ip.hardware.interconnect import IcntNetworkContext


__all__ = [
    "TenstorrentCoreType",
    "TenstorrentCoreMap",
    "TenstorrentDevice"
]


class TenstorrentCoreType(int):
    EMPTY   = 0
    NPU     = 1
    DMA     = 2
    
    @classmethod
    def is_valid_core_type(cls, core_type: 'TenstorrentCoreType') -> bool:
        return 0 <= core_type <= 2


class TenstorrentCoreMap:
    def __init__(self, grid: torch.Tensor):
        self._grid = grid
        
        for core_type in self._grid.flatten().unique():
            if not TenstorrentCoreType.is_valid_core_type(core_type):
                raise TypeError(f"The core type {core_type} is not a valid Tenstorrent core type.")
        
    def core_coord(self, core_type: TenstorrentCoreType) -> tuple[int, int]:
        coords = torch.argwhere(self._grid == core_type)
        if coords.size == 0:
            raise ValueError(f"No core of type {core_type} found in the core map.")
        return tuple(map(tuple, coords.tolist()))
        
    @classmethod
    def from_shape(cls, shape: tuple[int, int]) -> 'TenstorrentCoreMap':
        core_map = torch.full(shape, TenstorrentCoreType.EMPTY, dtype=int)
        return cls(core_map)
    
    @property
    def grid(self) -> torch.Tensor:
        return self._grid


class TenstorrentDevice(Device):
    def __init__(self, core_map: TenstorrentCoreMap):
        super().__init__()
        
        self.mem_context = MemoryContext()
        self.icnt_context = IcntNetworkContext(grid_shape=core_map.grid.shape)
        self.mxu_config = MXUConfig(pe_arr_height=32, pe_arr_width=32, seq_len=32, dtype=torch.int32, acc_dtype=torch.int32, dataflow=MXUDataflow.OS, op_latency_per_byte=1)
        self.vpu_config = VPUConfig(vreg_len=parse_mem_cap_str("128B"), vreg_num=32, vdtype=torch.int32, vlen_max=1024, vlen_min=32)

        self.npu_cores: list[NPUCore] = [
            NPUCore(coord=coord, mem_context=self.mem_context, icnt_context=self.icnt_context, mxu_config=self.mxu_config, vpu_config=self.vpu_config)
            for coord in core_map.core_coord(TenstorrentCoreType.NPU)
        ]
        
        self.dma_cores: list[DMACore] = [
            DMACore(coord=coord, mem_context=self.mem_context, icnt_context=self.icnt_context)
            for coord in core_map.core_coord(TenstorrentCoreType.DMA)
        ]
        
        self.icnt_core = IcntNetworkCore(icnt_context=self.icnt_context)
        

if __name__ == "__main__":
    from neuromta.common.core import *
    
    from neuromta.common.core import core_kernel_method
    from neuromta.common.buffer_handle import BufferHandle, CircularBufferHandle, PageHandle, TemporaryBufferHandle
    
    core_map = TenstorrentCoreMap.from_shape((10, 10))
    core_map.grid[:, 0   ] = TenstorrentCoreType.DMA
    core_map.grid[:, 5   ] = TenstorrentCoreType.DMA
    core_map.grid[:, 1:5 ] = TenstorrentCoreType.NPU
    core_map.grid[:, 6:10] = TenstorrentCoreType.NPU
    
    print(core_map.grid)
    
    device = TenstorrentDevice(core_map)
    device.initialize(create_trace=False)
    device.change_sim_model_options(use_cycle_model=True, use_functional_model=True)
    
    # Output Stationary Single Tile GEMM Kernel
    @core_kernel_method
    def tiled_gemm_os_kernel(
        core: NPUCore, 
        ifm_handle: BufferHandle, wgt_handle: BufferHandle, ofm_handle: BufferHandle,
        M: int, N: int, K: int,
        m_tile: int, n_tile: int, k_tile: int,
    ):
        m_tile_num = M // m_tile
        n_tile_num = N // n_tile
        k_tile_num = K // k_tile
        
        for n_tile_idx in range(n_tile_num):
            for m_tile_idx in range(m_tile_num):
                # core.create_new_parallel_kernel()
                start_parallel_kernel()
                
                ifm_page_idx = m_tile_idx * k_tile_num
                wgt_page_idx = n_tile_idx * k_tile_num
                ofm_page_idx = m_tile_idx * n_tile_num + n_tile_idx
                
                ifm_buffer_offset = ifm_page_idx * ifm_handle.page_size
                wgt_buffer_offset = wgt_page_idx * wgt_handle.page_size
                ofm_buffer_offset = ofm_page_idx * ofm_handle.page_size
                
                tmp_ofm_handle = TemporaryBufferHandle(page_size=ofm_handle.page_size, n_pages=1)

                core._atom_acquire_mxu_lock()
                core._atom_mxu_tiled_gemm(
                    ifm_handle, ifm_buffer_offset, 
                    wgt_handle, wgt_buffer_offset,
                    None, 0,    # No PSUM handle
                    tmp_ofm_handle, 0,
                    streaming_n_tiles=k_tile_num,
                    skip_wgt_preload=True,
                    skip_psum_preload=True,
                    skip_ofm_flush=False,
                )
                core._atom_release_mxu_lock()
                
                if isinstance(ofm_handle, CircularBufferHandle):
                    core.cb_reserve_back(ofm_handle, 1)
                    core.copy_page(src_handle=tmp_ofm_handle, src_page_idx=0, dst_handle=ofm_handle, dst_page_idx=ofm_page_idx, n_pages=1)
                    core.cb_push_back(ofm_handle, 1)
                else:
                    core.copy_page(src_handle=tmp_ofm_handle, src_page_idx=0, dst_handle=ofm_handle, dst_page_idx=ofm_page_idx, n_pages=1)
                    
                end_parallel_kernel()
                    
        # core.merge_parallel_kernels()

    # GEMM #1
    core1 = device.npu_cores[0]
    
    ifm_1 = torch.randint(0, 225, (64, 64), dtype=torch.int32)
    wgt_1 = torch.randint(0, 225, (64, 64), dtype=torch.int32)
    
    ifm_tiles_1 = ifm_1.reshape((2, 32, 2, 32)).permute(0, 2, 1, 3).reshape((-1, 32, 32))   # Mt, Kt, M, K
    wgt_tiles_1 = wgt_1.reshape((2, 32, 2, 32)).permute(2, 0, 1, 3).reshape((-1, 32, 32))   # Nt, Kt, K, N
    
    ifm_buffer_1 = BufferHandle("ifm_buffer_1", addr=device.mem_context.get_main_mem_addr(0, 0, 0), page_size=32*32*4, n_pages=4)
    wgt_buffer_1 = BufferHandle("wgt_buffer_1", addr=device.mem_context.get_main_mem_addr(1, 0, 0), page_size=32*32*4, n_pages=4)
    ofm_buffer_1 = CircularBufferHandle("ofm_buffer_1", addr=device.mem_context.get_l1_mem_addr(core1.mem_seg_id, 0, 0), page_size=32*32*4, n_pages=8)
    
    for i in range(ifm_tiles_1.shape[0]):
        ifm_buffer_1.add_page(i, PageHandle(content=ifm_tiles_1[i], page_size=32*32*4))
    for i in range(wgt_tiles_1.shape[0]):
        wgt_buffer_1.add_page(i, PageHandle(content=wgt_tiles_1[i], page_size=32*32*4))
        
    tiled_gemm_os_kernel(
        core1, ifm_buffer_1, wgt_buffer_1, ofm_buffer_1,
        M=64, N=64, K=64,
        m_tile=32, n_tile=32, k_tile=32,
    )
    
    device.verbose = True
    device.run_kernels()
    
    for i in range(ofm_buffer_1.n_pages):
        print(f"Output Page {i}: {ofm_buffer_1.get_page(i).content}")