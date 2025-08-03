import enum
import numpy as np

from neuromta.common.device import Device

from neuromta.ip.hardware.core.npu import NPUCore, DMACore, IcntNetworkCore
from neuromta.ip.hardware.core.matrix_unit import MXUConfig, MXUDataflow
from neuromta.ip.hardware.core.vector_unit import VPUConfig, VPUOperator
from neuromta.ip.hardware.core.memory import MemoryContext, parse_mem_cap_str
from neuromta.ip.hardware.core.interconnect import IcntNetworkContext


class TenstorrentCoreType(enum.Enum):
    EMPTY   = enum.auto()
    NPU     = enum.auto()
    DMA     = enum.auto()


class TenstorrentCoreMap:
    def __init__(self, grid: np.ndarray[int]):
        self._grid = grid
        
    def core_coord(self, core_type: TenstorrentCoreType) -> tuple[int, int]:
        coords = np.argwhere(self._grid == core_type.value)
        if coords.size == 0:
            raise ValueError(f"No core of type {core_type} found in the core map.")
        return tuple(map(tuple, coords.tolist()))
        
    @classmethod
    def from_shape(cls, shape: tuple[int, int]) -> 'TenstorrentCoreMap':
        core_map = np.full(shape, TenstorrentCoreType.EMPTY.value, dtype=int)
        return cls(core_map)
    
    @property
    def grid(self) -> np.ndarray[int]:
        return self._grid


class TenstorrentDevice(Device):
    def __init__(self, core_map: TenstorrentCoreMap):
        super().__init__()
        
        self.mem_context = MemoryContext()
        self.icnt_context = IcntNetworkContext(grid_shape=core_map.grid.shape)
        self.mxu_config = MXUConfig(pe_arr_height=32, pe_arr_width=32, seq_len=32, dtype=np.int32, acc_dtype=np.int32, dataflow=MXUDataflow.OS, op_latency_per_byte=1)
        self.vpu_config = VPUConfig(vreg_len=parse_mem_cap_str("128B"), vreg_num=32, vdtype=np.int32, vlen_max=1024, vlen_min=32)

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
    core_map = TenstorrentCoreMap.from_shape((4, 4))
    core_map.grid[0,  :] = TenstorrentCoreType.DMA.value
    core_map.grid[1:, :] = TenstorrentCoreType.NPU.value
    
    print(core_map.grid)
    
    device = TenstorrentDevice(core_map)
    device.initialize(create_trace=False)
    device.change_sim_model_options(use_cycle_model=True, use_functional_model=True)
    
    for core_idx, core in enumerate(device.npu_cores):
        print(f"NPU #{core_idx:<2d} -> {core.coord}")
        
    for core_idx, core in enumerate(device.dma_cores):
        print(f"DMA #{core_idx:<2d} -> {core.coord}")