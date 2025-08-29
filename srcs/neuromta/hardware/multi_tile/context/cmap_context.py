import torch
from typing import Sequence

from neuromta.framework import *
from neuromta.hardware.companions.booksim import BookSim2Config, PYBOOKSIM2_AVAILABLE


__all__ = [
    "CmapCoreType",
    "CmapConfig",
    "CmapContext",
]


ICNT_CORE_NAME = "ICNT"
DRAMSIM_CORE_NAME = "DRAMSIM"
BOOKSIM_CORE_NAME = "BOOKSIM"


class CmapCoreType(int):
    EMPTY   = 0
    NPU     = 1
    DMA     = 2
    
    @classmethod
    def is_valid_core_type(cls, core_type: 'CmapCoreType') -> bool:
        return 0 <= core_type <= 2

class CmapConfig:
    def __init__(
        self,
        
        grid: torch.Tensor,
        icnt_core_id: str   = ICNT_CORE_NAME,
        l1_mem_bank_size: int   = parse_mem_cap_str("1MB"),
        main_mem_bank_size: int = parse_mem_cap_str("1GB"),
    ):
        super().__init__()
        
        self._grid = grid
        self.icnt_core_id       = icnt_core_id
        self.l1_mem_bank_size   = l1_mem_bank_size
        self.main_mem_bank_size = main_mem_bank_size
        
        self.dramsim_core_id    = DRAMSIM_CORE_NAME
        self.booksim_core_id    = BOOKSIM_CORE_NAME

        for core_type in self._grid.flatten().unique():
            if not CmapCoreType.is_valid_core_type(core_type):
                raise TypeError(f"The core type {core_type} is not a valid Tenstorrent core type.")

    def core_coord(self, core_type: CmapCoreType) -> tuple[tuple[int, int]]:
        coords = torch.argwhere(self._grid == core_type)
        if coords.size == 0:
            raise ValueError(f"No core of type {core_type} found in the core map.")
        return tuple(map(tuple, coords.tolist()))
    
    def create_booksim2_config(self, cmd_wait_resolution: int) -> BookSim2Config:
        if not PYBOOKSIM2_AVAILABLE:
            return None  # if pybooksim2 is not installed, we cannot create a BookSim2 configuration

        if self._grid.numel() == 0:
            raise ValueError("The core map grid is empty. Cannot create BookSim2 configuration.")
        
        x_dim, y_dim = self._grid.shape
        
        return BookSim2Config(
            subnets=1,  # TODO
            x=x_dim,
            y=y_dim,
            xr=1,   # no concentration by default
            yr=1,   # no concentration by default
            cmd_wait_resolution=cmd_wait_resolution,
        )
        
    def count_core(self, core_type: CmapCoreType) -> int:
        return torch.count_nonzero(self._grid == core_type).item()
        
    @classmethod
    def from_shape(
        cls, 
        
        shape: tuple[int, int], 
        icnt_core_id=ICNT_CORE_NAME, 
        l1_mem_bank_size: int   = parse_mem_cap_str("1MB"),
        main_mem_bank_size: int = parse_mem_cap_str("1GB"),
    ) -> 'CmapConfig':
        core_map = torch.full(shape, CmapCoreType.EMPTY, dtype=int)
        return cls(core_map, icnt_core_id, l1_mem_bank_size, main_mem_bank_size)
    
    @property
    def grid(self) -> torch.Tensor:
        return self._grid
    
    @property
    def shape(self) -> tuple[int, int]:
        return self._grid.shape


class CmapContext:
    def __init__(
        self,
        
        core_map_config: CmapConfig,
    ):
        self._core_map_config = core_map_config
        
        self._base_addr_to_coord_mappings:  dict[int, tuple[tuple[int, int], int]] = {}
        self._coord_to_base_addr_mappings:  dict[tuple[int, int], int] = {}
        
        self._npu_core_coords: tuple[tuple[int, int]] = self.core_map.core_coord(CmapCoreType.NPU)
        self._dma_core_coords: tuple[tuple[int, int]] = self.core_map.core_coord(CmapCoreType.DMA)

        mem_base_addr = 0x00000000

        for coord in self._npu_core_coords:
            self._base_addr_to_coord_mappings[mem_base_addr] = (coord, self.core_map.main_mem_bank_size)
            self._coord_to_base_addr_mappings[coord] = mem_base_addr
            mem_base_addr += self.core_map.main_mem_bank_size
        
        for coord in self._dma_core_coords:
            self._base_addr_to_coord_mappings[mem_base_addr] = (coord, self.core_map.l1_mem_bank_size)
            self._coord_to_base_addr_mappings[coord] = mem_base_addr
            mem_base_addr += self.core_map.l1_mem_bank_size

    def get_base_addr_from_coord(self, coord: tuple[int, int]) -> int:
        if coord not in self._coord_to_base_addr_mappings:
            raise ValueError(f"Coordinate {coord} does not map to a valid base address.")
        return self._coord_to_base_addr_mappings[coord]
            
    def get_coord_from_address(self, addr: int) -> tuple[int, int]:
        keys = sorted(self._base_addr_to_coord_mappings.keys())
        left, right = 0, len(keys) - 1

        while left <= right:
            mid = (left + right) // 2
            base_addr = keys[mid]
            coord, size = self._base_addr_to_coord_mappings[base_addr]

            if base_addr <= addr < (base_addr + size):
                return coord
            if base_addr > addr:
                right = mid - 1
            else:
                left = mid + 1
                
        return None  # Address does not map to any core
    
    def get_node_id_from_coord(self, coord: tuple[int, int]) -> int:
        core_grid_width = self.core_map.grid.shape[-1]
        return coord[0] * core_grid_width + coord[1]
    
    @property
    def core_map(self) -> CmapConfig:
        return self._core_map_config

    @property
    def npu_core_coords(self) -> tuple[tuple[int, int]]:
        return self._npu_core_coords

    @property
    def dma_core_coords(self) -> tuple[tuple[int, int]]:
        return self._dma_core_coords
    
    @property
    def icnt_core_id(self) -> str:
        return self._core_map_config.icnt_core_id
    
    @property
    def dramsim_core_id(self) -> str:
        return self._core_map_config.dramsim_core_id
    
    @property
    def booksim_core_id(self) -> str:
        return self._core_map_config.booksim_core_id
