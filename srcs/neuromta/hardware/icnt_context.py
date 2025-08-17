import enum
import torch

from neuromta.common import *


__all__ = [
    "IcntCoreType",
    "IcntCoreMap",
    "IcntConfig",
    "IcntContext",
]


class IcntCoreType(int):
    EMPTY   = 0
    NPU     = 1
    DMA     = 2
    
    @classmethod
    def is_valid_core_type(cls, core_type: 'IcntCoreType') -> bool:
        return 0 <= core_type <= 2


class IcntCoreMap:
    def __init__(self, grid: torch.Tensor):
        self._grid = grid
        
        for core_type in self._grid.flatten().unique():
            if not IcntCoreType.is_valid_core_type(core_type):
                raise TypeError(f"The core type {core_type} is not a valid Tenstorrent core type.")

    def core_coord(self, core_type: IcntCoreType) -> tuple[int, int]:
        coords = torch.argwhere(self._grid == core_type)
        if coords.size == 0:
            raise ValueError(f"No core of type {core_type} found in the core map.")
        return tuple(map(tuple, coords.tolist()))
        
    @classmethod
    def from_shape(cls, shape: tuple[int, int]) -> 'IcntCoreMap':
        core_map = torch.full(shape, IcntCoreType.EMPTY, dtype=int)
        return cls(core_map)
    
    @property
    def grid(self) -> torch.Tensor:
        return self._grid


class IcntConfig(dict):
    def __init__(
        self,
        
        core_map: IcntCoreMap,
        l1_mem_bank_size: int = parse_mem_cap_str("1MB"),
        main_mem_bank_size: int = parse_mem_cap_str("1GB"),
        flit_size: int = parse_mem_cap_str("16B"),
        control_packet_size: int = parse_mem_cap_str("32B"),
    ):
        super().__init__()
        
        self["core_map"] = core_map
        self["l1_mem_bank_size"] = l1_mem_bank_size
        self["main_mem_bank_size"] = main_mem_bank_size
        self["flit_size"] = flit_size
        self["control_packet_size"] = control_packet_size

class IcntContext:
    def __init__(
        self,
        
        core_map: IcntCoreMap,
        l1_mem_bank_size: int,
        main_mem_bank_size: int,
        flit_size: int,
        control_packet_size: int,
    ):
        self._core_map = core_map
        self._l1_mem_bank_size = l1_mem_bank_size
        self._main_mem_bank_size = main_mem_bank_size
        self._flit_size = flit_size
        self._control_packet_size = control_packet_size
        
        self._coord_to_mem_handle_mappings: dict[tuple[int, int], MemoryHandle] = {}
        self._base_addr_to_coord_mappings:  dict[int, tuple[int, int]] = {}
        self._main_mem_ids: list[str] = []
        self._l1_mem_ids: list[str] = []

        mem_base_addr = 0x00000000

        for coord in self._core_map.core_coord(IcntCoreType.DMA):
            mem_id = f"MAIN{tuple(coord)}"
            
            self._base_addr_to_coord_mappings[mem_base_addr] = coord
            self._main_mem_ids.append(mem_id)

            mem_base_addr += self._main_mem_bank_size
        
        for coord in self._core_map.core_coord(IcntCoreType.NPU):
            mem_id = f"L1{tuple(coord)}"
            
            self._base_addr_to_coord_mappings[mem_base_addr] = coord
            self._l1_mem_ids.append(mem_id)

            mem_base_addr += self._l1_mem_bank_size
    
    @property
    def core_map(self) -> IcntCoreMap:
        return self._core_map
