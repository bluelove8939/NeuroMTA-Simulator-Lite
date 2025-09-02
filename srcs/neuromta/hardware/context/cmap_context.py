import enum
from typing import Any

from neuromta.framework import *
from neuromta.hardware.companions.booksim import BookSim2Config, PYBOOKSIM2_AVAILABLE


__all__ = [
    "CmapCoreType",
    "CmapMemType",
    "CmapConfig",
    "CmapContext",
]


ICNT_CORE_NAME      = "ICNT"
MAIN_MEM_CORE_NAME  = "MAIN_MEM"
BOOKSIM_MODULE_ID   = "BOOKSIM"
DRAMSIM_MODULE_ID   = "DRAMSIM"


class CmapCoreType(enum.Enum):
    NPU     = enum.auto()
    DMA     = enum.auto()
    CACHE   = enum.auto()
    
    @property
    def is_memory_space_owner(self) -> bool:
        return self in (CmapCoreType.NPU, CmapCoreType.DMA)


class CmapMemType(enum.Enum):
    L1      = enum.auto()
    MAIN    = enum.auto()
    
    
class AddressSpaceEntry:
    def __init__(self, mem_type: CmapMemType, size: int):
        self.mem_type = mem_type
        self.size = size
        
        self.core_ids: list[tuple[int, int]] = []
        
    def add_core_id(self, coord: tuple[int, int]):
        self.core_ids.append(coord)


class CmapConfig:
    def __init__(
        self,
        
        shape: tuple[int, int], 
        n_l1_mem_bank: int,
        n_main_mem_channels: int,
        l1_mem_bank_size: int       = parse_mem_cap_str("1MB"),
        main_mem_channel_size: int  = parse_mem_cap_str("1GB"),
    ):
        super().__init__()
        
        self.shape                  = shape
        self.n_l1_mem_bank          = n_l1_mem_bank
        self.n_main_mem_channels    = n_main_mem_channels
        self.l1_mem_bank_size       = l1_mem_bank_size
        self.main_mem_channel_size  = main_mem_channel_size
        
        self.icnt_core_id           = ICNT_CORE_NAME
        self.main_mem_core_id       = MAIN_MEM_CORE_NAME
        
        self.booksim_module_id      = BOOKSIM_MODULE_ID
        self.dramsim_module_id      = DRAMSIM_MODULE_ID 
        
        self.core_map: dict[Any, CmapCoreType] = {}
        self.addr_space: dict[int, AddressSpaceEntry] = {}

        self.main_mem_base_addr = 0x00000000
        self.l1_mem_base_addr = self.main_mem_base_addr + self.n_main_mem_channels * self.main_mem_channel_size
        
        for i in range(self.n_main_mem_channels):
            base_addr = self.main_mem_base_addr + i * self.main_mem_channel_size
            self.addr_space[base_addr] = AddressSpaceEntry(mem_type=CmapMemType.MAIN, size=self.main_mem_channel_size)
        
        for i in range(self.n_l1_mem_bank):
            base_addr = self.l1_mem_base_addr + i * self.l1_mem_bank_size
            self.addr_space[base_addr] = AddressSpaceEntry(mem_type=CmapMemType.L1, size=self.l1_mem_bank_size)

    def add_core(self, core_type: CmapCoreType, core_id: Any, mem_bank_idx: int=None):
        if not (0 <= core_id[0] < self.shape[0]) or not (0 <= core_id[1] < self.shape[1]):
            raise ValueError(f"Coordinate {core_id} is out of bounds for core map shape {self.shape}.")
        if core_id in self.core_map.keys():
            raise ValueError(f"Coordinate {core_id} is already assigned to a core.")

        self.core_map[core_id] = core_type

        if core_type.is_memory_space_owner:
            if mem_bank_idx is None:
                raise Exception(f"[ERROR] Memory bank index is required for {core_type}.")

            if core_type == CmapCoreType.DMA:
                mem_type=CmapMemType.MAIN
                base_addr=self.main_mem_base_addr + mem_bank_idx * self.main_mem_channel_size
                size=self.main_mem_channel_size
            elif core_type == CmapCoreType.NPU:    
                mem_type=CmapMemType.L1
                base_addr=self.l1_mem_base_addr + mem_bank_idx * self.l1_mem_bank_size
                size=self.l1_mem_bank_size

            if base_addr not in self.addr_space.keys():
                raise Exception(f"[ERROR] Address {base_addr} is not mapped in the address space.")
            
            addr_space_entry = self.addr_space[base_addr]
            
            if addr_space_entry.mem_type != mem_type or addr_space_entry.size != size:
                raise ValueError(f"Address {base_addr} is already assigned to a different memory type.")
            
            addr_space_entry.core_ids.append(core_id)

    def get_core_ids(self, core_type: CmapCoreType) -> tuple:
        coords = [coord for coord, ctype in self.core_map.items() if ctype == core_type]
        if not coords:
            raise ValueError(f"No core of type {core_type} found in the core map.")
        return tuple(map(tuple, coords))
    
    def create_booksim2_config(self) -> BookSim2Config:
        if not PYBOOKSIM2_AVAILABLE:
            return None  # if pybooksim2 is not installed, we cannot create a BookSim2 configuration

        x_dim = self.shape[0]
        y_dim = self.shape[1]

        return BookSim2Config(
            subnets=1,  # TODO
            x=x_dim,
            y=y_dim,
            xr=1,   # no concentration by default
            yr=1,   # no concentration by default
        )
        
    def count_core(self, core_type: CmapCoreType) -> int:
        return sum(1 for ctype in self.core_map.values() if ctype == core_type)
    
    def check_main_mem_addr(self, addr: int) -> bool:
        return self.main_mem_base_addr <= addr < self.l1_mem_base_addr
    
    def check_l1_mem_addr(self, addr: int) -> bool:
        return self.l1_mem_base_addr <= addr < (self.l1_mem_base_addr + self.n_l1_mem_bank * self.l1_mem_bank_size)
    
    def get_main_mem_ch_id_from_addr(self, addr: int) -> int:
        if not self.check_main_mem_addr(addr):
            raise ValueError(f"Address {hex(addr)} is not in the main memory address range.")
        
        offset = addr - self.main_mem_base_addr
        return offset // self.main_mem_channel_size
    

class CmapContext:
    def __init__(
        self,
        
        config: CmapConfig,
    ):
        self._config = config
        
        self._base_addr_to_coord_mappings:  dict[int, tuple[tuple[int, int], int]] = {}
        self._coord_to_base_addr_mappings:  dict[tuple[int, int], int] = {}
        self._coord_to_main_ch_id_mappings: dict[tuple[int, int], int] = {}
        
        self._npu_core_coords: tuple[tuple[int, int]] = self.config.get_core_ids(CmapCoreType.NPU)
        self._dma_core_coords: tuple[tuple[int, int]] = self.config.get_core_ids(CmapCoreType.DMA)
        
        for base_addr, addr_space_entry in self.config.addr_space.items():
            size = addr_space_entry.size

            for coord in addr_space_entry.core_ids:
                self._base_addr_to_coord_mappings[base_addr] = (coord, size)
                self._coord_to_base_addr_mappings[coord] = base_addr
                
                if coord in self._dma_core_coords:
                    self._coord_to_main_ch_id_mappings[coord] = self.config.get_main_mem_ch_id_from_addr(base_addr)

    def get_base_addr_from_coord(self, coord: tuple[int, int]) -> int:
        if coord not in self._coord_to_base_addr_mappings:
            raise ValueError(f"Coordinate {coord} does not map to a valid base address.")
        return self._coord_to_base_addr_mappings[coord]
            
    def get_coord_from_address(self, addr: int, hash_src_coord: tuple[int, int]=None) -> tuple[int, int]:
        keys = sorted(self.config.addr_space.keys())
        left, right = 0, len(keys) - 1

        while left <= right:
            mid = (left + right) // 2
            base_addr = keys[mid]
            
            addr_space_entry = self.config.addr_space[base_addr]
            size = addr_space_entry.size
            coords = addr_space_entry.core_ids

            if base_addr <= addr < (base_addr + size):
                if hash_src_coord is None:
                    if len(coords) == 1:
                        return coords[0]
                    else:
                        raise Exception(f"[ERROR] No hash key is provided for address {hex(addr)} but the number of coords associated is more than 1. To avoid this exception, please provide a hash key.")
                else:
                    return coords[self._coord_hash_with(key="row", src_coord=hash_src_coord, n_entry=len(coords))]
            if base_addr > addr:
                right = mid - 1
            else:
                left = mid + 1
                
        return []  # Address does not map to any core
    
    def get_node_id_from_coord(self, coord: tuple[int, int]) -> int:
        core_grid_width = self.config.shape[-1]
        return coord[0] * core_grid_width + coord[1]
    
    @property
    def config(self) -> CmapConfig:
        return self._config

    @property
    def npu_core_coords(self) -> tuple[tuple[int, int]]:
        return self._npu_core_coords

    @property
    def dma_core_coords(self) -> tuple[tuple[int, int]]:
        return self._dma_core_coords
    
    @property
    def icnt_core_id(self) -> str:
        return self._config.icnt_core_id
    
    @property
    def main_mem_core_id(self) -> str:
        return self._config.main_mem_core_id
    
    @staticmethod
    def _coord_hash_with(key: str, src_coord: tuple[int, int], n_entry: int) -> int:
        if key == "row":
            return src_coord[0] % n_entry
        raise ValueError(f"Unknown key: {key}")

