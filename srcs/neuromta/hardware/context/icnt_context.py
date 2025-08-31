from neuromta.framework import *
from neuromta.hardware.companions.booksim import BookSim2Config


__all__ = [
    "IcntConfig",
    "IcntContext",
]


class IcntConfig(dict):
    def __init__(
        self,
        
        flit_size: int                  = parse_mem_cap_str("16B"),
        control_packet_size: int        = parse_mem_cap_str("32B"),
        booksim2_enable: bool           = False,
        booksim2_config: BookSim2Config = None,
    ):
        super().__init__()
        
        self["flit_size"] = flit_size
        self["control_packet_size"] = control_packet_size

        self["booksim2_config"] = booksim2_config
        self["booksim2_enable"] = booksim2_enable

class IcntContext:
    def __init__(
        self,
        
        flit_size: int,
        control_packet_size: int,
        booksim2_config: str=None,
        booksim2_enable: bool=False,
    ):
        self.flit_size = flit_size
        self.control_packet_size = control_packet_size
        
        self.booksim2_config = booksim2_config
        self.booksim2_enable = booksim2_enable

    def compute_hop_cnt(self, src_coord: tuple[int, int], dst_coord: tuple[int, int]) -> int:
        return abs(src_coord[0] - dst_coord[0]) + abs(src_coord[1] - dst_coord[1])
    
    def get_control_packet_latency(self, src_coord: tuple[int, int], dst_coord: tuple[int, int]) -> int:
        hop_cnt = self.compute_hop_cnt(src_coord, dst_coord)
        return hop_cnt + (self.control_packet_size // self.flit_size)
    
    def get_data_packet_latency(self, src_coord: tuple[int, int], dst_coord: tuple[int, int], data_size: int) -> int:
        hop_cnt = self.compute_hop_cnt(src_coord, dst_coord)
        return hop_cnt + (data_size // self.flit_size) + 1
