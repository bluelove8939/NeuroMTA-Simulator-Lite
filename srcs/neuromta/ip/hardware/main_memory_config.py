import math

from neuromta.common.parser_utils import parse_freq_str


__all__ = [
    "MainMemoryConfig",
    "HBM2Config",
]


class MainMemoryConfig(object):
    def __init__(self, transfer_speed: int, ch_io_width: int, ch_num: int, burst_len: int, is_ddr: bool, processor_clock_freq: int):
        self.transfer_speed         = transfer_speed    # transfer speed per pin (MT/s)
        self.ch_io_width            = ch_io_width       # io channel width (bits)
        self.ch_num                 = ch_num            # number of channels
        self.burst_len              = burst_len         # burst length
        self.is_ddr                 = is_ddr
        self.processor_clock_freq   = processor_clock_freq
        
    def get_cycles(self, size: int) -> int:
        self.transfer_speed_bytes = (self.transfer_speed * (2 ** 20) * self.ch_io_width * self.ch_num // 8)  # Byte/s
        self.transfer_speed_per_cycles = self.transfer_speed_bytes / self.processor_clock_freq   # Byte/cycle
        
        return math.ceil(size / self.transfer_speed_per_cycles)
    

class HBM2Config(MainMemoryConfig):
    def __init__(self):
        super().__init__(
            transfer_speed=1600, 
            ch_io_width=1024, 
            ch_num=1, 
            burst_len=256, 
            is_ddr=True, 
            processor_clock_freq=parse_freq_str("1GHz")
        )