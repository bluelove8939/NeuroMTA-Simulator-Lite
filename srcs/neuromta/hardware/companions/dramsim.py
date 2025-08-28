import os
import math

from neuromta.framework import *

try:
    import pydramsim3
    PYDRAMSIM3_AVAILABLE = True
except ImportError as e:
    PYDRAMSIM3_AVAILABLE = False
    

__all__ = [
    "PYDRAMSIM3_AVAILABLE",
    "DRAMSim3",
    "DRAMSim3Config"
]


class DRAMSim3Config:
    def __init__(
        self, 
        config_path: str,  #="GDDR5_8Gb_x32", 
        processor_clock_freq: int,  #=parse_freq_str("1GHz"),
        cmd_wait_resolution: int
    ):  
        if not os.path.isfile(config_path):
            config_path = pydramsim3.PYDRAMSIM_MSYS_CONFIG_PATH(config_path)
        if not os.path.isfile(config_path):
            raise FileNotFoundError(f"[ERROR] DRAMSim3 config file '{config_path}' not found.")

        self.config_path = config_path
        self.processor_clock_freq = processor_clock_freq
        
        # NOTE: This factor determines how frequent the cycle-level simulator waiting check is. The waiting check 
        # will be done for every (expected_cmd_cycles / cmd_wait_resolution). This factor is important for balancing 
        # performance and simulation accuracy. If you want to increase the accuracy of the simulation, you can 
        # increase this value or simply set this value to None. For example, if expected cmd cycle is 100 and the 
        # factor is 50, the icnt core will check for every 2 cycles.
        self.cmd_wait_resolution = cmd_wait_resolution


class DRAMSim3(CompanionModule):
    def __init__(self, config: DRAMSim3Config):
        super().__init__()
        
        self.config = config
        
        self._msys = pydramsim3.create_msys(
            config_file=self.config.config_path,
            output_dir=pydramsim3.PYDRAMSIM_DEFAULT_OUT_DIR
        )
        
        self._mem_clock_time = pydramsim3.msys_get_tck(self._msys)
        self._ref_clock_time = 1 / (self.config.processor_clock_freq * (1e-9))
        self._rem_clock_sync_time = 0

    def update_cycle_time(self, cycle_time):
        self._rem_clock_sync_time += cycle_time * self._ref_clock_time
        
        mem_cycles = math.floor(self._rem_clock_sync_time / self._mem_clock_time)
        self._rem_clock_sync_time -= mem_cycles * self._mem_clock_time
        
        pydramsim3.msys_cycle_step(msys=self._msys, cycles=mem_cycles)

    def create_cmd(self, addr: int, size: int, is_write: bool):
        return pydramsim3.create_msys_cmd(addr=addr, size=size, is_write=is_write)
    
    def dispatch_cmd(self, cmd) -> bool:
        return pydramsim3.msys_dispatch_cmd(msys=self._msys, cmd=cmd)
        
    def check_cmd_executed(self, cmd) -> bool:
        return pydramsim3.check_msys_cmd_executed(cmd=cmd)
    
    def get_cmd_wait_check_interval(self, cmd) -> int:
        if self.config.cmd_wait_resolution is None:
            return 1

        mem_to_ref_clock_ratio = self._mem_clock_time / self._ref_clock_time

        cycles = pydramsim3.get_expected_cmd_cycles(msys=self._msys, cmd=cmd)
        cycles *= mem_to_ref_clock_ratio
        cycles = math.floor(cycles / self.config.cmd_wait_resolution)
        return max(cycles, 1)