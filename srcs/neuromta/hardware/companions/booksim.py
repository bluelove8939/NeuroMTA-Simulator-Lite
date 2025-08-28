import math
from ctypes import c_void_p
from neuromta.framework import *

try:
    import pybooksim2
    PYBOOKSIM2_AVAILABLE = True
except ImportError as e:
    PYBOOKSIM2_AVAILABLE = False
 
    
__all__ = [
    "PYBOOKSIM2_AVAILABLE",
    "BookSim2",
    "BookSim2Config",
]


class BookSim2Config:
    def __init__(self, subnets: int, x: int, y: int, xr: int, yr: int, cmd_wait_resolution: int):
        if not PYBOOKSIM2_AVAILABLE:
            raise RuntimeError("[ERROR] BookSim2 is not available. Please install pybooksim2 to use this module.")

        self._config: c_void_p = pybooksim2.create_config_torus_2d(subnets, x, y, xr, yr)
        self._is_registered: bool = False
        
        # NOTE: This factor determines how frequent the cycle-level simulator waiting check is. The waiting check 
        # will be done for every (expected_cmd_cycles / cmd_wait_resolution). This factor is important for balancing 
        # performance and simulation accuracy. If you want to increase the accuracy of the simulation, you can 
        # increase this value or simply set this value to None. For example, if expected cmd cycle is 100 and the 
        # factor is 50, the icnt core will check for every 2 cycles.
        self.cmd_wait_resolution: int = cmd_wait_resolution
    
    def create_icnt(self) -> c_void_p:
        if self.is_registered:
            raise RuntimeError("[ERROR] Cannot creatae interconnect network with this config since it is already registered.")
        
        self._is_registered = True
        return pybooksim2.create_icnt(config=self._config)
    
    def update(self, **kwargs):
        if self.is_registered:
            raise RuntimeError("[ERROR] Cannot update this config since it is already registered.")

        for key, value in kwargs.items():
            if hasattr(self._config, key):
                setattr(self._config, key, value)
            else:
                raise AttributeError(f"[ERROR] BookSim2Config has no attribute '{key}'")

    @property
    def is_registered(self) -> bool:
        return self._is_registered


class BookSim2(CompanionModule):
    def __init__(self, config: BookSim2Config):
        super().__init__()
        
        if not PYBOOKSIM2_AVAILABLE:
            raise RuntimeError("[ERROR] BookSim2 is not available. Please install pybooksim2 to use this module.")

        self._icnt = config.create_icnt()
        self.config = config

    def update_cycle_time(self, cycle_time):
        pybooksim2.icnt_cycle_step(icnt=self._icnt, cycles=cycle_time)
        
    def create_data_cmd(self, src_id: int, dst_id: int, subnet: int, size: int, is_write: bool, is_response: bool):
        return pybooksim2.create_icnt_cmd_data_packet(src_id, dst_id, subnet, size, is_write, is_response)

    def create_control_cmd(self, src_id: int, dst_id: int, subnet: int, size: int, is_response: bool):
        return pybooksim2.create_icnt_cmd_control_packet(src_id, dst_id, subnet, size, is_response)

    def dispatch_cmd(self, cmd):
        return pybooksim2.icnt_dispatch_cmd(icnt=self._icnt, cmd=cmd)
        
    def check_cmd_executed(self, cmd) -> bool:
        return pybooksim2.check_icnt_cmd_received(cmd=cmd)
    
    def get_cmd_wait_check_interval(self, cmd) -> int:
        if self.config.cmd_wait_resolution is None:
            return 1
        
        cycles = pybooksim2.get_expected_cmd_cycles(cmd=cmd)
        cycles = math.floor(cycles / self.config.cmd_wait_resolution)
        return max(cycles, 1)
