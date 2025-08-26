from neuromta.framework import *

try:
    import pydramsim3
    PYDRAMSIM3_AVAILABLE = True
except ImportError as e:
    PYDRAMSIM3_AVAILABLE = False
    

__all__ = [
    "PYDRAMSIM3_AVAILABLE",
    "DRAMSim3"
]
    
    
class DRAMSim3(CompanionModule):
    def __init__(self, config_name: str):
        super().__init__()
        
        self._msys = pydramsim3.create_msys(
            config_file=pydramsim3.PYDRAMSIM_MSYS_CONFIG_PATH(config_name),
            output_dir=pydramsim3.PYDRAMSIM_DEFAULT_OUT_DIR
        )
        
    def update_cycle_time(self, cycle_time):
        pydramsim3.msys_cycle_step(msys=self._msys, cycles=cycle_time)

    def create_cmd(self, addr: int, size: int, is_write: bool):
        return pydramsim3.create_msys_cmd(addr=addr, size=size, is_write=is_write)
    
    def dispatch_cmd(self, cmd):
        pydramsim3.msys_dispatch_cmd(msys=self._msys, cmd=cmd)
        
    def check_cmd_executed(self, cmd):
        return pydramsim3.check_msys_cmd_executed(cmd=cmd)