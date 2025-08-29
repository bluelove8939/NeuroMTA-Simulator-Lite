from neuromta.framework import *

from neuromta.hardware.multi_tile.context.mem_context import MemContext
from neuromta.hardware.multi_tile.context.cmap_context import CmapContext

from neuromta.hardware.companions.dramsim import PYDRAMSIM3_AVAILABLE, DRAMSim3


__all__ = [
    "DRAMSimCore"
]


class DRAMSimCore(Core):
    def __init__(
        self,
        mem_context: MemContext, 
        cmap_context: CmapContext,
    ):
        super().__init__(
            core_id=cmap_context.dramsim_core_id, 
            cycle_model=DRAMSimCoreCycleModel(core=self)
        )
        
        self.mem_context = mem_context
        self.cmap_context = cmap_context
        
        if self.mem_context.main_config.dramsim3_enable and PYDRAMSIM3_AVAILABLE:
            self.dramsim3_module = DRAMSim3(config=self.mem_context.main_config.dramsim3_config)
            self.register_companion_module("DRAMSIM3", self.dramsim3_module)
        else:
            self.dramsim3_module = None

    #############################################################
    # DRAMSim3 Commands
    #############################################################  
    
    @property
    def is_dramsim3_enabled(self) -> bool:
        return PYDRAMSIM3_AVAILABLE and self.mem_context.main_config.dramsim3_enable
            
    @core_conditional_command_method
    def dramsim_send_mem_cmd(self, cmd):
        return self.dramsim3_module.dispatch_cmd(cmd)
        
    @core_conditional_command_method
    def dramsim_wait_mem_cmd(self, cmd):
        return self.dramsim3_module.check_cmd_executed(cmd)
    
    @core_kernel_method
    def dramsim_mem_load_page(self, ptr: Pointer):
        cmd = self.dramsim3_module.create_cmd(addr=ptr.addr, size=ptr.size, is_write=True)
        self.dramsim_send_mem_cmd(cmd)
        self.dramsim_wait_mem_cmd(cmd)
        
    @core_kernel_method
    def dramsim_mem_store_page(self, ptr: Pointer):
        cmd = self.dramsim3_module.create_cmd(addr=ptr.addr, size=ptr.size, is_write=False)
        self.dramsim_send_mem_cmd(cmd)
        self.dramsim_wait_mem_cmd(cmd)
        
class DRAMSimCoreCycleModel(CoreCycleModel):
    def __init__(self, core: DRAMSimCore):
        super().__init__()
        
        self.core = core
        
    # def dramsim_wait_mem_cmd(self, cmd):
    #     # return self.core.dramsim3_module.get_cmd_wait_check_interval(cmd)
    #     # return self.core.mem_context.main_config.get_cycles(size=4096) // 5
    #     return 1