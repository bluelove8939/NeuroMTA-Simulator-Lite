import math

from neuromta.framework import *

from neuromta.hardware.multi_tile.context.cmap_context import CmapContext
from neuromta.hardware.multi_tile.context.icnt_context import IcntContext

from neuromta.hardware.companions.booksim import PYBOOKSIM2_AVAILABLE, BookSim2


class BookSimCore(Core):
    def __init__(
        self,
        icnt_context: IcntContext,
        cmap_context: CmapContext,
    ):
        super().__init__(
            core_id=cmap_context.booksim_core_id,
            cycle_model=BookSimCoreCycleModel(core=self)
        )
        
        self.icnt_context = icnt_context
        self.cmap_context = cmap_context
        
        if self.icnt_context.booksim2_enable and not PYBOOKSIM2_AVAILABLE:
            raise RuntimeError("[ERROR] BookSim2 is not available. Please install pybooksim2 to enable BookSim2 support.")
        
        if self.is_booksim2_enabled:
            self.booksim2_module = BookSim2(config=self.icnt_context.booksim2_config)
            self.register_companion_module("BOOKSIM2", self.booksim2_module)
            
    #############################################################
    # BookSim2 Commands
    #############################################################  
        
    @property
    def is_booksim2_enabled(self) -> bool:
        return PYBOOKSIM2_AVAILABLE and self.icnt_context.booksim2_enable
    
    @core_conditional_command_method
    def booksim_send_packet_with_cmd(self, cmd):
        return self.booksim2_module.dispatch_cmd(cmd)
    
    @core_conditional_command_method
    def booksim_wait_cmd_and_handle(self, cmd):
        return self.booksim2_module.check_cmd_executed(cmd)
    
    @core_kernel_method
    def booksim_create_data_read_transaction(self, src_id: int, dst_id: int, size: int):
        if self.is_booksim2_enabled:    # TODO: subnet index  
            n_flits = math.ceil(size / self.icnt_context.flit_size)
            req_cmd = self.booksim2_module.create_data_cmd(src_id, dst_id, 0, n_flits, is_write=False, is_response=False)  # TODO: subnet index 
            self.booksim_send_packet_with_cmd(req_cmd)
            self.booksim_wait_cmd_and_handle(req_cmd)
            
            rsp_cmd = self.booksim2_module.create_data_cmd(dst_id, src_id, 0, n_flits, is_write=False, is_response=True)   # TODO: subnet index 
            self.booksim_send_packet_with_cmd(rsp_cmd)
            self.booksim_wait_cmd_and_handle(rsp_cmd)
        else:
            raise RuntimeError("[ERROR] BookSim2 is not enabled in this core.")
        
    @core_kernel_method
    def booksim_create_data_write_transaction(self, src_id: int, dst_id: int, size: int):
        if self.is_booksim2_enabled:    # TODO: subnet index 
            n_flits = math.ceil(size / self.icnt_context.flit_size)
            req_cmd = self.booksim2_module.create_data_cmd(src_id, dst_id, 0, n_flits, is_write=True, is_response=False)   # TODO: subnet index 
            self.booksim_send_packet_with_cmd(req_cmd)
            self.booksim_wait_cmd_and_handle(req_cmd)
            
            rsp_cmd = self.booksim2_module.create_data_cmd(dst_id, src_id, 0, n_flits, is_write=True, is_response=True)    # TODO: subnet index 
            self.booksim_send_packet_with_cmd(rsp_cmd)
            self.booksim_wait_cmd_and_handle(rsp_cmd)
        else:
            raise RuntimeError("[ERROR] BookSim2 is not enabled in this core.")
        
class BookSimCoreCycleModel(CoreCycleModel):
    def __init__(self, core: BookSimCore):
        super().__init__()
        
        self.core = core
        
    def booksim_wait_cmd_and_handle(self, cmd):
        return self.core.booksim2_module.get_cmd_wait_check_interval(cmd=cmd)