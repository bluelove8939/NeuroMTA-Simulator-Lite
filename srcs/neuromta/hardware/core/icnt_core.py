import math

from neuromta.framework import *

from neuromta.hardware.context.cmap_context import CmapContext
from neuromta.hardware.context.icnt_context import IcntContext

from neuromta.hardware.companions.booksim import PYBOOKSIM2_AVAILABLE, BookSim2


__all__ = [
    "IcntCore",
]


class IcntCore(Core):
    def __init__(
        self,
        icnt_context: IcntContext,
        cmap_context: CmapContext,
    ):
        super().__init__(
            core_id=cmap_context.icnt_core_id,
            cycle_model=IcntCoreCycleModel(core=self)
        )
        
        self.icnt_context = icnt_context
        self.cmap_context = cmap_context
        
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
    
    #############################################################
    # NoC Commands
    #############################################################

    @core_kernel_method
    def noc_create_data_read_transaction(self, src_coord: tuple[int, int], dst_coord: tuple[int, int], data_size: int):
        if self.is_booksim2_enabled:
            src_id = self.cmap_context.get_node_id_from_coord(src_coord)
            dst_id = self.cmap_context.get_node_id_from_coord(dst_coord)
            self.booksim_create_data_read_transaction(src_id, dst_id, data_size)
        else:
            self._static_noc_create_data_read_transaction(src_coord, dst_coord, data_size)
            
    @core_kernel_method
    def noc_create_data_write_transaction(self, src_coord: tuple[int, int], dst_coord: tuple[int, int], data_size: int):
        if self.is_booksim2_enabled:
            src_id = self.cmap_context.get_node_id_from_coord(src_coord)
            dst_id = self.cmap_context.get_node_id_from_coord(dst_coord)
            self.booksim_create_data_write_transaction(src_id, dst_id, data_size)
        else:
            self._static_noc_create_data_write_transaction(src_coord, dst_coord, data_size)

    @core_command_method
    def _static_noc_create_data_read_transaction(self, src_coord: tuple[int, int], dst_coord: tuple[int, int], data_size: int):
        pass

    @core_command_method
    def _static_noc_create_data_write_transaction(self, src_coord: tuple[int, int], dst_coord: tuple[int, int], data_size: int):
        pass

class IcntCoreCycleModel(CoreCycleModel):
    def __init__(self, core: IcntCore):
        super().__init__()
        
        self.core = core
    
    def booksim_wait_cmd_and_handle(self, cmd):
        return self.core.booksim2_module.get_cmd_wait_check_interval(cmd=cmd)

    def _static_noc_create_data_read_transaction(self, src_coord: tuple[int, int], dst_coord: tuple[int, int], data_size: int):
        return self.core.icnt_context.get_data_packet_latency(src_coord, dst_coord, data_size)

    def _static_noc_create_data_write_transaction(self, src_coord: tuple[int, int], dst_coord: tuple[int, int], data_size: int):
        return self.core.icnt_context.get_data_packet_latency(src_coord, dst_coord, data_size)