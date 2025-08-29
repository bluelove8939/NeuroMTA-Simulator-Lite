import math

from neuromta.framework import *

from neuromta.hardware.multi_tile.context.cmap_context import CmapContext
from neuromta.hardware.multi_tile.context.icnt_context import IcntContext

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
        
    @property
    def is_booksim2_enabled(self) -> bool:
        return PYBOOKSIM2_AVAILABLE and self.icnt_context.booksim2_enable and self.check_rpc_inbox(self.cmap_context.booksim_core_id)

    #############################################################
    # NoC Commands
    #############################################################

    @core_kernel_method
    def noc_create_data_read_transaction(self, src_coord: tuple[int, int], dst_coord: tuple[int, int], data_size: int):
        if self.is_booksim2_enabled:
            src_id = self.cmap_context.get_node_id_from_coord(src_coord)
            dst_id = self.cmap_context.get_node_id_from_coord(dst_coord)
            
            msg = RPCMessage(
                msg_type=0,
                src_core_id=self.core_id,
                dst_core_id=self.cmap_context.booksim_core_id,
                kernel_id=get_global_kernel_context().kernel_id,
                cmd_id="booksim_create_data_read_transaction"
            ).with_args(
                src_id=src_id,
                dst_id=dst_id,
                size=data_size,
            )
            self.async_rpc_send_req_msg(msg)
            self.async_rpc_wait_rsp_msg(msg)
        else:
            self._static_noc_create_data_read_transaction(src_coord, dst_coord, data_size)
            
    @core_kernel_method
    def noc_create_data_write_transaction(self, src_coord: tuple[int, int], dst_coord: tuple[int, int], data_size: int):
        if self.is_booksim2_enabled:
            src_id = self.cmap_context.get_node_id_from_coord(src_coord)
            dst_id = self.cmap_context.get_node_id_from_coord(dst_coord)
            # self.booksim_create_data_write_transaction(src_id, dst_id, data_size)
            
            msg = RPCMessage(
                msg_type=0,
                src_core_id=self.core_id,
                dst_core_id=self.cmap_context.booksim_core_id,
                kernel_id=get_global_kernel_context().kernel_id,
                cmd_id="booksim_create_data_write_transaction"
            ).with_args(
                src_id=src_id,
                dst_id=dst_id,
                size=data_size,
            )
            self.async_rpc_send_req_msg(msg)
            self.async_rpc_wait_rsp_msg(msg)
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
        
    # def booksim_wait_cmd_and_handle(self, cmd):
    #     return self.core.booksim2_module.get_cmd_wait_check_interval(cmd=cmd)

    def _static_noc_create_data_read_transaction(self, src_coord: tuple[int, int], dst_coord: tuple[int, int], data_size: int):
        return self.core.icnt_context.get_data_packet_latency(src_coord, dst_coord, data_size)

    def _static_noc_create_data_write_transaction(self, src_coord: tuple[int, int], dst_coord: tuple[int, int], data_size: int):
        return self.core.icnt_context.get_data_packet_latency(src_coord, dst_coord, data_size)