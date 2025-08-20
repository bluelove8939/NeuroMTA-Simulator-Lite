from neuromta.framework import *

from neuromta.hardware.multi_tile.context.mem_context import MemContext
from neuromta.hardware.multi_tile.context.icnt_context import IcntContext


__all__ = [
    "IcntCore",
]


class IcntCore(Core):
    def __init__(
        self,
        icnt_context: IcntContext,
    ):
        super().__init__(
            core_id=icnt_context.icnt_core_id,
            cycle_model=IcntCoreCycleModel(core=self)
        )
        
        self.icnt_context = icnt_context
    
    @core_command_method
    def noc_send_control_packet(self, src_coord: tuple[int, int], dst_coord: tuple[int, int]):
        pass    # TODO: the actual implementation of the interconnect network will be replaced by the BookSim2

    @core_command_method
    def noc_send_data_packet(self, src_coord: tuple[int, int], dst_coord: tuple[int, int], data_size: int):
        pass    # TODO: the actual implementation of the interconnect network will be replaced by the BookSim2

class IcntCoreCycleModel(CoreCycleModel):
    def __init__(self, core: IcntCore):
        super().__init__()
        
        self.core = core

    def noc_send_control_packet(self, src_coord: tuple[int, int], dst_coord: tuple[int, int]):
        return self.core.icnt_context.get_control_packet_latency(src_coord, dst_coord)
    
    def noc_send_data_packet(self, src_coord: tuple[int, int], dst_coord: tuple[int, int], data_size: int):
        return self.core.icnt_context.get_data_packet_latency(src_coord, dst_coord, data_size)