from neuromta.framework import *

from neuromta.hardware.context.mem_context import MemContext
from neuromta.hardware.context.cmap_context import CmapContext

__all__ = [
    "DMACore",
]


class DMACore(Core):
    def __init__(
        self, 
        coord: tuple[int, int],
        mem_context: MemContext, 
        cmap_context: CmapContext,
    ):
        super().__init__(
            core_id=coord, 
            cycle_model=DMACoreCycleModel(core=self)
        )
        
        self.coord = coord
        self.mem_context = mem_context
        self.cmap_context = cmap_context
        
        self.channel_id = self.cmap_context._coord_to_main_ch_id_mappings[self.coord]
    
    @core_kernel_method
    def mem_load_page_from_container(self, ptr: Pointer, container: DataContainer):
        msg = RPCMessage(
            msg_type=0,
            src_core_id=self.core_id,
            dst_core_id=self.cmap_context.main_mem_core_id,
            kernel_id=get_global_kernel_context().kernel_id,
            cmd_id="mem_load_page_from_container"
        ).with_args(
            ptr=ptr,
            container=container
        )
        
        self.async_rpc_send_req_msg(msg)
        self.async_rpc_wait_rsp_msg(msg)
        
    @core_kernel_method
    def mem_store_page_to_container(self, ptr: Pointer, container: DataContainer):
        msg = RPCMessage(
            msg_type=0,
            src_core_id=self.core_id,
            dst_core_id=self.cmap_context.main_mem_core_id,
            kernel_id=get_global_kernel_context().kernel_id,
            cmd_id="mem_store_page_to_container"
        ).with_args(
            ptr=ptr,
            container=container
        )
        
        self.async_rpc_send_req_msg(msg)
        self.async_rpc_wait_rsp_msg(msg)
        
class DMACoreCycleModel(CoreCycleModel):
    def __init__(self, core: DMACore):
        super().__init__()
        
        self.core = core