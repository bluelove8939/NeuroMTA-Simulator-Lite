from neuromta.framework import *

from neuromta.hardware.context.mem_context import MemContext
from neuromta.hardware.context.cmap_context import CmapContext, CmapCoreType

from neuromta.hardware.companions.dramsim import PYDRAMSIM3_AVAILABLE, DRAMSim3


__all__ = [
    "MainMemoryCore"
]


class MainMemoryCore(Core):
    def __init__(
        self,
        mem_context: MemContext, 
        cmap_context: CmapContext,
    ):
        super().__init__(
            core_id=cmap_context.main_mem_core_id, 
            cycle_model=MainMemoryCoreCycleModel(core=self)
        )
        
        self.mem_context = mem_context
        self.cmap_context = cmap_context
        
        self.mem_handle = MemoryHandle(
            mem_id=self.core_id,
            base_addr=self.cmap_context.config.main_mem_base_addr,
            size=self.cmap_context.config.main_mem_channel_size * self.cmap_context.config.n_main_mem_channels,
            n_channels=self.cmap_context.config.n_main_mem_channels
        )
        
        if self.mem_context.main_config.dramsim3_enable and PYDRAMSIM3_AVAILABLE:
            self.dramsim3_module = DRAMSim3(config=self.mem_context.main_config.dramsim3_config)
            self.register_companion_module("DRAMSIM3", self.dramsim3_module)
        else:
            self.dramsim3_module = None
            
    #############################################################
    # Data Container (NoC Interface)
    #############################################################
    
    @core_kernel_method
    def mem_load_page_from_container(self, ptr: Pointer, container: DataContainer):
        if self.is_dramsim3_enabled:
            msg = RPCMessage(
                msg_type=0,
                src_core_id=self.core_id,
                dst_core_id=self.cmap_context.main_mem_core_id,
                kernel_id=get_global_kernel_context().kernel_id,
                cmd_id="dramsim_mem_load_page",
            ).with_args(
                ptr=ptr
            )
            
            self.async_rpc_send_req_msg(msg)
            self.async_rpc_wait_rsp_msg(msg)

        self._static_mem_load_page_from_container(ptr, container)
        
    @core_kernel_method
    def mem_store_page_to_container(self, ptr: Pointer, container: DataContainer):
        if self.is_dramsim3_enabled:
            msg = RPCMessage(
                msg_type=0,
                src_core_id=self.core_id,
                dst_core_id=self.cmap_context.main_mem_core_id,
                kernel_id=get_global_kernel_context().kernel_id,
                cmd_id="dramsim_mem_store_page",
            ).with_args(
                ptr=ptr
            )
            
            self.async_rpc_send_req_msg(msg)
            self.async_rpc_wait_rsp_msg(msg)

        self._static_mem_store_page_to_container(ptr, container)

    @core_command_method
    def _static_mem_load_page_from_container(self, ptr: Pointer, container: DataContainer):
        if ptr.ptr_type != PointerType.PAGE:
            raise ValueError("[ERROR] Memory copy requires page pointer.")

        if not isinstance(container, DataContainer):
            raise ValueError("[ERROR] The source container must be a DataContainer instance.")

        page_elem: Page = self.mem_handle.get_data_element(ptr)
        page_elem.content = container.data

    @core_command_method
    def _static_mem_store_page_to_container(self, ptr: Pointer, container: DataContainer):
        if ptr.ptr_type != PointerType.PAGE:
            raise ValueError("[ERROR] Memory copy requires page pointer.")

        if not isinstance(container, DataContainer):
            raise ValueError("[ERROR] The target container must be a DataContainer instance.")

        page_elem: Page = self.mem_handle.get_data_element(ptr)
        container.data = page_elem.content.clone()  # Copy the content of the page element to the container

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
        
class MainMemoryCoreCycleModel(CoreCycleModel):
    def __init__(self, core: MainMemoryCore):
        super().__init__()
        
        self.core = core
        
    def _static_mem_load_page_from_container(self, ptr: Pointer, container: DataContainer):
        if self.core.is_dramsim3_enabled:
            return 1    # if DRAMSim is enabled, simulation time will be reflected at the behavioral model
        return self.core.mem_context.main_config.get_cycles(size=ptr.size)

    def _static_mem_store_page_to_container(self, ptr: Pointer, container: DataContainer):
        if self.core.is_dramsim3_enabled:
            return 1    # if DRAMSim is enabled, simulation time will be reflected at the behavioral model
        return self.core.mem_context.main_config.get_cycles(size=ptr.size)