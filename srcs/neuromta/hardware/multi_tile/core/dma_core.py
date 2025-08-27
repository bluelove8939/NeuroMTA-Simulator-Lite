from neuromta.framework import *

from neuromta.hardware.multi_tile.context.mem_context import MemContext
from neuromta.hardware.multi_tile.context.icnt_context import IcntContext

from neuromta.hardware.companions.dramsim import PYDRAMSIM3_AVAILABLE, DRAMSim3

__all__ = [
    "DMACore",
]


class DMACore(Core):
    def __init__(
        self, 
        coord: tuple[int, int],
        mem_context: MemContext, 
        icnt_context: IcntContext,
    ):
        super().__init__(
            core_id=coord, 
            cycle_model=DMACoreCycleModel(core=self)
        )
        
        self.coord = coord
        self.mem_context = mem_context
        self.icnt_context = icnt_context
        
        self.mem_handle = MemoryHandle(
            mem_id=self.coord.__str__(),
            base_addr=self.icnt_context.get_base_addr_from_coord(self.coord),
            size=self.icnt_context._main_mem_bank_size
        )
        
        if self.mem_context.main_config.dramsim3_enable and not PYDRAMSIM3_AVAILABLE:
            raise RuntimeError("[ERROR] DRAMSim3 is not available. Please install PyDRAMSim3 to enable DRAMSim3 support.")
        
        if self.is_dramsim3_enabled:
            dramsim3_module = DRAMSim3(config_name=self.mem_context.main_config.dramsim3_config_name)
            self.register_companion_module("DRAMSIM3", dramsim3_module)

    @property
    def is_dramsim3_enabled(self) -> bool:
        return PYDRAMSIM3_AVAILABLE and self.mem_context.main_config.dramsim3_enable
            
    #############################################################
    # Data Container (NoC Interface)
    #############################################################
    
    @core_kernel_method
    def mem_load_page_from_container(self, ptr: Pointer, container: DataContainer):
        if self.is_dramsim3_enabled:
            module = self.get_companion_module("DRAMSIM3")
            cmd = module.create_cmd(addr=ptr.addr, size=ptr.size, is_write=True)
            self.companion_send_cmd(module, cmd)
            self.companion_wait_cmd_executed(module, cmd)

        self._static_mem_load_page_from_container(ptr, container)
        
    @core_kernel_method
    def mem_store_page_to_container(self, ptr: Pointer, container: DataContainer):
        if self.is_dramsim3_enabled:
            module = self.get_companion_module("DRAMSIM3")
            cmd = module.create_cmd(addr=ptr.addr, size=ptr.size, is_write=False)
            self.companion_send_cmd("DRAMSIM3", cmd)
            self.companion_wait_cmd_executed("DRAMSIM3", cmd)
            
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
        
class DMACoreCycleModel(CoreCycleModel):
    def __init__(self, core: DMACore):
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