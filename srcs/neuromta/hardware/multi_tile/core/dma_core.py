from neuromta.framework import *

from neuromta.hardware.multi_tile.context.mem_context import MemContext
from neuromta.hardware.multi_tile.context.icnt_context import IcntContext


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
        
    #############################################################
    # Buffer Management
    #############################################################
        
    @core_command_method
    def main_memcopy_page(self, dst_ptr: Pointer, src_ptr: Pointer):
        if dst_ptr.ptr_type != PointerType.PAGE or src_ptr.ptr_type != PointerType.PAGE:
            raise ValueError("[ERROR] Memory copy requires page pointers.")

        if dst_ptr.size != src_ptr.size:
            raise ValueError(f"[ERROR] Page sizes do not match: {dst_ptr.size} != {src_ptr.size}")

        dst_elem: Page = self.mem_handle.get_data_element(dst_ptr)
        src_elem: Page = self.mem_handle.get_data_element(src_ptr)

        dst_elem.copy_from(src_elem)
        
    @core_kernel_method
    def main_memcopy_buffer(self, dst_ptr: BufferHandle, dst_offset_page_idx: int, src_ptr: BufferHandle, src_offset_page_idx: int, n_pages: int):
        dst_st = dst_offset_page_idx
        dst_ed = dst_st + n_pages
        
        src_st = src_offset_page_idx
        src_ed = src_st + n_pages
        
        dst_ptrs = dst_ptr[dst_st:dst_ed]
        src_ptrs = src_ptr[src_st:src_ed]

        for dst_ptr, src_ptr in zip(dst_ptrs, src_ptrs):
            self.main_memcopy_page(dst_ptr, src_ptr)
            
    #############################################################
    # Data Container (NoC Interface)
    #############################################################
    
    @core_command_method
    def mem_load_page_from_container(self, ptr: Pointer, container: DataContainer):
        if ptr.ptr_type != PointerType.PAGE:
            raise ValueError("[ERROR] Memory copy requires page pointer.")

        if not isinstance(container, DataContainer):
            raise ValueError("[ERROR] The source container must be a DataContainer instance.")

        page_elem: Page = self.mem_handle.get_data_element(ptr)
        page_elem.content = container.data

    @core_command_method
    def mem_store_page_to_container(self, ptr: Pointer, container: DataContainer):
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

    def main_memcopy_page(self, dst_ptr: Pointer, src_ptr: Pointer):
        return self.core.mem_context.main_config.get_cycles(size=src_ptr.size)
    
    @core_command_method
    def mem_load_page_from_container(self, ptr: Pointer, container: DataContainer):
        return self.core.mem_context.l1_config.get_cycles(size=ptr.size)

    @core_command_method
    def mem_store_page_to_container(self, ptr: Pointer, container: DataContainer):
        return self.core.mem_context.l1_config.get_cycles(size=ptr.size)