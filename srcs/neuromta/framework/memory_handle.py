import enum
import torch
from typing import Any


__all__ = [
    "VariableHandle",
    "PageHandle",
    "BufferHandle",
    "CircularBufferHandle",
    "MemoryHandle",
    "PointerType",
    "Pointer",
]


class VariableHandle:
    def __init__(self, addr: int, size: int=4, initial_value: int=0):
        self.addr = addr
        self.size = size
        self.value = initial_value

    def copy_from(self, var: 'VariableHandle'):
        self.value = var.value
        
    def __getstate__(self):
        return {"value": self.value}

    def __setstate__(self, state: dict):
        self.value = state["value"]

    def __str__(self):
        return f"Semaphore(value={self.value})"


class PageHandle:
    def __init__(self, addr: int, size: int):
        self.addr = addr
        self.size = size
        self._content: torch.Tensor = None

    def set_content(self, content: torch.Tensor, offset: int=0):
        if isinstance(content, torch.Tensor):
            if (content.numel() * content.element_size()) != self.size:
                raise ValueError(f"[ERROR] Content size {content.numel() * content.element_size()} does not match pointer size {self.size}.")
            if self.size % content.element_size() != 0:
                raise ValueError(f"[ERROR] Pointer size {self.size} is not a multiple of content item size {content.element_size()}.")
        else:
            raise TypeError(f"[ERROR] Content must be a torch.Tensor, but got {type(content).__name__}.")
        
        if offset == 0:
            self._content = content
        else:
            self._content = torch.zeros(self.size, dtype=torch.uint8)
            self._content[offset:offset + content.numel() * content.element_size()] = content.view(dtype=torch.uint8).flatten()
        
    def content_view(self, shape: tuple[int, ...], dtype: torch.dtype = torch.uint8) -> torch.Tensor:
        if self.content is None:
            return None
        if not isinstance(self.content, torch.Tensor):
            raise TypeError(f"[ERROR] Content is not a tensor, but a scalar value of type {type(self.content).__name__}.")
        return self.content.view(dtype=dtype).reshape(shape)
        
    def copy_from(self, page: 'PageHandle'):
        if not isinstance(page, PageHandle):
            raise Exception(f"[ERROR] Can only copy from another PageHandle, but got {type(page).__name__}.")
        
        if self.content is None or page.content is None:
            self._content = page.content
        else:
            dtype = self.content.dtype
            self._content.flatten()[:] = page.content_view(shape=(-1,), dtype=dtype)
            
    def __getstate__(self):
        return {
            "addr": self.addr, 
            "size": self.size, 
            "_content": self._content
        }
        
    def __setstate__(self, state: dict):
        self.addr = state["addr"]
        self.size = state["size"]
        self._content = state["_content"]
        
    @property
    def content(self) -> torch.Tensor:
        return self._content
    
    @property
    def is_empty(self) -> bool:
        return self._content is None
    
    def __str__(self):
        return f"PageHandle(addr={self.addr}, size={self.size})"
        
        
class BufferHandle:
    def __init__(self, page_size: int, pages: list[PageHandle]):
        self._page_size = page_size
        self._pages = pages
        self._n_pages = len(self._pages)

        for page in self._pages:
            if not isinstance(page, PageHandle):
                raise TypeError(f"[ERROR] Each page in the buffer must be a PointerHandle, but got {type(page).__name__}.")
            if page.size != self._page_size:
                raise ValueError(f"[ERROR] Each page in the buffer must have size {self._page_size}, but got {page.size}.")

    def get_page_handle(self, page_idx: int) -> PageHandle:
        if self._pages is None:
            raise Exception(f"[ERROR] Cannot get page handles because pages are not initialized.")
        if page_idx < 0 or page_idx >= self._n_pages:
            raise IndexError(f"[ERROR] Page index {page_idx} out of range for buffer with {self._n_pages} pages.")
        
        return self._pages[page_idx]
    
    def set_content(self, content: torch.Tensor, offset: int=0):
        if not isinstance(content, torch.Tensor):
            raise TypeError(f"[ERROR] Content must be a torch.Tensor, but got {type(content).__name__}.")
        
        raw_content = self.content
        raw_content[offset:offset + content.numel() * content.element_size()] = content.view(dtype=torch.uint8).flatten()
        raw_content = raw_content.reshape((self.n_pages, self.page_size))

        for i in range(self._n_pages):
            self._pages[i].set_content(raw_content[i, :])
    
    def content_view(self, shape: tuple[int, ...], dtype: torch.dtype = torch.uint8) -> torch.Tensor:
        return self.content.view(dtype=dtype).reshape(shape)
    
    def __getstate__(self):
        return {
            "page_size": self._page_size,
            "pages": self._pages,
            "n_pages": self._n_pages
        }

    def __setstate__(self, state: dict):
        self._page_size = state["page_size"]
        self._pages = state["pages"]
        self._n_pages = state["n_pages"]
    
    @property
    def content(self) -> torch.Tensor:
        content_list = []
        
        for i in range(self.n_pages):
            page = self.get_page_handle(i)
                
            if not page.is_empty:
                content_list.append(page.content_view(shape=(-1,), dtype=torch.uint8))
            else:
                content_list.append(torch.zeros((self._page_size,), dtype=torch.uint8))
        
        content = torch.concatenate(content_list)
        
        return content
    
    @property
    def pages(self) -> list[PageHandle]:
        return self._pages
    
    @property
    def page_size(self) -> int:
        return self._page_size
    
    @property
    def n_pages(self) -> int:
        return self._n_pages
    
    def __str__(self) -> str:
        return f"BufferHandle(page_size={self._page_size}, n_pages={self._n_pages})"
    
    
class CircularBufferHandle(BufferHandle):
    def __init__(self, page_size: int, pages: list[PageHandle]):
        super().__init__(page_size=page_size, pages=pages,)
        
        self._rd_ptr    = 0
        self._wr_ptr    = 0
        self._rsvd_ptr  = 0
        
    def __getstate__(self):
        return super().__getstate__() | {
            "rd_ptr": self._rd_ptr,
            "wr_ptr": self._wr_ptr,
            "rsvd_ptr": self._rsvd_ptr
        }

    def __setstate__(self, state: dict):
        self._rd_ptr = state["rd_ptr"]
        self._wr_ptr = state["wr_ptr"]
        self._rsvd_ptr = state["rsvd_ptr"]
        
        return super().__setstate__(state)

    @property
    def _alloc_space(self) -> int:
        if self._rsvd_ptr >= self._rd_ptr:
            return self._rsvd_ptr - self._rd_ptr
        else:
            return self._n_pages - (self._rd_ptr - self._rsvd_ptr)
        
    @property
    def _real_space(self) -> int:
        if self._wr_ptr >= self._rd_ptr:
            return self._wr_ptr - self._rd_ptr
        else:
            return self._n_pages - (self._rd_ptr - self._wr_ptr)
        
    def check_vacancy(self, n_pages: int) -> bool:
        return self._alloc_space + n_pages <= self._n_pages
    
    def check_occupancy(self, n_pages: int) -> bool:
        return self._real_space >= n_pages
    
    def allocate_cb_space(self, n_pages: int):
        self._rsvd_ptr = (self._rsvd_ptr + n_pages) % self._n_pages
        
    def occupy_cb_space(self, n_pages: int):
        self._wr_ptr = (self._wr_ptr + n_pages) % self._n_pages
        
    def deallocate_cb_space(self, n_pages: int):
        self._rd_ptr = (self._rd_ptr + n_pages) % self._n_pages
    
    def get_page_handle(self, page_idx):    # override
        rel_page_idx = page_idx + self._rd_ptr
        
        if rel_page_idx >= self.n_pages:
            rel_page_idx -= self.n_pages

        return super().get_page_handle(rel_page_idx)


class MemoryHandle:
    def __init__(self, mem_id: str, base_addr: int, size: int):
        self._mem_id = mem_id
        self._base_addr = base_addr
        self._size = size
        
        self._data_elements: dict[int, PageHandle | VariableHandle] = {}
        
    def __getstate__(self):
        return {
            "_mem_id": self._mem_id,
            "_base_addr": self._base_addr,
            "_size": self._size,
            "_pages": self._data_elements
        }

    def __setstate__(self, state: dict):
        self._mem_id = state["_mem_id"]
        self._base_addr = state["_base_addr"]
        self._size = state["_size"]

        state_pages: dict[str, PageHandle | VariableHandle] = state["_pages"]

        for key in state_pages.keys():
            if key not in self._data_elements.keys():
                self._data_elements[int(key)] = state_pages[key]
            else:
                self._data_elements[int(key)].copy_from(state_pages[key])

    def get_overlapping_data_addr(self, addr: int, size: int=1) -> int: # returns False if the address space is not overlapping with any memory handles
        keys = sorted(self._data_elements.keys())
        left, right = 0, len(keys) - 1

        while left <= right:
            mid = (left + right) // 2
            mid_addr = keys[mid]
            page = self._data_elements[mid_addr]

            if not (addr + size <= page.addr or addr >= page.addr + page.size):
                return mid_addr
            if addr < page.addr:
                right = mid - 1
            else:
                left = mid + 1
                
        return -1

    def create_page_handle(self, addr: int, page_size: int) -> PageHandle:
        if addr < self.base_addr or (addr + page_size) > (self.base_addr + self.size):
            raise ValueError(f"[ERROR] Address {addr} with page size {page_size} is out of bounds for memory handle with base address {self._base_addr} and size {self._size}.")
        if addr in self._data_elements.keys():
            raise Exception(f"[ERROR] Page at address {addr} already exists in memory handle with base address {self._base_addr}.")
        
        page = PageHandle(addr=addr, size=page_size)
        self._data_elements[addr] = page
        
        return page
    
    def remove_page_handle(self, addr: int):
        if isinstance(addr, PageHandle):
            addr = addr.addr
        if addr not in self._data_elements:
            raise KeyError(f"[ERROR] No page found at address {addr} in memory handle with base address {self._base_addr}.")
        
        del self._data_elements[addr]
        
    def allocate_page_handles(self, page_size: int, n_pages: int=1) -> list[PageHandle]:
        allocated_pages: list[PageHandle] = []
        
        for i in range(self.size // page_size):
            if len(allocated_pages) >= n_pages:
                break
            
            addr = self.base_addr + i * page_size
            overlap = self.get_overlapping_data_addr(addr, size=page_size)
            
            if overlap < 0:
                allocated_pages.append(self.create_page_handle(addr, page_size))
            else:
                continue
        
        return allocated_pages
    
    def deallocate_page_handles(self, pages: list[PageHandle | int]):
        for page in pages:
            addr = page.addr if isinstance(page, PageHandle) else page
            self.remove_page_handle(addr=addr)
            
    def get_page_handle(self, addr: int) -> PageHandle:
        overlapping_addr = self.get_overlapping_data_addr(addr, 1)

        if overlapping_addr < 0:
            raise KeyError(f"[ERROR] No page found at address {addr} in memory handle with base address {self._base_addr}.")

        return self._data_elements[overlapping_addr]
            
    def create_variable_handle(self, addr: int, initial_value: int=0) -> VariableHandle:
        if addr < self.base_addr or (addr + 4) > (self.base_addr + self.size):
            raise ValueError(f"[ERROR] Address {addr} is out of bounds for memory handle with base address {self._base_addr} and size {self._size}.")
        if addr in self._data_elements.keys():
            raise Exception(f"[ERROR] Variable at address {addr} already exists in memory handle with base address {self._base_addr}.")

        var = VariableHandle(initial_value)
        self._data_elements[addr] = var

        return var
    
    def remove_variable_handle(self, addr: int):
        if isinstance(addr, VariableHandle):
            addr = addr.addr
        if addr not in self._data_elements:
            raise KeyError(f"[ERROR] No variable found at address {addr} in memory handle with base address {self._base_addr}.")
        del self._data_elements[addr]

    def allocate_variable_handle(self, initial_value: int=0) -> VariableHandle:
        for i in range(self.size // 4):
            addr = self.base_addr + i * 4
            overlap = self.get_overlapping_data_addr(addr, size=4)

            if overlap < 0:
                var = self.create_variable_handle(addr, initial_value)
                return var
            else:
                continue

    def deallocate_variable_handle(self, var: VariableHandle | int):
        addr = var.addr if isinstance(var, VariableHandle) else var
        self.remove_variable_handle(addr=addr)

    def get_variable_handle(self, addr: int) -> VariableHandle:
        overlapping_addr = self.get_overlapping_data_addr(addr, 4)

        if overlapping_addr < 0:
            raise KeyError(f"[ERROR] No variable found at address {addr} in memory handle with base address {self._base_addr}.")

        return self._data_elements[overlapping_addr]
    
    def clear(self):
        self._data_elements.clear()

    @property
    def mem_id(self) -> str:
        return self._mem_id

    @property
    def base_addr(self) -> int:
        return self._base_addr
    
    @property
    def size(self) -> int:
        return self._size
    
    def __str__(self) -> str:
        return f"MemoryHandle(mem_id={self._mem_id}, base_addr={self._base_addr}, size={self._size})"


class PointerType(enum.Enum):
    UNDEFINED   = enum.auto()
    PAGE        = enum.auto()
    BUFFER      = enum.auto()
    VARIABLE    = enum.auto()
    
    @classmethod
    def get_pointer_type_with_handle(cls, handle: Any) -> 'PointerType':
        if isinstance(handle, VariableHandle):
            return cls.VARIABLE
        elif isinstance(handle, BufferHandle):
            return cls.BUFFER
        elif isinstance(handle, PageHandle):
            return cls.PAGE
        return cls.UNDEFINED


class Pointer:
    def __init__(self, ptr_id: int | str, item: Any=None):
        self._ptr_id    = ptr_id
        self._ptr_type  = PointerType.get_pointer_type_with_handle(item)
        self._item      = item
        
    def copy_from(self, other: 'Pointer'):
        if not isinstance(other, Pointer):
            raise TypeError(f"[ERROR] Can only copy from another Pointer, but got {type(other).__name__}.")
        if self._ptr_type != other._ptr_type:
            raise Exception(f"[ERROR] Pointer types do not match: {self._ptr_type.name} != {other._ptr_type.name}.")
        
        if self._ptr_type == PointerType.VARIABLE:
            self.var.value = other.var.value
        elif self._ptr_type in (PointerType.BUFFER, PointerType.PAGE):
            self.handle.set_content(other.handle.content, offset=0)

    def clear(self):
        self._item = None
        self._ptr_type = PointerType.UNDEFINED

    def __getitem__(self, item):
        if self._ptr_type == PointerType.BUFFER:
            buffer: BufferHandle = self.handle
            if isinstance(item, int):
                return Pointer(ptr_id=f"{self.ptr_id}[{item}]", item=buffer.get_page_handle(item))
            elif isinstance(item, slice):
                return [Pointer(ptr_id=f"{self.ptr_id}[{i}]", item=buffer.get_page_handle(i)) for i in range(item.start, item.stop)]
            elif isinstance(item, tuple):
                return [Pointer(ptr_id=f"{self.ptr_id}[{i}]", item=buffer.get_page_handle(i)) for i in item]
        return super().__getitem__(item)

    def __getstate__(self):
        return {
            "ptr_id": self._ptr_id,
            "ptr_type": self._ptr_type,
            "item": self._item
        }

    def __setstate__(self, state: dict):
        self._ptr_id = state["ptr_id"]
        self._ptr_type = state["ptr_type"]
        self._item = state["item"]

    def __hash__(self):
        return hash((self._ptr_id, self._ptr_type))
    
    def __eq__(self, value: 'Pointer'):
        if not isinstance(value, Pointer):
            return False
        return (self._ptr_id, self._ptr_type) == (value.ptr_id, value.ptr_type)

    def __ne__(self, value):
        return not self.__eq__(value)

    @property
    def ptr_id(self):
        return self._ptr_id

    @property
    def ptr_type(self):
        return self._ptr_type

    @property
    def handle(self) -> BufferHandle | PageHandle:
        if self.ptr_type not in (PointerType.BUFFER, PointerType.PAGE):
            raise Exception(f"[ERROR] Pointer type {self.ptr_type.name} does not have a buffer handle.")
        return self._item
    
    @handle.setter
    def handle(self, value: BufferHandle | PageHandle):
        if not isinstance(value, (BufferHandle, PageHandle)):
            raise TypeError(f"[ERROR] Pointer handle must be a BufferHandle or PageHandle, but got {type(value).__name__}.")
        
        self._item = value
        self._ptr_type = PointerType.get_pointer_type_with_handle(value)

    @property
    def var(self) -> VariableHandle:
        if self.ptr_type != PointerType.VARIABLE:
            raise Exception(f"[ERROR] Pointer type {self.ptr_type.name} does not have a variable handle.")

        return self._item
    
    @var.setter
    def var(self, value: VariableHandle):
        if not isinstance(value, VariableHandle):
            raise Exception(f"[ERROR] Pointer of the variable must be a VariableHandle, but got {type(value).__name__}.")
        
        self._item = value
        self._ptr_type = PointerType.get_pointer_type_with_handle(value)

    def __str__(self):
        return f"Pointer(id={self._ptr_id}, type={self._ptr_type.name})"
