import torch
from typing import Any, Sequence

from neuromta.common.synchronizer import TicketLock


__all__ = [
    "PageHandle",
    "BufferHandle",
    "CircularBufferHandle",
    "TemporaryBufferHandle",
]
    

class PageHandle:
    def __init__(self, page_size: int=None, content: torch.Tensor = None):
        if page_size is None and content is None:
            raise ValueError("Either page_size or content must be provided.")
        elif page_size is None:
            page_size = content.size() * content.itemsize
            content = content.reshape(-1).view(torch.uint8)
        elif content is None:
            content = torch.zeros(page_size, dtype=torch.uint8)
        
        self._page_size: int = page_size
        self._content: torch.Tensor = content
    
    def content_view(self, shape: tuple[int, ...], dtype: torch.dtype = torch.uint8) -> torch.Tensor:
        return self._content.view(dtype=dtype).reshape(shape)
    
    @property
    def content(self) -> torch.Tensor:
        return self._content
    
    @property
    def page_size(self) -> int:
        return self._page_size
    
    def __str__(self):
        return f"Page(size={self.page_size})"


class BufferHandle:
    def __init__(self, buffer_id: str, addr: int, page_size: int, n_pages: int, pages: Sequence[PageHandle] = None):
        self._buffer_id = buffer_id
        self._addr = addr
        self._page_size = page_size
        self._n_pages  = n_pages
        
        self._pages = [PageHandle(self._page_size) for _ in range(self._n_pages)]
        self._lock = TicketLock()
        
        if pages is not None:
            if not isinstance(pages, Sequence):
                raise Exception("[ERROR] Pages must be a sequence.")
            self.add_page_burst(0, pages)
            
    def parse_buffer_offset(self, buffer_offset: int) -> tuple[int, int]:
        if buffer_offset < 0 or buffer_offset >= self.size:
            raise ValueError(f"Address {buffer_offset} is out of bounds for buffer with size {self.size}.")
        
        page_idx = buffer_offset // self._page_size
        page_offset = buffer_offset % self._page_size
        return page_idx, page_offset
    
    def get_buffer_offset(self, page_idx: int, page_offset: int) -> int:
        if page_idx < 0 or page_idx >= self._n_pages:
            raise ValueError(f"Page index {page_idx} is out of bounds for buffer with {self._n_pages} pages.")
        if page_offset < 0 or page_offset >= self._page_size:
            raise ValueError(f"Page offset {page_offset} is out of bounds for page size {self._page_size}.")
        
        return page_idx * self._page_size + page_offset

    def add_page(self, page_idx: int, page: PageHandle):
        if not isinstance(page, PageHandle):
            page = PageHandle(page_size=self._page_size, content=page)
        
        if 0 <= page_idx < self._n_pages:
            self._pages[page_idx] = page
        else:
            raise IndexError(f"Page index {page_idx} out of range for buffer with {self._n_pages} pages.")
        
    def get_page(self, page_idx: int) -> PageHandle:
        if 0 <= page_idx < self._n_pages:
            return self._pages[page_idx]
        return None

    def add_page_burst(self, page_idx: int, pages: Sequence[PageHandle]):
        if page_idx + len(pages) > self._n_pages:
            raise Exception(f"[ERROR] Cannot set pages starting at index {page_idx} with {len(pages)} items, exceeds buffer size of {self._n_pages} pages.")

        for i in range(len(pages)):
            self.add_page(page_idx + i, pages[i])

    def get_page_burst(self, page_idx: int, n_pages: int) -> Sequence[PageHandle]:
        if 0 <= page_idx < self._n_pages and page_idx + n_pages <= self._n_pages:
            return self._pages[page_idx:page_idx + n_pages]
        return []

    @property
    def addr(self) -> int:
        return self._addr
    
    @property
    def page_size(self) -> int:
        return self._page_size
    
    @property
    def n_pages(self) -> int:
        return self._n_pages
    
    @property
    def size(self) -> int:  
        return self._page_size * self._n_pages
    
    @property
    def buffer_id(self) -> str:
        return self._buffer_id
    
    @property
    def lock(self) -> TicketLock:
        return self._lock
    
    def __str__(self):
        return f"Buffer({self.buffer_id})"


class CircularBufferHandle(BufferHandle):
    def __init__(self, buffer_id: str, addr: int, page_size: int, n_pages: int):
        super().__init__(buffer_id, addr, page_size, n_pages)
        
        self._rd_ptr    = 0
        self._wr_ptr    = 0
        self._rsvd_ptr  = 0
        
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

    def add_page(self, page_idx: int, page: PageHandle):
        page_idx = (self._wr_ptr + page_idx) % self._n_pages
        return super().add_page(page_idx, page)

    def get_page(self, page_idx: int) -> PageHandle:
        page_idx = (self._rd_ptr + page_idx) % self._n_pages
        return super().get_page(page_idx)

    def add_page_burst(self, page_idx: int, pages: Sequence[PageHandle]):
        for i, page in enumerate(pages):
            self.add_page(page_idx + i, page)

    def get_page_burst(self, page_idx: int, n_pages: int) -> Sequence[PageHandle]:
        result = []
        for i in range(n_pages):
            result.append(self.get_page(page_idx + i))
        return result
    
    def __str__(self):
        return f"CircularBuffer({self.buffer_id})"


class TemporaryBufferHandle(BufferHandle):
    def __init__(self, page_size: int, n_pages: int, pages: Sequence[Any] = None):
        super().__init__(
            buffer_id = "TEMP", 
            addr = -1, 
            page_size = page_size, 
            n_pages = n_pages, 
            pages = pages
        )
        
    def __str__(self):
        return f"Buffer(TEMP)"