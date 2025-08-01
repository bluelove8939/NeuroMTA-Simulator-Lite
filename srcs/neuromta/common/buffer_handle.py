from typing import Any, Sequence
from neuromta.common.synchronizer import TicketLock


__all__ = [
    "PageHandle",
    "BufferHandle",
    "CircularBufferHandle",
    "TemporaryBufferHandle",
]


class PageHandle:
    def __init__(self, page_size: int, content: Any):
        self.page_size = page_size
        self.content = content
        
    def __str__(self):
        return f"Page(size={self.page_size}, content={self.content})"


class BufferHandle:
    def __init__(self, buffer_id: str, addr: int, page_size: int, n_pages: int, pages: Sequence[PageHandle] = None):
        self._buffer_id = buffer_id
        self._addr = addr
        self._page_size = page_size
        self._n_pages  = n_pages
        
        self._pages = [PageHandle(self._page_size, None) for _ in range(self._n_pages)]
        self._lock = TicketLock()
        
        if pages is not None:
            if not isinstance(pages, Sequence):
                raise Exception("[ERROR] Pages must be a sequence.")
            self.data_set_page_burst(0, pages)

    def data_set_page(self, page_idx: int, page: PageHandle):
        if not isinstance(page, PageHandle):
            page = PageHandle(self._page_size, page)
        
        if 0 <= page_idx < self._n_pages:
            self._pages[page_idx] = page
        else:
            raise IndexError(f"Page index {page_idx} out of range for buffer with {self._n_pages} pages.")
        
    def data_get_page(self, page_idx: int) -> PageHandle:
        if 0 <= page_idx < self._n_pages:
            return self._pages[page_idx]
        return None

    def data_set_page_burst(self, page_idx: int, pages: Sequence[PageHandle]):
        if page_idx + len(pages) > self._n_pages:
            raise Exception(f"[ERROR] Cannot set pages starting at index {page_idx} with {len(pages)} items, exceeds buffer size of {self._n_pages} pages.")

        for i in range(len(pages)):
            self.data_set_page(page_idx + i, pages[i])

    def data_get_page_burst(self, page_idx: int, n_pages: int) -> Sequence[PageHandle]:
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

    def data_set_page(self, page_idx: int, page: PageHandle):
        page_idx = (self._wr_ptr + page_idx) % self._n_pages
        return super().data_set_page(page_idx, page)

    def data_get_page(self, page_idx: int) -> PageHandle:
        page_idx = (self._rd_ptr + page_idx) % self._n_pages
        return super().data_get_page(page_idx)

    def data_set_page_burst(self, page_idx: int, pages: Sequence[PageHandle]):
        for i, page in enumerate(pages):
            self.data_set_page(page_idx + i, page)

    def data_get_page_burst(self, page_idx: int, n_pages: int) -> Sequence[PageHandle]:
        result = []
        for i in range(n_pages):
            result.append(self.data_get_page(page_idx + i))
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