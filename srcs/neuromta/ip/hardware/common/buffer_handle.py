from typing import Any, Sequence
from neuromta.ip.hardware.common.synchronizer import TicketLock


__all__ = [
    "BufferHandle",
    "CircularBufferHandle",
    "TemporaryBufferHandle",
]


class BufferHandle:
    def __init__(self, buffer_id: str, addr: int, page_size: int, n_pages: int, pages: Sequence[Any] = None):
        self._buffer_id = buffer_id
        self._addr = addr
        self._page_size = page_size
        self._n_pages  = n_pages
        
        self._pages = [None for _ in range(self._n_pages)]
        self._lock = TicketLock()
        
        if pages is not None:
            if not isinstance(pages, Sequence):
                raise Exception("[ERROR] Pages must be a sequence.")
            self.data_set_page_burst(0, pages)

    def data_set_page(self, page_idx: int, data: Any):
        if 0 <= page_idx < self._n_pages:
            self._pages[page_idx] = data
        else:
            raise IndexError(f"Page index {page_idx} out of range for buffer with {self._n_pages} pages.")
        
    def data_get_page(self, page_idx: int) -> Any:
        if 0 <= page_idx < self._n_pages:
            return self._pages[page_idx]
        return None
    
    def data_set_page_burst(self, page_idx: int, data: Sequence[Any]):
        if page_idx + len(data) > self._n_pages:
            raise Exception(f"[ERROR] Cannot set pages starting at index {page_idx} with {len(data)} items, exceeds buffer size of {self._n_pages} pages.")
        
        for i in range(len(data)):
            self._pages[page_idx + i] = data[i]
            
    def data_get_page_burst(self, page_idx: int, n_pages: int) -> Sequence[Any]:
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
        
        self._alloc_space = 0
        self._real_space = 0
        
        self._rd_ptr = 0
        self._wr_ptr = 0
        
    def check_vacancy(self, n_pages: int) -> bool:
        return self._alloc_space + n_pages <= self._n_pages
    
    def check_occupancy(self, n_pages: int) -> bool:
        return self._real_space >= n_pages
        
    def allocate_cb_space(self, n_pages: int):
        self._alloc_space += n_pages
        
    def occupy_cb_space(self, n_pages: int):
        self._real_space += n_pages
        self._wr_ptr = (self._wr_ptr + 1) % self._n_pages
        
    def evacuate_cb_space(self, n_pages: int):
        self._real_space -= n_pages
        
    def deallocate_cb_space(self, n_pages: int):
        self._alloc_space -= n_pages
        self._rd_ptr = (self._rd_ptr + 1) % self._n_pages
    
    def data_set_page(self, page_idx: int, data: Any):
        page_idx = (self._wr_ptr + page_idx) % self._n_pages
        return super().data_set_page(page_idx, data)
        
    def data_get_page(self, page_idx: int) -> Any:
        page_idx = (self._rd_ptr + page_idx) % self._n_pages
        return super().data_get_page(page_idx)
    
    def data_set_page_burst(self, page_idx: int, data: Sequence[Any]):
        for i, d in enumerate(data):
            self.data_set_page(page_idx + i, d)
            
    def data_get_page_burst(self, page_idx: int, n_pages: int) -> Sequence[Any]:
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