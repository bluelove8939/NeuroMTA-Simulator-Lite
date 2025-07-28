__all__ = [
    "CircularBufferHandle",
    "TicketLock",
]


class CircularBufferHandle:
    def __init__(self, entry_num: int):
        self._entry_num = entry_num
        
        self._full_sem  = self._entry_num
        self._rsvd_sem  = 0
        
    def check_vacancy(self, entry_num: int) -> bool:
        return self._full_sem >= entry_num
    
    def check_occupancy(self, entry_num: int) -> bool:
        return (self._entry_num - self._full_sem) >= entry_num
    
    def lock(self, entry_num: int=1):
        if self.is_locked:
            raise Exception("[ERROR] The circular buffer is already locked but trying to lock again. This is most likely a faulty implementation of the kernel.")
        
        self._rsvd_sem += entry_num
        
    def unlock(self, entry_num: int=1):
        if not self.is_locked:
            raise Exception("[ERROR] The circular buffer is not locked but trying to unlock. This is most likely a faulty implementation of the kernel.")
        
        self._rsvd_sem -= entry_num
        
    def push_back(self, entry_num: int=1):
        self._full_sem -= entry_num
        
    def pop_front(self, entry_num: int=1):
        self._full_sem += entry_num 
        
    @property
    def is_locked(self) -> bool:
        return self._rsvd_sem > 0
    

class TicketLock:
    def __init__(self, ticket_max: int = 2 ** 32 - 1):
        self._now_serving = 0
        self._next_ticket = 0
        self._ticket_max  = ticket_max

    def get_ticket(self):
        t = self._next_ticket
        self._increase_next_ticket()
        return t

    def is_locked_with(self, ticket: int) -> bool:
        return self._now_serving != ticket

    def unlock(self):
        self._increase_now_serving()

    def _increase_now_serving(self):
        self._now_serving += 1

        if self._now_serving > self._ticket_max:
            self._now_serving = 0
            
    def _increase_next_ticket(self):
        self._next_ticket += 1

        if self._next_ticket > self._ticket_max:
            self._next_ticket = 0