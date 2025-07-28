from neuromta.common.core import *
from neuromta.hardware.synchronizer import CircularBufferHandle, TicketLock


__all__ = [
    "NPUGlobalController",
]


class NPUGlobalController(Core):
    def __init__(
        self,
        core_id: str,
        
        cb_sem_access_latency: int = 1,
        cb_access_latency_per_entry: int = 1,
    ):
        super().__init__(core_id=core_id)

        self.cb_sem_access_latency = cb_sem_access_latency
        self.cb_access_latency_per_entry = cb_access_latency_per_entry

        self.cb_handles: dict[str, CircularBufferHandle] = {}
        
        self._global_lock = TicketLock()
        self._global_tickets: dict[str, int] = {}
        
    # Mutual exclusiveness methods
    def is_locked_with(self, pid: str) -> bool:
        if pid not in self._global_tickets:
            return False
        
        return self._global_lock.is_locked_with(self._global_tickets[pid])
    
    @core_command_method(1)
    def acquire_global_lock(self):
        pid = get_global_context()
        if pid in self._global_tickets:
            raise Exception("[ERROR] Global lock is already acquired by the process. Please release it before acquiring again.")
        
        ticket = self._global_lock.get_ticket()
        self._global_tickets[pid] = ticket
        
    @core_command_method(1)
    def release_global_lock(self):
        pid = get_global_context()
        if pid not in self._global_tickets:
            raise Exception("[ERROR] Global lock is not acquired by the process. Please acquire it before releasing.")
        
        ticket = self._global_tickets[pid]
        if not self._global_lock.is_locked_with(ticket):
            raise Exception("[ERROR] The global lock is not locked by the core but trying to release. This is most likely a faulty implementation of the kernel.")
        
        self._global_lock.unlock()
        del self._global_tickets[pid]
    
    # Circular buffer methods    
    def _cb_access_latency(self, name: str, entry_num: int):
        return self.cb_access_latency_per_entry * entry_num

    def cb_create_buffer_handle(self, name: str, entry_num: int):
        if name in self.cb_handles:
            raise Exception(f"[ERROR] Circular buffer with name '{name}' already exists.")
        if entry_num <= 0:
            raise Exception("[ERROR] Entry number must be greater than 0.")
        
        self.cb_handles[name] = CircularBufferHandle(entry_num)

    def cb_remove_buffer_handle(self, name: str):
        if name not in self.cb_handles:
            raise Exception(f"[ERROR] Circular buffer with name '{name}' does not exist.")
        
        del self.cb_handles[name]
        
    @core_command_method("cb_sem_access_latency")
    def cb_reserve_back(self, name: str, entry_num: int):
        pid = get_global_context()
        if pid not in self._global_tickets:
            raise Exception("[ERROR] Global lock is not acquired. Please acquire it before reserving a circular buffer.")
        
        if name not in self.cb_handles:
            raise Exception(f"[ERROR] Circular buffer with name '{name}' does not exist.")
        
        if not self.is_locked_with(pid):
            return False
        
        handle = self.cb_handles[name]
        
        if not handle.check_vacancy(entry_num): return False
        if handle.is_locked: return False
        
        handle.lock(entry_num)
        
    @core_command_method("_cb_access_latency")
    def cb_push_back(self, name: str, entry_num: int):
        pid = get_global_context()
        if pid not in self._global_tickets:
            raise Exception("[ERROR] Global lock is not acquired. Please acquire it before pushing to a circular buffer.")
        
        if name not in self.cb_handles:
            raise Exception(f"[ERROR] Circular buffer with name '{name}' does not exist.")
        
        if not self.is_locked_with(pid):
            return False
        
        handle = self.cb_handles[name]
        handle.push_back(entry_num)
        handle.unlock(entry_num)
        
    @core_command_method("cb_sem_access_latency")
    def cb_wait_front(self, name: str, entry_num: int):
        pid = get_global_context()
        if pid not in self._global_tickets:
            raise Exception("[ERROR] Global lock is not acquired. Please acquire it before waiting for a circular buffer.")
        
        if name not in self.cb_handles:
            raise Exception(f"[ERROR] Circular buffer with name '{name}' does not exist.")
        
        if not self.is_locked_with(pid):
            return False
        
        handle = self.cb_handles[name]
        
        if not handle.check_occupancy(entry_num): return False
        if handle.is_locked: return False
        
        handle.lock(entry_num)

    @core_command_method("_cb_access_latency")
    def cb_pop_front(self, name: str, entry_num: int=1):
        pid = get_global_context()
        if pid not in self._global_tickets:
            raise Exception("[ERROR] Global lock is not acquired. Please acquire it before popping from a circular buffer.")
        
        if name not in self.cb_handles:
            raise Exception(f"[ERROR] Circular buffer with name '{name}' does not exist.")
        
        if not self.is_locked_with(pid):
            return False
        
        handle = self.cb_handles[name]        
        handle.pop_front(entry_num)
        handle.unlock(entry_num)