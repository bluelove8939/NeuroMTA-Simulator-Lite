from neuromta.common.core import *
from neuromta.common.parser_utils import parse_mem_cap_str

from neuromta.common.context import Context
from neuromta.common.synchronizer import TicketLock


__all__ = ['IcntNetworkContext', 'IcntNetworkCore', 'RouterCore']


class RouterCore(Core):
    def __init__(self, coord: tuple[int, int], icnt_context: 'IcntNetworkContext', **kwargs):
        super().__init__(**kwargs)
        
        self.coord = coord
        self.icnt_context = icnt_context
        self.icnt_context.register_core_as_node(self.coord, self)

        
class IcntNetworkContext(Context):
    def __init__(
        self,
        grid_shape: tuple[int, int] = (4, 4),
        flit_size: int = parse_mem_cap_str("32B"),
        control_packet_size: int = parse_mem_cap_str("64B"),
    ):
        super().__init__(n_owners_per_res=1)
        
        self._grid_shape = tuple(grid_shape)
        self._flit_size  = flit_size
        self._control_packet_size = control_packet_size

        self._icnt_core = None
        self._router_locks: dict[tuple[int, int], TicketLock] = {}
        
    ################################################
    # Interconnection Network Core Management
    ################################################
    
    def register_core_as_global_control(self, core: 'IcntNetworkCore'):
        self._icnt_core = core

    def register_core_as_node(self, coord: tuple[int, int], core: RouterCore):
        if coord[0] < 0 or coord[0] >= self._grid_shape[0]:
            raise Exception(f"[ERROR] X coordinate {coord[0]} is out of bounds for the interconnection network grid shape {self._grid_shape}.")
        if coord[1] < 0 or coord[1] >= self._grid_shape[1]:
            raise Exception(f"[ERROR] Y coordinate {coord[1]} is out of bounds for the interconnection network grid shape {self._grid_shape}.")
        if coord in self._router_locks:
            raise Exception(f"[ERROR] Coordinate {coord} is already registered in the interconnection network.")
        
        self.register_owner(coord, core)
        self._router_locks[coord] = TicketLock()
        
    def get_core_by_coord(self, coord: tuple[int, int]) -> RouterCore:
        if self.is_valid_resource_type(coord):
            raise Exception(f"[ERROR] Coordinate {coord} is not a valid resource type in the interconnection network.")
        
        return self.get_owner(coord)
    
    def get_router_lock(self, coord: tuple[int, int]) -> TicketLock:
        if coord[0] < 0 or coord[0] >= self._grid_shape[0]:
            raise Exception(f"[ERROR] X coordinate {coord[0]} is out of bounds for the interconnection network grid shape {self._grid_shape}.")
        if coord[1] < 0 or coord[1] >= self._grid_shape[1]:
            raise Exception(f"[ERROR] Y coordinate {coord[1]} is out of bounds for the interconnection network grid shape {self._grid_shape}.")
        
        return self._router_locks[coord]
    
    #################################################
    # Interconnection Network Latency
    #################################################
    
    def compute_hop_cnt(self, src_coord: tuple[int, int], dst_coord: tuple[int, int]) -> int:
        return abs(src_coord[0] - dst_coord[0]) + abs(src_coord[1] - dst_coord[1])
    
    def get_control_packet_latency(self, src_coord: tuple[int, int], dst_coord: tuple[int, int]) -> int:
        hop_cnt = self.compute_hop_cnt(src_coord, dst_coord)
        return hop_cnt + (self._control_packet_size // self._flit_size)
    
    def get_data_packet_latency(self, src_coord: tuple[int, int], dst_coord: tuple[int, int], data_size: int) -> int:
        hop_cnt = self.compute_hop_cnt(src_coord, dst_coord)
        return hop_cnt + (data_size // self._flit_size) + 1
    
    @property
    def icnt_core(self) -> 'IcntNetworkCore':
        return self._icnt_core
    
    
################################################
# Interconnect Network Core
################################################

class IcntNetworkCore(Core):
    def __init__(
        self,
        icnt_context: IcntNetworkContext,
    ):
        super().__init__(
            core_id=DEFAULT_CORE_ID, 
            cycle_model=IcntNetworkCoreCycleModel(self), 
            functional_model=IcntNetworkCoreFunctionalModel(self)
        )
        
        self.icnt_context = icnt_context
        self.icnt_context.register_core_as_global_control(self)
        
    @core_command_method
    def acquire_router_lock(self, coord: tuple[int, int]):
        pid = get_global_context()
        lock = self.icnt_context.get_router_lock(coord)

        if not lock.is_acquired_with(key=pid):
            lock.acquire(key=pid)
            
    @core_command_method
    def release_router_lock(self, coord: tuple[int, int]):
        pid = get_global_context()
        lock = self.icnt_context.get_router_lock(coord)
        
        if lock.is_locked_with(key=pid):
            lock.release(key=pid)
    
    @core_command_method
    def send_control_packet(self, src_coord: tuple[int, int], dst_coord: tuple[int, int]):
        pid = get_global_context()
        lock = self.icnt_context.get_router_lock(src_coord)
        
        if not lock.is_locked_with(key=pid):
            return False
        
    @core_command_method
    def send_data_packet(self, src_coord: tuple[int, int], dst_coord: tuple[int, int], data_size: int):
        pid = get_global_context()
        lock = self.icnt_context.get_router_lock(src_coord)
        
        if not lock.is_locked_with(key=pid):
            return False

    @core_kernel_method
    def request_data_transfer(self, consumer_coord: tuple[int, int], producer_coord: tuple[int, int], data_size: int):
        self.acquire_router_lock(consumer_coord)
        self.send_control_packet(consumer_coord, producer_coord)
        self.release_router_lock(consumer_coord)
        
        self.acquire_router_lock(producer_coord)
        self.send_data_packet(producer_coord, consumer_coord, data_size)
        self.release_router_lock(producer_coord)

class IcntNetworkCoreCycleModel(CoreCycleModel):
    def __init__(self, core: 'IcntNetworkCore'):
        super().__init__()
        
        self.core = core
        
    def send_control_packet(self, src_coord: tuple[int, int], dst_coord: tuple[int, int]):
        latency = self.core.icnt_context.get_control_packet_latency(src_coord, dst_coord)
        return latency
    
    def send_data_packet(self, src_coord: tuple[int, int], dst_coord: tuple[int, int], data_size: int):
        latency = self.core.icnt_context.get_data_packet_latency(src_coord, dst_coord, data_size)
        return latency
        
class IcntNetworkCoreFunctionalModel(CoreFunctionalModel):
    def __init__(self, core: 'IcntNetworkCore'):
        super().__init__()
        
        self.core = core