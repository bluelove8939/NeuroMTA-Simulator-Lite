import copy
import math
from typing import Sequence

from neuromta.simulation.common import *
from neuromta.simulation.descriptor import *


__all__ = [
    "CoreCluster",
    "MappingStage", 
    "Mapping",
    "Profiler",
]


class CoreCluster:
    def __init__(
        self, 
        word_size:      int             = 1,
        ofm_word_size:  int             = 4,
        m_tile:         int             = 32,
        n_tile:         int             = 32,
        k_tile:         int             = 32,
        dataflow:       SACoreDataflow  = SACoreDataflow.OS,
        
        onc_mem_cap:    int             = parse_mem_cap_str("32MB"),
        onc_mem_type:   MemoryType      = MemoryType.L2,
        
        shape:          Sequence[int]   = (2, 2),
        spatial_reuse:  bool            = False,
    ):
        # Core Configs
        self.word_size      = word_size
        self.ofm_word_size  = ofm_word_size
        self.m_tile         = m_tile
        self.n_tile         = n_tile
        self.k_tile         = k_tile
        self.dataflow       = dataflow
        
        # On-chip Memory Configs
        self.onc_mem_cap    = onc_mem_cap
        self.onc_mem_type   = onc_mem_type
        
        # Cluster Configs (NoC)
        self.shape          = shape
        self.spatial_reuse  = spatial_reuse  # use NoC if True ...
            
        if len(self.shape) != 2:
            raise Exception(f"[ERROR] The shape of the core cluster should be 2D, not {len(self.shape)}D")
        
        if self.dataflow == SACoreDataflow.WS and self.spatial_reuse:
            raise Exception(f"[ERROR] The simulator does not support WS and spatial inter-core data reuse")
        
    def core_id(self, r, c):
        if r >= self.shape[0]:
            raise Exception(f"[ERROR] Invalid row index '{r}' since it exceeds its limit '{self.shape[0]}'")
        if c >= self.shape[1]:
            raise Exception(f"[ERROR] Invalid column index '{c}' since it exceeds its limit '{self.shape[1]}'")
        
        return r * self.shape[1] + c
    
    @property
    def cycles_preload(self) -> int:
        if self.dataflow == SACoreDataflow.OS:
            return self.m_tile
        elif self.dataflow == SACoreDataflow.WS:
            return self.k_tile
        else:
            raise Exception(f"[ERROR] Preload cycles undefined for the '{self.dataflow}' dataflow")
    
    @property
    def cycles_execute(self) -> int:
        if self.dataflow == SACoreDataflow.OS:
            return self.k_tile
        elif self.dataflow == SACoreDataflow.WS:
            return self.m_tile
        else:
            raise Exception(f"[ERROR] Execute cycles undefined for the '{self.dataflow}' dataflow")
    
    @property
    def cycles_flush(self) -> int:
        if self.dataflow == SACoreDataflow.OS:
            return self.m_tile
        elif self.dataflow == SACoreDataflow.WS:
            return self.k_tile
        else:
            raise Exception(f"[ERROR] Flush cycles undefined for the '{self.dataflow}' dataflow")
        
    def cycles(self, op_type: SACoreOperatorType):
        if op_type == SACoreOperatorType.PRELOAD:
            return self.cycles_preload
        elif op_type == SACoreOperatorType.EXECUTE:
            return self.cycles_execute
        elif op_type == SACoreOperatorType.FLUSH:
            return self.cycles_flush
        else:
            raise Exception(f"[ERROR] Undefined operator type: '{op_type}'")
            
    @property
    def n_cores(self) -> int:
        return self.shape[0] * self.shape[1]

    def print_info(self, indent: int=0):
        print(indent * " " + f"- tile shape: M({self.m_tile}), N({self.n_tile}), K({self.k_tile})")
        print(indent * " " + f"- dataflow:   {self.dataflow.name}")
        print(indent * " " + f"- memory:     {self.onc_mem_type.name}(capacity={self.onc_mem_cap})")
        print(indent * " " + f"- cluster:    shape={self.shape} / {'spatial reuse' if self.spatial_reuse else 'do not reuse data'}")
        

class MappingStage:
    def __init__(
        self,
        cluster:        CoreCluster,
        onc_buffers:    list[BufferDescriptor],
        n_preload_ops:  int,
        n_execute_ops:  int,
        n_flush_ops:    int,
    ):
        self.cluster        = cluster
        self.onc_buffers    = onc_buffers
        self.n_preload_ops  = n_preload_ops
        self.n_execute_ops  = n_execute_ops
        self.n_flush_ops    = n_flush_ops
        
        # if not isinstance(self.load_buffers, list):
        #     raise Exception(f"[ERROR] The load buffers should be passed as a list, not '{type(self.load_buffers).__name__}'")
        # if not isinstance(self.store_buffers, list):
        #     raise Exception(f"[ERROR] The store buffers should be passed as a list, not '{type(self.store_buffers).__name__}'")
        if not isinstance(self.onc_buffers, list):
            raise Exception(f"[ERROR] The on-chip buffers should be passed as a list, not '{type(self.onc_buffers).__name__}'")
    
    def print_info(self, indent: int=0):
        print(indent * " " + f"- on-chip buffers: {len(self.onc_buffers)}")
        print(indent * " " + f"- operations:  preload({self.n_preload_ops}) / execute({self.n_execute_ops}) / flush({self.n_flush_ops})")


class Mapping:
    def __init__(self, cluster: CoreCluster, variables: list[VariableDescriptor], stages: list[MappingStage]):
        self.cluster    = cluster
        self.variables  = variables
        self.stages     = stages
        
    @classmethod
    def create_matmul_mapping(cls, cluster: CoreCluster, M: int, N: int, K: int) -> 'Mapping':
        if cluster.dataflow == SACoreDataflow.OS:
            return cls.create_matmul_mapping_os(cluster, M, N, K)
        else:
            raise Exception("Weight Stationary dataflow will be implemented later")  # TODO: WS dataflow
    
    @classmethod
    def create_matmul_mapping_os(cls, cluster: CoreCluster, M: int, N: int, K: int, k_tile_blocking: bool=True) -> 'Mapping':
        # STEP 1: Define variables with tiling factors
        m_tile = cluster.m_tile
        n_tile = cluster.n_tile
        k_tile = cluster.k_tile
        
        ifm = VariableDescriptor("IFM", shape=(M, K), tile_shape=(m_tile, k_tile), word_size=cluster.word_size)
        wgt = VariableDescriptor("WGT", shape=(K, N), tile_shape=(k_tile, n_tile), word_size=cluster.word_size)
        ofm = VariableDescriptor("OFM", shape=(M, N), tile_shape=(m_tile, n_tile), word_size=cluster.ofm_word_size)
        
        variables = [ifm, wgt, ofm]
        
        # STEP 2: Define M/N/K grid dimension based on the on-chip memory capacity
        #   - m_grid and n_grid is related to the core cluster shape, which is critical to the GEMM core with
        #     output stationary dataflow.
        #   - k_grid is not related to the core cluster dimension but the on-chip buffer capacity. This specific
        #     loop blocking factor reduces the potential on-chip memory contention. You can turn off this loop
        #     blocking feature by setting 'k_tile_blocking' option to 'False'.
        m_grid = cluster.shape[0]
        n_grid = cluster.shape[1]
        k_grid = math.ceil(K / k_tile)
        
        if k_tile_blocking:    
            if cluster.onc_mem_type == MemoryType.L2 or cluster.spatial_reuse:
                seq_factor_base = m_grid * n_grid * m_tile * n_tile
                seq_factor_unit = ((m_grid * m_tile) + (n_grid * n_tile)) * k_tile
                seq_len_max     = math.ceil(K / k_tile)
            else:
                seq_factor_base = ofm.tile_size
                seq_factor_unit = ifm.tile_size + wgt.tile_size
                seq_len_max     = math.ceil(K / k_tile)
                    
                seq_factor_base = seq_factor_base * cluster.n_cores
                seq_factor_unit = seq_factor_unit * cluster.n_cores
            
            seq_len = min(
                math.floor((cluster.onc_mem_cap - seq_factor_base) / seq_factor_unit),
                seq_len_max,
            )
            
            if seq_len < 1:
                raise Exception(f"[ERROR] Insufficient on-chip memory capacity for the given workload configuration")
            
            k_grid = seq_len
            
        # STEP 3: Create mapping stages
        m_part = math.ceil(M / (m_grid * m_tile))
        n_part = math.ceil(N / (n_grid * n_tile))
        k_part = math.ceil(K / (k_grid * k_tile))
        
        stages: list[MappingStage] = []
        l1_mem_banks: dict[tuple[int, int], MemoryDescriptor] = {
            cluster.core_id(mg, ng): MemoryDescriptor.l1(cluster.core_id(mg, ng))
            for mg in range(m_grid)
            for ng in range(n_grid) 
        }
        l2_mem_bank = MemoryDescriptor.l2()
        
        for kp in range(k_part):
            for mp in range(m_part):
                for np in range(n_part):
                    
                    ifm_buffers = set()
                    wgt_buffers = set()
                    ofm_buffers = set()
                    
                    for mg in range(m_grid):
                        for ng in range(n_grid):
                            core_id = cluster.core_id(mg, ng)
                            
                            for kg in range(k_grid):
                                m = mp * m_grid + mg
                                n = np * n_grid + ng
                                k = kp * k_grid + kg
                                
                                ifm_tile_id = ifm.tile_id(m, k)
                                wgt_tile_id = wgt.tile_id(k, n)
                                ofm_tile_id = ofm.tile_id(m, n)
                                
                                if cluster.onc_mem_type == MemoryType.L2:
                                    ifm_mem_desc = l2_mem_bank
                                    wgt_mem_desc = l2_mem_bank
                                    ofm_mem_desc = l2_mem_bank
                                elif cluster.spatial_reuse:
                                    ifm_mem_desc = l1_mem_banks[ifm_tile_id % cluster.n_cores]
                                    wgt_mem_desc = l1_mem_banks[wgt_tile_id % cluster.n_cores]
                                    ofm_mem_desc = l1_mem_banks[core_id]
                                else:
                                    ifm_mem_desc = l1_mem_banks[core_id]
                                    wgt_mem_desc = l1_mem_banks[core_id]
                                    ofm_mem_desc = l1_mem_banks[core_id]
                                
                                ifm_buffers.add(ifm.create_buffer(ifm_mem_desc, ifm_tile_id, readonly=True))
                                wgt_buffers.add(wgt.create_buffer(wgt_mem_desc, wgt_tile_id, readonly=True))
                                ofm_buffers.add(ofm.create_buffer(ofm_mem_desc, ofm_tile_id, readonly=False))
                                
                    onc_buffers = []
                    onc_buffers += list(ifm_buffers)
                    onc_buffers += list(wgt_buffers)
                    onc_buffers += list(ofm_buffers)
                    
                    stage = MappingStage(
                        cluster=cluster,
                        # load_buffers=list(ifm_buffers.union(wgt_buffers)),
                        # store_buffers=list(ofm_buffers),
                        onc_buffers=onc_buffers,
                        n_preload_ops=1,
                        n_execute_ops=k_grid,
                        n_flush_ops=1,
                    )
                    
                    stages.append(stage)
        
        return cls(cluster=cluster, variables=variables, stages=stages)
    
    @classmethod
    def create_matmul_mapping_ws(cls, cluster: CoreCluster, M: int, N: int, K: int) -> 'Mapping':
        raise Exception(f"[ERROR] Weight Stationary is ")
        
    def print_info(self):
        # print("=== Core Clsuter")
        # self.cluster.print_info(indent=2)
        
        print("= Variables")
        for var in self.variables:
            print(f"  - {var}")
            
        print("= Stages")
        for idx, stage in enumerate(self.stages):
            print(f"STAGE {idx}")
            stage.print_info(indent=2)


class MappingProfileEntry:
    def __init__(
        self,
        # load_buffers:   list[BufferDescriptor] = [],
        # store_buffers:  list[BufferDescriptor] = [],
        # n_preload_ops:  int = 0,
        # n_execute_ops:  int = 0,
        # n_flush_ops:    int = 0,
    ):
        self.load_buffers:  list[BufferDescriptor] = []
        self.store_buffers: list[BufferDescriptor] = []
        self.n_preload_ops: int = 0
        self.n_execute_ops: int = 0
        self.n_flush_ops:   int = 0
        
    def __str__(self):
        return f"load: {len(self.load_buffers):<5d}  store: {len(self.store_buffers):<5d}  operators: preload({self.n_preload_ops:<2d}) execute({self.n_execute_ops:<2d}) flush({self.n_flush_ops:<2d})"
        
        
class ExecutionProfileEntry:
    def __init__(
        self,
        operation_cycles:   int = 0,
        load_traffic:       int = 0,
        store_traffic:      int = 0,
    ):
        self.operation_cycles   = operation_cycles
        self.load_traffic       = load_traffic
        self.store_traffic      = store_traffic
    
    @property
    def load_bandwidth(self) -> float:
        if self.operation_cycles == 0:
            return math.inf
        return self.load_traffic / self.operation_cycles
    
    @property
    def store_bandwidth(self) -> float:
        if self.operation_cycles == 0:
            return math.inf
        return self.store_traffic / self.operation_cycles
    
    def __str__(self):
        return f"cycles: {self.operation_cycles:<5d}  traffic: LOAD {self.load_traffic:<6d} [Bytes] STORE {self.store_traffic:<6d} [Bytes]   bandwidth: LOAD {self.load_bandwidth:<5.0f} [B/cycles] STORE {self.store_bandwidth:<5.0f} [B/cycles]"
        

class Profiler:
    def __init__(self, mapping: Mapping):
        self.mapping = mapping
        
        self.mapping_profile_entries:   list[MappingProfileEntry]   = []
        self.execution_profile_entries: list[ExecutionProfileEntry] = []
        
        self.generate_mapping_profile()
        self.generate_execution_profile()
    
    def generate_mapping_profile(self):
        onc_memory_buffers: list[BufferDescriptor] = []
        onc_memory_usage = 0
        
        for stage in self.mapping.stages:
            entry = MappingProfileEntry()
            
            # STEP 1: Compute load/store traffic
            for request_buffer in stage.onc_buffers:
                if request_buffer in onc_memory_buffers:
                    onc_memory_buffers.remove(request_buffer)
                    onc_memory_usage -= request_buffer.size
                else:
                    entry.load_buffers.append(request_buffer)
                    
                onc_memory_buffers.append(request_buffer)
                onc_memory_usage += request_buffer.size
                    
                while self.mapping.cluster.onc_mem_cap < onc_memory_usage:
                    victim_buffer = onc_memory_buffers.pop(0)
                    onc_memory_usage -= victim_buffer.size
                    
                    if not victim_buffer.readonly:
                        entry.store_buffers.append(victim_buffer)
                        
                if request_buffer not in onc_memory_buffers:
                    raise Exception(f"[ERROR] Invalid mapping: lack of on-chip memory space while generating mapping profile. This exception is may caused by the faulty implementation of the mapping algorithm. Please make sure that mapping algorithm guarantees that there aren't any out-of-memory situation.")

            # STEP 2: Compute operation cycles      
            entry.n_preload_ops += stage.n_preload_ops
            entry.n_execute_ops += stage.n_execute_ops
            entry.n_flush_ops   += stage.n_flush_ops
                        
            self.mapping_profile_entries.append(entry)
            
        for victim_buffer in onc_memory_buffers:
            if not victim_buffer.readonly:
                self.mapping_profile_entries[-1].store_buffers.append(victim_buffer)
            
    def generate_execution_profile(self):
        self.execution_profile_entries = []
        
        for cursor_mpe_idx in range(len(self.mapping_profile_entries) + 2):
            entry = ExecutionProfileEntry()
            
            store_stage_mpe_idx     = cursor_mpe_idx - 2
            execute_stage_mpe_idx   = cursor_mpe_idx - 1
            load_stage_mpe_idx      = cursor_mpe_idx
            
            if 0 <= store_stage_mpe_idx < len(self.mapping_profile_entries):
                mp_entry = self.mapping_profile_entries[store_stage_mpe_idx]
                entry.store_traffic += sum(map(lambda x: x.size, mp_entry.store_buffers))
                
            if 0 <= execute_stage_mpe_idx < len(self.mapping_profile_entries):
                mp_entry = self.mapping_profile_entries[execute_stage_mpe_idx]
                entry.operation_cycles += mp_entry.n_preload_ops * self.mapping.cluster.cycles_preload
                entry.operation_cycles += mp_entry.n_execute_ops * self.mapping.cluster.cycles_execute
                entry.operation_cycles += mp_entry.n_flush_ops   * self.mapping.cluster.cycles_flush
                
            if 0 <= load_stage_mpe_idx < len(self.mapping_profile_entries):
                mp_entry = self.mapping_profile_entries[load_stage_mpe_idx]
                entry.load_traffic += sum(map(lambda x: x.size, mp_entry.load_buffers))
            
            self.execution_profile_entries.append(entry)
            
    def print_mapping_profile(self):
        for entry_idx, entry in enumerate(self.mapping_profile_entries):
            print(f"entry: {entry_idx:<3d} >> {entry}")
            
    def print_execution_profile(self):
        total_cycles = 0
        for entry_idx, entry in enumerate(self.execution_profile_entries):
            total_cycles += entry.operation_cycles
            print(f"step:  {entry_idx:<3d} >> {entry}")
        print(f"total cycles: {total_cycles} cycles")