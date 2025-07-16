import copy
import math
from typing import Sequence

from neuromta.common import *
from neuromta.descriptor import *


__all__ = [
    "CoreCluster",
    "ClusterMappingStage", 
    "ClusterMapping"
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
        
    def cycles_preload(self) -> int:
        if self.dataflow == SACoreDataflow.OS:
            return self.m_tile
        elif self.dataflow == SACoreDataflow.WS:
            return self.k_tile
        else:
            raise Exception(f"[ERROR] Preload cycles undefined for the '{self.dataflow.name}' dataflow")
    
    def cycles_execute(self) -> int:
        if self.dataflow == SACoreDataflow.OS:
            return self.k_tile
        elif self.dataflow == SACoreDataflow.WS:
            return self.m_tile
        else:
            raise Exception(f"[ERROR] Execute cycles undefined for the '{self.dataflow.name}' dataflow")
        
    def cycles_flush(self) -> int:
        if self.dataflow == SACoreDataflow.OS:
            return self.m_tile
        elif self.dataflow == SACoreDataflow.WS:
            return self.k_tile
        else:
            raise Exception(f"[ERROR] Flush cycles undefined for the '{self.dataflow.name}' dataflow")
            
    @property
    def n_cores(self) -> int:
        return self.shape[0] * self.shape[1]

    def print_info(self, indent: int=0):
        print(indent * " " + f"- tile shape: M({self.m_tile}), N({self.n_tile}), K({self.k_tile})")
        print(indent * " " + f"- dataflow:   {self.dataflow.name}")
        print(indent * " " + f"- memory:     {self.onc_mem_type.name}(capacity={self.onc_mem_cap})")
        print(indent * " " + f"- cluster:    shape={self.shape} / {'spatial reuse' if self.spatial_reuse else 'do not reuse data'}")
        

class ClusterMappingStage:
    def __init__(
        self,
        cluster:        CoreCluster,
        load_buffers:   list[BufferDescriptor],
        store_buffers:  list[BufferDescriptor],
        n_preload_ops:  int,
        n_execute_ops:  int,
        n_flush_ops:    int,
    ):
        self.cluster        = cluster
        self.load_buffers   = copy.deepcopy(load_buffers)
        self.store_buffers  = copy.deepcopy(store_buffers)
        self.n_preload_ops  = n_preload_ops
        self.n_execute_ops  = n_execute_ops
        self.n_flush_ops    = n_flush_ops
        
        if not isinstance(self.load_buffers, list):
            raise Exception(f"[ERROR] The load buffers should be passed as a list, not '{type(self.load_buffers).__name__}'")
        if not isinstance(self.store_buffers, list):
            raise Exception(f"[ERROR] The store buffers should be passed as a list, not '{type(self.store_buffers).__name__}'")

        self._cached_cycles = None
        self._cached_load_traffic = None
        self._cached_store_traffic = None
    
    def cycles(self) -> int:
        if self._cached_cycles is not None:
            return self._cached_cycles
        
        self._cached_cycles = 0
        self._cached_cycles += self.n_preload_ops * self.cluster.cycles_preload()
        self._cached_cycles += self.n_execute_ops * self.cluster.cycles_execute()
        self._cached_cycles += self.n_flush_ops   * self.cluster.cycles_flush()
        
        return self._cached_cycles
    
    def load_traffic(self):
        if self._cached_load_traffic is not None:
            return self._cached_load_traffic
        
        self._cached_load_traffic = sum(map(lambda x: x.size, self.load_buffers))
        return self._cached_load_traffic
    
    def store_traffic(self):
        if self._cached_store_traffic is not None:
            return self._cached_store_traffic
        
        self._cached_store_traffic = sum(map(lambda x: x.size, self.store_buffers))
        return self._cached_store_traffic
    
    def print_info(self, indent: int=0):
        print(indent * " " + f"- number of load buffers:  {len(self.load_buffers)}")
        print(indent * " " + f"- number of store buffers: {len(self.store_buffers)}")
        print(indent * " " + f"- operations:  preload({self.n_preload_ops}) / execute({self.n_execute_ops}) / flush({self.n_flush_ops})")


class ClusterMapping:
    def __init__(self, cluster: CoreCluster, variables: Sequence[VariableDescriptor], stages: Sequence[ClusterMappingStage]):
        self.cluster    = cluster
        self.variables  = variables
        self.stages     = stages
        
    @classmethod
    def create_matmul_mapping(cls, cluster: CoreCluster, M: int, N: int, K: int) -> 'ClusterMapping':
        if cluster.dataflow == SACoreDataflow.OS:
            return cls.create_matmul_mapping_os(cluster, M, N, K)
        else:
            raise Exception("Weight Stationary dataflow will be implemented later")  # TODO: WS dataflow
    
    @classmethod
    def create_matmul_mapping_os(cls, cluster: CoreCluster, M: int, N: int, K: int) -> 'ClusterMapping':
        m_tile = cluster.m_tile
        n_tile = cluster.n_tile
        k_tile = cluster.k_tile
        
        m_tile_num = math.ceil(M / m_tile)
        n_tile_num = math.ceil(N / n_tile)
        k_tile_num = math.ceil(K / k_tile)
        
        # STEP 1: Define variables
        ifm = VariableDescriptor("IFM", shape=(M, K), tile_shape=(m_tile, k_tile), word_size=cluster.word_size)
        wgt = VariableDescriptor("WGT", shape=(K, N), tile_shape=(k_tile, n_tile), word_size=cluster.word_size)
        ofm = VariableDescriptor("OFM", shape=(M, N), tile_shape=(m_tile, n_tile), word_size=cluster.ofm_word_size)
        
        variables = [ifm, wgt, ofm]
        
        # STEP 2: Define M/N/K grid dimension based on the on-chip memory capacity
        m_grid = cluster.shape[0]
        n_grid = cluster.shape[1]
                    
        if cluster.onc_mem_type == MemoryType.L2 or cluster.spatial_reuse:
            seq_factor_base = m_grid * n_grid * m_tile * n_tile
            seq_factor_unit = ((m_grid * m_tile) + (n_grid * n_tile)) * k_tile
            seq_len_max     = k_tile_num
        else:
            seq_factor_base = ofm.tile_size
            seq_factor_unit = ifm.tile_size + wgt.tile_size
            seq_len_max     = k_tile_num
                
            seq_factor_base = seq_factor_base * cluster.n_cores
            seq_factor_unit = seq_factor_unit * cluster.n_cores
        
        seq_len = min(
            math.floor((cluster.onc_mem_cap - seq_factor_base) / seq_factor_unit),
            seq_len_max,
        )
        
        if seq_len < 1:
            raise Exception(f"[ERROR] Insufficient on-chip memory capacity for the given workload configuration")
        
        k_grid = seq_len
            
        # STEP 2: Create mapping stages
        m_part = math.ceil(M / (m_grid * m_tile))
        n_part = math.ceil(N / (n_grid * n_tile))
        k_part = math.ceil(K / (k_grid * k_tile))
        
        stages: list[ClusterMappingStage] = []
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
                            # mem_desc = MemoryDescriptor.l1(core_id=core_id) if cluster.onc_mem_type == MemoryType.L1 else MemoryDescriptor.l2()
                            
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
                                    ifm_mem_desc = l1_mem_banks[ifm_tile_id // cluster.n_cores]
                                    wgt_mem_desc = l1_mem_banks[wgt_tile_id // cluster.n_cores]
                                    ofm_mem_desc = l1_mem_banks[core_id]
                                else:
                                    ifm_mem_desc = l1_mem_banks[core_id]
                                    wgt_mem_desc = l1_mem_banks[core_id]
                                    ofm_mem_desc = l1_mem_banks[core_id]
                                
                                ifm_buffers.add(ifm.create_buffer(ifm_mem_desc, ifm_tile_id))
                                wgt_buffers.add(wgt.create_buffer(wgt_mem_desc, wgt_tile_id))
                                ofm_buffers.add(ofm.create_buffer(ofm_mem_desc, ofm_tile_id))
                    
                    stage = ClusterMappingStage(
                        cluster=cluster,
                        load_buffers=list(ifm_buffers.union(wgt_buffers)),
                        store_buffers=list(ofm_buffers),
                        n_preload_ops=1,
                        n_execute_ops=k_grid,
                        n_flush_ops=1,
                    )
                    
                    stages.append(stage)
        
        # STEP 3: Remove overlapping buffer load operations (temporal data reuse)       
        for sidx, stage in enumerate(stages):    
            if sidx == 0:
                continue
            
            prev_load_buffers = set(stages[sidx-1].load_buffers)
            curr_load_buffers = set(stage.load_buffers)
            
            stage.load_buffers = list(curr_load_buffers.difference(prev_load_buffers))
        
        return cls(cluster=cluster, variables=variables, stages=stages)
    
    @classmethod
    def create_matmul_mapping_ws(cls, cluster: CoreCluster, M: int, N: int, K: int) -> 'ClusterMapping':
        raise Exception(f"[ERROR] Weight Stationary is ")
    
    def print_info(self):
        print("=== Core Clsuter")
        self.cluster.print_info(indent=2)
        
        print("\n=== Variables")
        for var in self.variables:
            print(f"  - {var}")
            
        print("\n=== Stages")
        for idx, stage in enumerate(self.stages):
            print(f"= STAGE {idx}")
            stage.print_info(indent=2)
        