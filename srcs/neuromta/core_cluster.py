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
        load_buffers:   Sequence[BufferDescriptor],
        store_buffers:  Sequence[BufferDescriptor],
        n_preload_ops:  int,
        n_execute_ops:  int,
        n_flush_ops:    int,
    ):
        self.load_buffers   = copy.deepcopy(load_buffers)
        self.store_buffers  = copy.deepcopy(store_buffers)
        self.n_preload_ops  = n_preload_ops
        self.n_execute_ops  = n_execute_ops
        self.n_flush_ops    = n_flush_ops
        
        if not isinstance(self.load_buffers, Sequence):
            self.load_buffers = [self.load_buffers,]
        if not isinstance(self.store_buffers, Sequence):
            self.store_buffers = [self.store_buffers,]
            
    def print_info(self, indent: int=0):
        print(indent * " " + f"- load buffers")
        for b in self.load_buffers:
            print(indent * " " + f"  * {b}")
            
        print(indent * " " + f"- store buffers")
        for b in self.store_buffers:
            print(indent * " " + f"  * {b}")
            
        print(indent * " " + f"- operations")
        print(indent * " " + f"  * preload: {self.n_preload_ops}")
        print(indent * " " + f"  * execute: {self.n_execute_ops}")
        print(indent * " " + f"  * flush:   {self.n_flush_ops}")


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
        
        for kp in range(k_part):
            for mp in range(m_part):
                ifm_buffers = [
                    ifm.create_buffer(ifm.tile_id(mp * m_grid + mg, kp * k_grid + kg,))
                    for mg in range(m_grid)
                    for kg in range(k_grid)
                ]
                
                for np in range(n_part):  
                    wgt_buffers = [
                        wgt.create_buffer(wgt.tile_id(kp * k_grid + kg, np * n_grid + ng,))
                        for kg in range(k_grid)
                        for ng in range(n_grid)
                    ]
                    
                    ofm_buffers = [
                        ofm.create_buffer(ofm.tile_id(mp * m_grid + mg, np * n_grid + ng,))
                        for mg in range(m_grid)
                        for ng in range(n_grid)
                    ]
                    
                    stage = ClusterMappingStage(
                        load_buffers=ifm_buffers + wgt_buffers,
                        store_buffers=ofm_buffers,
                        n_preload_ops=1,
                        n_execute_ops=k_grid,
                        n_flush_ops=1,
                    )
                    
                    stages.append(stage)
                    
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
        