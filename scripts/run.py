from neuromta.common import SACoreDataflow, MemoryType, parse_mem_cap_str
from neuromta.core_cluster import CoreCluster, ClusterMapping


if __name__ == "__main__":
    cluster = CoreCluster(
        word_size=4, ofm_word_size=4,
        m_tile=32, n_tile=32, k_tile=32, dataflow=SACoreDataflow.OS,
        onc_mem_cap=parse_mem_cap_str("32MB"), onc_mem_type=MemoryType.L1,
        shape=(4, 4), spatial_reuse=False,
    )
    
    mapping = ClusterMapping.create_matmul_mapping(cluster=cluster, M=256, N=256, K=256)
    mapping.print_info()