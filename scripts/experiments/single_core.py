from neuromta.simulation.common import SACoreDataflow, MemoryType, parse_mem_cap_str
from neuromta.simulation.profiler import CoreCluster, Mapping, Profiler


if __name__ == "__main__":
    print(f"=== CLUSTER INFORMATION ===")
    cluster = CoreCluster(
        word_size=1, ofm_word_size=1,
        m_tile=64, n_tile=64, k_tile=128, dataflow=SACoreDataflow.OS,
        onc_mem_cap=parse_mem_cap_str("32MB"), onc_mem_type=MemoryType.L1,
        shape=(1, 1), spatial_reuse=False,
    )
    cluster.print_info()
    
    print("\n=== MAPPING INFORMATION ===")
    mapping = Mapping.create_matmul_mapping(cluster=cluster, M=64, N=512, K=512)
    mapping.print_info()
    
    print("\n=== PROFILE INFORMATION ===")
    profiler = Profiler(mapping=mapping)
    print("= Mapping Profile")
    profiler.print_mapping_profile()
    print("= Execution Profile")
    profiler.print_execution_profile()