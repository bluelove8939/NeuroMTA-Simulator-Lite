import enum
import torch

from neuromta.framework import *
from neuromta.hardware.multi_tile import *


__all__ = [
    "TenstorrentConfig",
    "TenstorrentDevice",
]


class TenstorrentConfig(dict):
    def __init__(
        self,
        
        icnt_config: IcntConfig, 
        mem_config: MemConfig,
        mxu_config: MXUConfig,
        vpu_config: VPUConfig, 
    ):
        self["icnt_config"] = icnt_config
        self["mem_config"] = mem_config
        self["mxu_config"] = mxu_config
        self["vpu_config"] = vpu_config
        
    @classmethod
    def BLACKHOLE(cls) -> 'TenstorrentConfig':
        core_map = IcntCoreMap.from_shape((12, 16))
        core_map.grid[ :, 0   ] = IcntCoreType.DMA
        core_map.grid[ :, 8   ] = IcntCoreType.DMA
        core_map.grid[2:, 1:8 ] = IcntCoreType.NPU
        core_map.grid[2:, 9:16] = IcntCoreType.NPU
        
        icnt_config = IcntConfig(
            core_map=core_map,
            l1_mem_bank_size=parse_mem_cap_str("1.5MB"),
            main_mem_bank_size=parse_mem_cap_str("1.2GB"),
            flit_size=parse_mem_cap_str("16B"),
            control_packet_size=parse_mem_cap_str("32B"),
        )
        
        l1_mem_config = L1MemoryConfig(
            access_gran=parse_mem_cap_str("512B"),
        )
        
        main_mem_config = MainMemoryConfig(
            transfer_speed=9600,      # MT/s (DDR6 typical speed)
            ch_io_width=128,          # bits (DDR6 typical channel width)
            ch_num=1,                 # channels (example for DDR6)
            burst_len=256,            # bytes (typical burst length)
            is_ddr=True,
            processor_clock_freq=parse_freq_str("1.35GHz"),
        )
        
        mem_config = MemConfig(
            l1_config=l1_mem_config,
            main_config=main_mem_config,
        )
        
        mxu_config = MXUConfig(
            pe_arr_height=32,
            pe_arr_width=32,
            seq_len=32,
            dtype=torch.float32,
            acc_dtype=torch.float32,
            dataflow=MXUDataflow.OS,
            op_latency_per_byte=1,
        )
        
        vpu_config = VPUConfig(
            vreg_len=parse_mem_cap_str("128B"),
            vreg_num=32,
            vdtype=torch.float32,
            
            vlen_max=1024,
            vlen_min=32,
            
            unary_op_latency=1,
            arith_op_latency=2,
        )
        
        return cls(
            icnt_config=icnt_config,
            mem_config=mem_config,
            mxu_config=mxu_config,
            vpu_config=vpu_config,
        )
        
    @classmethod
    def WORMHOLE(cls) -> 'TenstorrentConfig':
        core_map = IcntCoreMap.from_shape((10, 10))
        core_map.grid[:, 0   ] = IcntCoreType.DMA
        core_map.grid[:, 5   ] = IcntCoreType.DMA
        core_map.grid[:, 1:5 ] = IcntCoreType.NPU
        core_map.grid[:, 6:10] = IcntCoreType.NPU
        
        icnt_config = IcntConfig(
            core_map=core_map,
            l1_mem_bank_size=parse_mem_cap_str("1MB"),
            main_mem_bank_size=parse_mem_cap_str("1.2GB"),
            flit_size=parse_mem_cap_str("16B"),
            control_packet_size=parse_mem_cap_str("32B"),
        )
        
        l1_mem_config = L1MemoryConfig(
            access_gran=parse_mem_cap_str("512B"),
        )
        
        main_mem_config = MainMemoryConfig(
            transfer_speed=9600,      # MT/s (DDR6 typical speed)
            ch_io_width=128,          # bits (DDR6 typical channel width)
            ch_num=1,                 # channels (example for DDR6)
            burst_len=256,            # bytes (typical burst length)
            is_ddr=True,
            processor_clock_freq=parse_freq_str("1GHz"),
        )
        
        mem_config = MemConfig(
            l1_config=l1_mem_config,
            main_config=main_mem_config,
        )
        
        mxu_config = MXUConfig(
            pe_arr_height=32,
            pe_arr_width=32,
            seq_len=32,
            dtype=torch.float32,
            acc_dtype=torch.float32,
            dataflow=MXUDataflow.OS,
            op_latency_per_byte=1,
        )
        
        vpu_config = VPUConfig(
            vreg_len=parse_mem_cap_str("128B"),
            vreg_num=32,
            vdtype=torch.float32,
            
            vlen_max=1024,
            vlen_min=32,
            
            unary_op_latency=1,
            arith_op_latency=2,
        )
        
        return cls(
            icnt_config=icnt_config,
            mem_config=mem_config,
            mxu_config=mxu_config,
            vpu_config=vpu_config,
        )
    
    @classmethod
    def EAGLE_N1(cls) -> 'TenstorrentConfig':
        core_map = IcntCoreMap.from_shape((6, 5))
        core_map.grid[:, 0 ] = IcntCoreType.DMA
        core_map.grid[:, 1:] = IcntCoreType.NPU
        
        icnt_config = IcntConfig(
            core_map=core_map,
            l1_mem_bank_size=parse_mem_cap_str("1500KB"),
            main_mem_bank_size=parse_mem_cap_str("11GB"),
            flit_size=parse_mem_cap_str("16B"),
            control_packet_size=parse_mem_cap_str("32B"),
        )
        
        l1_mem_config = L1MemoryConfig(
            access_gran=parse_mem_cap_str("512B"),
        )
        
        main_mem_config = MainMemoryConfig(
            transfer_speed=9600,      # MT/s (DDR6 typical speed)
            ch_io_width=128,          # bits (DDR6 typical channel width)
            ch_num=1,                 # channels (example for DDR6)
            burst_len=256,            # bytes (typical burst length)
            is_ddr=True,
            processor_clock_freq=parse_freq_str("500MHz"),
        )
        
        mem_config = MemConfig(
            l1_config=l1_mem_config,
            main_config=main_mem_config,
        )
        
        mxu_config = MXUConfig(
            pe_arr_height=32,
            pe_arr_width=32,
            seq_len=32,
            dtype=torch.float32,
            acc_dtype=torch.float32,
            dataflow=MXUDataflow.OS,
            op_latency_per_byte=1,
        )
        
        vpu_config = VPUConfig(
            vreg_len=parse_mem_cap_str("128B"),
            vreg_num=32,
            vdtype=torch.float32,
            
            vlen_max=1024,
            vlen_min=32,
            
            unary_op_latency=1,
            arith_op_latency=2,
        )
        
        return cls(
            icnt_config=icnt_config,
            mem_config=mem_config,
            mxu_config=mxu_config,
            vpu_config=vpu_config,
        )
        

class TenstorrentDevice(MTAccelerator):
    def __init__(self, icnt_config, mem_config, mxu_config, vpu_config):
        super().__init__(icnt_config, mem_config, mxu_config, vpu_config)
