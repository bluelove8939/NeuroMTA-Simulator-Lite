import enum
import torch

from neuromta.framework import *
from neuromta.hardware.multi_tile import *

from neuromta.hardware.companions.booksim import PYBOOKSIM2_AVAILABLE
from neuromta.hardware.companions.dramsim import PYDRAMSIM3_AVAILABLE, DRAMSim3Config


__all__ = [
    "TenstorrentConfig",
    "TenstorrentDevice",
]


class TenstorrentConfig(dict):
    def __init__(
        self,
        
        processor_clock_freq: int,
        icnt_config: IcntConfig, 
        cmap_config: CmapConfig,
        mem_config: MemConfig,
        mxu_config: MXUConfig,
        vpu_config: VPUConfig, 
    ):
        self["processor_clock_freq"] = processor_clock_freq
        self["icnt_config"] = icnt_config
        self["cmap_config"] = cmap_config
        self["mem_config"] = mem_config
        self["mxu_config"] = mxu_config
        self["vpu_config"] = vpu_config
        
    @classmethod
    def BLACKHOLE(cls) -> 'TenstorrentConfig':
        processor_clock_freq = parse_freq_str("1.35GHz")
        
        cmap_config = CmapConfig.from_shape(
            shape=(12, 16),
            l1_mem_bank_size=parse_mem_cap_str("1.5MB"),
            main_mem_bank_size=parse_mem_cap_str("1.2GB"),
        )
        
        cmap_config.grid[ :, 0   ] = CmapCoreType.DMA
        cmap_config.grid[ :, 8   ] = CmapCoreType.DMA
        cmap_config.grid[2:, 1:8 ] = CmapCoreType.NPU
        cmap_config.grid[2:, 9:16] = CmapCoreType.NPU
        
        if PYBOOKSIM2_AVAILABLE:
            booksim2_config = cmap_config.create_booksim2_config(
                cmd_wait_resolution=50,
            )
        else:
            booksim2_config = None
        
        icnt_config = IcntConfig(
            flit_size=parse_mem_cap_str("16B"),
            control_packet_size=parse_mem_cap_str("32B"),
            booksim2_enable=PYBOOKSIM2_AVAILABLE,
            booksim2_config=booksim2_config,
        )
        
        l1_mem_config = L1MemoryConfig(
            access_gran=parse_mem_cap_str("512B"),
        )

        if PYDRAMSIM3_AVAILABLE:
            dramsim3_config = DRAMSim3Config(
                config_path="GDDR5_8Gb_x32",
                processor_clock_freq=processor_clock_freq,
                cmd_wait_resolution=5,
            )
        else:
            dramsim3_config = None

        main_mem_config = MainMemoryConfig(
            transfer_speed=9600,      # MT/s (DDR6 typical speed)
            ch_io_width=128,          # bits (DDR6 typical channel width)
            ch_num=1,                 # channels (example for DDR6)
            burst_len=256,            # bytes (typical burst length)
            is_ddr=True,
            processor_clock_freq=processor_clock_freq,
            
            dramsim3_enable=PYDRAMSIM3_AVAILABLE,
            dramsim3_config=dramsim3_config,
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
            processor_clock_freq=processor_clock_freq,
            cmap_config=cmap_config,
            icnt_config=icnt_config,
            mem_config=mem_config,
            mxu_config=mxu_config,
            vpu_config=vpu_config,
        )
        

class TenstorrentDevice(MTAccelerator):
    def __init__(self, processor_clock_freq, cmap_config, icnt_config, mem_config, mxu_config, vpu_config):
        super().__init__(cmap_config, icnt_config, mem_config, mxu_config, vpu_config)
        
        self.processor_clock_freq = processor_clock_freq
