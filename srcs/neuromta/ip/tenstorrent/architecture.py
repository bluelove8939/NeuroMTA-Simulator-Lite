import os
import math
import torch

from neuromta.framework import *
from neuromta.hardware.multi_tile import *

from neuromta.hardware.companions.booksim import PYBOOKSIM2_AVAILABLE
from neuromta.hardware.companions.dramsim import PYDRAMSIM3_AVAILABLE, DRAMSim3Config, create_new_dramsim_config_file


__all__ = [
    "TenstorrentConfig",
    "TenstorrentDevice",
]


TENSTORRENT_IP_ROOT = os.path.abspath(os.path.dirname(__file__))
TENSTORRENT_IP_CACHE_DIR = os.path.join(TENSTORRENT_IP_ROOT, ".cache")
TENSTORRENT_IP_DRAMSIM_CONFIG_FMT = os.path.join(TENSTORRENT_IP_CACHE_DIR, "dramsim_{config_name}.ini").format


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
        config_name = "blackhole"
        
        processor_clock_freq    = parse_freq_str("1GHz")
        main_mem_channel_size   = parse_mem_cap_str("4GB")
        l1_mem_bank_size        = parse_mem_cap_str("1.5MB")

        cmap_shape = (12, 16)
        n_npu_core = 12 * 14
        n_dma_core = 12 * 2
        n_dma_core_per_channel    = 3
        n_main_mem_channels = math.ceil(n_dma_core / n_dma_core_per_channel)
        
        cmap_config = CmapConfig(
            shape=cmap_shape,
            n_l1_mem_bank=n_npu_core,
            n_main_mem_channels=n_main_mem_channels,
            l1_mem_bank_size=l1_mem_bank_size,
            main_mem_channel_size=main_mem_channel_size,
        )
        
        for row in range(12):
            ch_row = row // n_dma_core_per_channel   # the channel ID of the given row

            cmap_config.add_core(CmapCoreType.DMA, coord=(row, 0), mem_bank_idx=ch_row * 2)         # DMA cores that are located at 3 consecutive row will be assigned the same memory channel
            cmap_config.add_core(CmapCoreType.DMA, coord=(row, 8), mem_bank_idx=ch_row * 2 + 1)     # DMA cores that are located at 3 consecutive row will be assigned the same memory channel

            for i in range(7):
                cmap_config.add_core(CmapCoreType.NPU, coord=(row, 1 + i), mem_bank_idx=(row * 14) + i)
                cmap_config.add_core(CmapCoreType.NPU, coord=(row, 9 + i), mem_bank_idx=(row * 14) + i + 7)

        if PYBOOKSIM2_AVAILABLE:
            booksim2_config = cmap_config.create_booksim2_config(
                cmd_wait_resolution=50,
            )
        else:
            booksim2_config = None
        
        icnt_config = IcntConfig(
            flit_size=parse_mem_cap_str("16B"),  # TODO: (flit size) * (processor clock) * (full-duplex) * (node per router) = 16B * 1GHz * 2 * 6 = 192GB/s (???)
            control_packet_size=parse_mem_cap_str("32B"),
            booksim2_enable=PYBOOKSIM2_AVAILABLE,
            booksim2_config=booksim2_config,
        )
        
        l1_mem_config = L1MemoryConfig(
            access_gran=parse_mem_cap_str("512B"),
        )

        if PYDRAMSIM3_AVAILABLE:
            dramsim3_config_path    = TENSTORRENT_IP_DRAMSIM_CONFIG_FMT(config_name=config_name)
            dramsim3_channel_size   = main_mem_channel_size // (1024 * 1024)    # GB -> MB

            create_new_dramsim_config_file(
                src_config_path="GDDR5_8Gb_x32.ini",  # TODO: originally, the source config file should be GDDR6_8Gb_x16.ini, but there are some errors ...
                new_config_path=dramsim3_config_path,
                channel_size=dramsim3_channel_size,
                n_channel=n_main_mem_channels,
            )
            
            dramsim3_config = DRAMSim3Config(
                config_path=dramsim3_config_path,
                processor_clock_freq=processor_clock_freq,
                cmd_wait_resolution=5,
            )
        else:
            dramsim3_config = None

        main_mem_config = MainMemoryConfig(
            # STATIC MEMORY CONFIG (used if pydramsim is not available)
            transfer_speed=7000,      # MT/s (DDR6 typical speed)
            ch_io_width=32,           # bits (DDR6 typical channel width)
            ch_num=1,                 # channels (example for DDR6)
            burst_len=32,             # bytes (typical burst length)
            is_ddr=True,
            processor_clock_freq=processor_clock_freq,
            
            # DRAMSIM CONFIG
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
