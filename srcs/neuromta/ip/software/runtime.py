import torch

from neuromta.common.buffer_handle import BufferHandle

from neuromta.ip.hardware.memory import MemoryContext
from neuromta.ip.hardware.matrix_unit import MXUConfig
from neuromta.ip.hardware.vector_unit import VPUConfig


class MemoryLayout:
    def __init__(self, tensor: torch.Tensor, tile_shape: tuple[int, ...]):
        self._tensor = tensor
        self._tile_shape = tile_shape
        self._tiled_view = self._tensor.view(-1, *self._tile_shape)
        
        
class RuntimeEnvironment:
    def __init__(self, cores: list[tuple[int, int]], mem_context: MemoryContext, mxu_config: MXUConfig, vpu_config: VPUConfig):
        self._cores         = cores
        self._mem_context   = mem_context
        self._mxu_config    = mxu_config
        self._vpu_config    = vpu_config


class Runtime:
    # def __init__(self, mem_context: MemoryContext = None, mxu_config: MXUConfig = None, vpu_config: VPUConfig = None):
    #     self._mem_context = mem_context if mem_context is not None else MemoryContext()
    #     self._mxu_config  = mxu_config  if mxu_config  is not None else MXUConfig()
    #     self._vpu_config  = vpu_config  if vpu_config  is not None else VPUConfig()

    # def initialize_runtime(self, mem_context: MemoryContext=None, mxu_config: MXUConfig=None, vpu_config: VPUConfig=None):
    #     self._mem_context = mem_context if mem_context is not None else self._mem_context
    #     self._mxu_config  = mxu_config  if mxu_config  is not None else self._mxu_config
    #     self._vpu_config  = vpu_config  if vpu_config  is not None else self._vpu_config

    def linear(
        self, x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor = None
    ) -> torch.Tensor:
        M, K = x.shape
        _, N = weight.shape

    # def conv2d(
    #     self, x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor = None,
    #     stride: tuple[int, int] = (1, 1), padding: tuple[int, int] = (0, 0)
    # ):
    #     if isinstance(stride, int):
    #         stride = (stride, stride)
    #     if isinstance(padding, int):
    #         padding = (padding, padding)
        
    #     N, C, H, W = x.shape
    #     K, _, FH, FW = weight.shape
    #     SH, SW = stride
    #     PH, PW = padding
        
    #     OH = (H + 2 * PH - FH) // SH + 1
    #     OW = (W + 2 * PW - FW) // SW + 1
        
    #     y = torch.zeros((N, K, OH, OW), dtype=x.dtype)
        
    # @property
    # def is_initialized(self) -> bool:
    #     return self._mem_context is not None and self._mxu_config is not None and self._vpu_config is not None
    
    
if __name__ == "__main__":
    tensor = torch.randn((4, 3, 224, 224), dtype=torch.float32)
    tile_shape = (1, 1, 112, 112)
    memory_layout = MemoryLayout(tensor, tile_shape)
    print("Tensor shape:", tensor.shape)
    print("Tiled view shape:", memory_layout._tiled_view.shape)