import enum


__all__ = [
    "TensorProcessorConfig",
]


class MXUDataflow(enum.Enum):
    OS = enum.auto()  # Output Stationary
    WS = enum.auto()  # Weight Stationary

class TensorProcessorConfig:
    def __init__(
        self,
        
        mxu_pe_arr_height: int = 32,
        mxu_pe_arr_width: int = 32,
        mxu_op_latency_per_byte: int = 1,
        mxu_dataflow: MXUDataflow = MXUDataflow.OS,
        
        vpu_max_vlen: int = 256,
        vpu_unary_op_latency: int = 2,
        vpu_arith_op_latency: int = 4,
    ):
        self.mxu_pe_arr_height = mxu_pe_arr_height
        self.mxu_pe_arr_width = mxu_pe_arr_width
        self.mxu_op_latency_per_byte = mxu_op_latency_per_byte
        
        self.vpu_max_vlen = vpu_max_vlen
        self.vpu_unary_op_latency = vpu_unary_op_latency
        self.vpu_arith_op_latency = vpu_arith_op_latency