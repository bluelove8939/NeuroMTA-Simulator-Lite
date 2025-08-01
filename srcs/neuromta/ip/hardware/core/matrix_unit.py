import enum
import numpy as np


__all__ = [
    "MXUDataflow",
    "MXUConfig",
    "MXUContext",
]


class MXUDataflow(enum.Enum):
    OS = enum.auto()  # Output Stationary
    WS = enum.auto()  # Weight Stationary
    
    
class MXUConfig(dict):
    def __init__(
        self,
        
        pe_arr_height: int = 32,
        pe_arr_width: int = 32,
        seq_len: int = 256,
        acc_dtype: np.dtype = np.float32,
        dataflow: MXUDataflow = MXUDataflow.OS,
        op_latency_per_byte: int = 1,
    ):
        super().__init__()
        
        self["pe_arr_height"] = pe_arr_height
        self["pe_arr_width"] = pe_arr_width
        self["seq_len"] = seq_len
        self["acc_dtype"] = acc_dtype
        self["dataflow"] = dataflow
        self["op_latency_per_byte"] = op_latency_per_byte
        
    def create_context(self) -> "MXUContext":
        return MXUContext(**self)


class MXUContext:
    def __init__(
        self,
        
        pe_arr_height: int = 32,
        pe_arr_width: int = 32,
        seq_len: int = 256,
        dtype: np.dtype = np.float32,
        acc_dtype: np.dtype = np.float32,
        dataflow: MXUDataflow = MXUDataflow.OS,
        op_latency_per_byte: int = 1,
    ):
        self.pe_arr_height  = pe_arr_height
        self.pe_arr_width   = pe_arr_width
        self.seq_len        = seq_len
        self._dtype         = np.dtype(dtype)
        self._acc_dtype     = np.dtype(acc_dtype)
        self._dataflow      = dataflow
        self.op_latency_per_byte = op_latency_per_byte
        
        # Initialize registers
        self._pe_arr_regs: np.ndarray = np.zeros((self.pe_arr_height, self.pe_arr_width), dtype=self._acc_dtype)
        self._acc_regs:    np.ndarray = np.zeros((self.seq_len, self.pe_arr_width), dtype=self._acc_dtype) if self._dataflow == MXUDataflow.WS else None
        
    def get_preload_pe_arr_cycles(self) -> int:
        return self.pe_arr_width
    
    def get_preload_acc_regs_cycles(self, partial_seq_len: int=None) -> int:
        return self.seq_len if partial_seq_len is None else partial_seq_len
    
    def get_execute_cycles(self) -> int:
        return self.seq_len * self.op_latency_per_byte * self.dtype.itemsize
    
    def get_flush_pe_arr_cycles(self) -> int:
        return self.pe_arr_width
    
    def get_flush_acc_regs_cycles(self, partial_seq_len: int=None) -> int:
        return self.seq_len if partial_seq_len is None else partial_seq_len
    
    def get_pe_arr_regs(self) -> np.ndarray:
        return self._pe_arr_regs.copy()
    
    def get_acc_regs(self) -> np.ndarray:
        if self._acc_regs is None:
            return self.get_pe_arr_regs()
        return self._acc_regs.copy()
    
    def load_tile_pe_arr(self, tile: np.ndarray):
        if tile.shape != self.pe_arr_shape:
            raise Exception(f"[ERROR] Tile shape {tile.shape} does not match PE array shape {(self.pe_arr_height, self.pe_arr_width)}.")
        
        self._pe_arr_regs[:, :] = tile.astype(dtype=self._acc_dtype)
        
    def load_tile_acc_regs(self, tile: np.ndarray, offset: int=0):
        if self._acc_regs is None:
            raise Exception("[ERROR] Accumulator registers are not available in this dataflow.")
        if tile.shape != self.acc_regs_shape:
            raise Exception(f"[ERROR] Tile shape {tile.shape} does not match accumulator registers shape {self.acc_regs_shape}.")

        st = offset
        ed = offset + tile.shape[0]
        
        self._acc_regs[st:ed, :] = tile.astype(dtype=self._acc_dtype)
        
    def execute_gemm(self, ifm_tile: np.ndarray, wgt_tile: np.ndarray=None) -> np.ndarray:
        if self._dataflow == MXUDataflow.OS:
            if wgt_tile is None:
                raise Exception("[ERROR] WGT tile must be provided for OS dataflow.")
            if ifm_tile.shape != self.ifm_tile_shape:
                raise Exception(f"[ERROR] IFM tile shape {ifm_tile.shape} does not match expected shape {self.ifm_tile_shape}.")
            if wgt_tile.shape != self.wgt_tile_shape:
                raise Exception(f"[ERROR] WGT tile shape {wgt_tile.shape} does not match expected shape {self.wgt_tile_shape}.")
            
            self._pe_arr_regs[:, :] = (ifm_tile @ wgt_tile) + self._pe_arr_regs
        elif self._dataflow == MXUDataflow.WS:
            if wgt_tile is not None:
                raise Exception("[ERROR] WGT tile should not be provided for WS dataflow.")
            if ifm_tile.shape != self.ifm_tile_shape:
                raise Exception(f"[ERROR] IFM tile shape {ifm_tile.shape} does not match expected shape {self.ifm_tile_shape}.")
            
            self._acc_regs[:, :] = (ifm_tile @ self._pe_arr_regs) + self._acc_regs
        else:
            raise Exception(f"[ERROR] Unsupported MXU dataflow: {self._dataflow}.")
        
    def flush_pe_arr(self):
        self._pe_arr_regs[:, :] = 0
        
    def flush_acc_regs(self) -> np.ndarray:
        if self._acc_regs is None:
            raise Exception("[ERROR] Accumulator registers are not available in this dataflow.")
        
        self._acc_regs[:, :] = 0
        
    @property
    def acc_dtype(self) -> np.dtype:
        return self._acc_dtype
    
    @property
    def dtype(self) -> np.dtype:
        return self._dtype
    
    @property
    def dataflow(self) -> MXUDataflow:
        return self._dataflow
        
    @property
    def pe_arr_shape(self) -> tuple[int, int]:
        return (self.pe_arr_height, self.pe_arr_width)
    
    @property
    def acc_regs_shape(self) -> tuple[int, int]:
        if self._acc_regs is None:
            return self.pe_arr_shape
        return (self.seq_len, self.pe_arr_width)
        
    @property
    def m_tile(self) -> int:
        if self._dataflow == MXUDataflow.OS:
            return self.pe_arr_height
        elif self._dataflow == MXUDataflow.WS:
            return self.seq_len
        else:
            raise Exception(f"[ERROR] Unsupported MXU dataflow: {self._dataflow}.")
        
    @property
    def n_tile(self) -> int:
        if self._dataflow == MXUDataflow.OS:
            return self.pe_arr_width
        elif self._dataflow == MXUDataflow.WS:
            return self.pe_arr_width
        else:
            raise Exception(f"[ERROR] Unsupported MXU dataflow: {self._dataflow}.")
        
    @property
    def k_tile(self) -> int:
        if self._dataflow == MXUDataflow.OS:
            return self.seq_len
        elif self._dataflow == MXUDataflow.WS:
            return self.pe_arr_height
        else:
            raise Exception(f"[ERROR] Unsupported MXU dataflow: {self._dataflow}.")
    
    @property
    def ifm_tile_numel(self) -> int:
        return self.m_tile * self.k_tile
    
    @property
    def wgt_tile_numel(self) -> int:
        return self.k_tile * self.n_tile
    
    @property
    def ofm_tile_numel(self) -> int:
        return self.m_tile * self.n_tile
    
    @property
    def ifm_tile_shape(self) -> tuple[int, int]:
        return (self.m_tile, self.k_tile)
    
    @property
    def wgt_tile_shape(self) -> tuple[int, int]:
        return (self.k_tile, self.n_tile)

    @property
    def ofm_tile_shape(self) -> tuple[int, int]:
        return (self.m_tile, self.n_tile)
    
    @property
    def ifm_tile_size(self) -> int:
        return self.ifm_tile_numel * self.dtype.itemsize
    
    @property
    def wgt_tile_size(self) -> int:
        return self.wgt_tile_numel * self.dtype.itemsize
    
    @property
    def ofm_tile_size(self) -> int:
        return self.ofm_tile_numel * self.acc_dtype.itemsize
