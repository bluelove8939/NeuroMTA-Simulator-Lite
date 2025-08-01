import numpy as np

from neuromta.common.parser_utils import parse_mem_cap_str


__all__ = [
    "VPUConfig",
    "VPUContext",
]


class VPUConfig(dict):
    def __init__(
        self, 
        
        vreg_len: int = parse_mem_cap_str("128B"),
        vreg_num: int = 32,
        vdtype: np.dtype = np.float32,
        
        vlen_max: int = 1024,
        vlen_min: int = 32,
        
        unary_op_latency: int = 2,
        arith_op_latency: int = 4,
    ):
        super().__init__()
        
        self["vreg_len"] = vreg_len
        self["vreg_num"] = vreg_num
        self["vdtype"] = np.dtype(vdtype)
        self["vlen_max"] = vlen_max
        self["vlen_min"] = vlen_min
        self["unary_op_latency"] = unary_op_latency
        self["arith_op_latency"] = arith_op_latency
        
    def create_context(self) -> "VPUContext":
        return VPUContext(**self)


class VPUContext:
    def __init__(
        self,
        
        vreg_len: int = parse_mem_cap_str("128B"),
        vreg_num: int = 32,
        vdtype: np.dtype = np.float32,
        
        vlen_max: int = 1024,
        vlen_min: int = 32,
        
        unary_op_latency: int = 2,
        arith_op_latency: int = 4,
    ):
        self._physical_vreg_len = vreg_len    # physical vector register length in bytes
        self._physical_vreg_num = vreg_num    # number of physical vector registers
        
        self.vlen_max = vlen_max    # maximum vector length in elements
        self.vlen_min = vlen_min    # minimum vector length in elements

        self.unary_op_latency = unary_op_latency
        self.arith_op_latency = arith_op_latency
        
        self._physical_vrf: np.ndarray = np.zeros((self._physical_vreg_num * self._physical_vreg_len,), dtype=np.uint8)
        
        self._vdtype:   np.dtype    = np.dtype(vdtype)
        self._vlen:     int         = self._physical_vreg_len // self._vdtype.itemsize
        self._n_vregs:  int         = self._physical_vreg_num
        
        self._vreg_view:    np.ndarray = self._physical_vrf.view(dtype=self._vdtype).reshape(self._n_vregs, self._vlen)

    def reconfigure_vector_reg_file(self, vlen: int, vdtype: np.dtype):
        if isinstance(vdtype, str):
            vdtype = np.dtype(vdtype)
            
        if vlen < self.vlen_min or vlen > self.vlen_max:
            raise Exception(f"[ERROR] Vector length {vlen} is out of bounds ({self.vlen_min}, {self.vlen_max}).")
            
        self._vdtype    = np.dtype(vdtype)
        self._vlen      = vlen
        self._n_vregs   = self._physical_vreg_len // (self._vlen * self._vdtype.itemsize)
        
        self._vreg_view = self._physical_vrf.view(dtype=self._vdtype).reshape(self._n_vregs, self._vlen)
        
    def set_vector_reg(self, vreg_idx: int, data: np.ndarray):
        if not isinstance(data, np.ndarray):
            raise Exception("[ERROR] Data must be a numpy array.")
        if data.dtype != self._vdtype:
            raise Exception(f"[ERROR] Data type {data.dtype} does not match vector register type {self._vdtype}.")
        if data.size != self._vlen:
            raise Exception(f"[ERROR] Data size {data.size} does not match vector length {self._vlen}.")
        if vreg_idx < 0 or vreg_idx >= self._n_vregs:
            raise Exception(f"[ERROR] Vector register index {vreg_idx} out of bounds (0, {self._n_vregs}).")

        self._vreg_view[vreg_idx, :] = data
        
    def get_vector_reg(self, vreg_idx: int) -> np.ndarray:
        if vreg_idx < 0 or vreg_idx >= self._n_vregs:
            raise Exception(f"[ERROR] Vector register index {vreg_idx} out of bounds (0, {self._n_vregs}).")
        
        return self._vreg_view[vreg_idx, :].copy()

    @property
    def vdtype(self) -> np.dtype:
        return self._vdtype
    
    @property
    def vlen(self) -> int:
        return self._vlen