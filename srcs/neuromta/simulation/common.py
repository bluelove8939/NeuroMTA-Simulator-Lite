import enum


__all__ = [
    "K_UNIT",
    "M_UNIT",
    "G_UNIT",
    "T_UNIT",
    
    "SACoreDataflow",
    "SACoreOperatorType",
    "MemoryType",
    
    "parse_freq_str",
    "parse_mem_cap_str",
]


K_UNIT = 1024
M_UNIT = K_UNIT * 1024
G_UNIT = M_UNIT * 1024
T_UNIT = G_UNIT * 1024


class SACoreDataflow(enum.Enum):
    OS = enum.auto()
    WS = enum.auto()
    
    @classmethod
    def from_str(cls, expr: str):
        if expr == "OS":
            return cls.OS
        elif expr == "WS":
            return cls.WS
        else:
            raise Exception(f"[ERROR] Invalid dataflow expression '{expr}'")
        

class SACoreOperatorType(enum.Enum):
    PRELOAD = enum.auto()
    EXECUTE = enum.auto()
    FLUSH   = enum.auto()
    

class MemoryType(enum.Enum):
    L1   = enum.auto()
    L2   = enum.auto()
    
    
def parse_freq_str(expr: str) -> int:
    if expr.lower().endswith("khz"):
        expr = int(expr[:-3]) * K_UNIT
    elif expr.lower().endswith("mhz"):
        expr = int(expr[:-3]) * M_UNIT
    elif expr.lower().endswith("ghz"):
        expr = int(expr[:-3]) * G_UNIT
    elif expr.lower().endswith("hz"):
        expr = int(expr[:-2])
    else:
        try:
            expr = int(expr)
        except:
            raise Exception(f"[ERROR] Invalid frequency expression: {expr}")
    return expr


def parse_mem_cap_str(expr: str) -> int:
    if expr.lower().endswith("bytes"):
        expr = expr[:-4]
    elif expr.lower().endswith("byte"):
        expr = expr[:-3]
    
    if expr.lower().endswith("kb"):
        expr = int(expr[:-2]) * K_UNIT
    elif expr.lower().endswith("mb"):
        expr = int(expr[:-2]) * M_UNIT
    elif expr.lower().endswith("gb"):
        expr = int(expr[:-2]) * G_UNIT
    elif expr.lower().endswith("b"):
        expr = int(expr[:-1]) * G_UNIT
    else:
        try:
            expr = int(expr)
        except:
            raise Exception(f"[ERROR] Invalid memory capacity expression: {expr}")
    return expr