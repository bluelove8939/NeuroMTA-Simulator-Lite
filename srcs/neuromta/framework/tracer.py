from typing import Any

from neuromta.framework.core import *


__all__ = [
    "TraceEntry",
    "Tracer",
]


class TraceEntry:
    def __init__(self, timestamp: int, core_id: Any, kernel_id: str, cmd_id: str):
        self.timestamp = timestamp
        self.core_id = core_id
        self.kernel_id = kernel_id
        self.cmd_id = cmd_id
    
    def __getstate__(self):
        return {
            "timestamp": self.timestamp,
            "core_id": self.core_id,
            "kernel_id": self.kernel_id,
            "cmd_id": self.cmd_id,
        }
        
    def __setstate__(self, state: dict):
        self.timestamp = state["timestamp"]
        self.core_id = state["core_id"]
        self.kernel_id = state["kernel_id"]
        self.cmd_id = state["cmd_id"]

    def to_csv_entry(self) -> str:
        return f"{self.timestamp},{self.core_id},{self.kernel_id},{self.cmd_id}"
    
    def __str__(self):
        return f"CommandTrace(timestamp={self.timestamp}, core_id={self.core_id}, kernel_id={self.kernel_id}, cmd_id={self.cmd_id})"


class Tracer:
    def __init__(self):
        self._trace_entries: list[TraceEntry] = []
        self._debug_hook_handles: dict[str, str] = {}
    
    @staticmethod
    def convert_valid_core_id(core_id: Any) -> str:
        core_id: str = core_id.__str__()
        core_id = core_id.replace("/", "_").replace("\\", "_").replace(" ", "").replace(",", "_").replace("(", "").replace(")", "")
        return core_id
        
    def __getstate__(self):
        return {
            "_trace_entries": self._trace_entries,
        }

    def __setstate__(self, state: dict):
        self._trace_entries = state["_trace_entries"]

    def core_debug_hook(self, core: Core, kernel: Kernel, cmd: Command):
        entry = TraceEntry(
            timestamp=core.total_timestamp,
            core_id=self.convert_valid_core_id(core.core_id),
            kernel_id=kernel.kernel_id,
            cmd_id=cmd.cmd_id,
        )
        self.add_trace(entry)
        
    def register_core(self, core: Core):
        hook = core.register_command_debug_hook(self.core_debug_hook)
        self._debug_hook_handles[core.core_id] = hook
        
    def unregister_core(self, core: Core):
        hook = self._debug_hook_handles.pop(core.core_id, None)
        if hook is not None:
            core.unregister_command_debug_hook(hook)
    
    def add_trace(self, trace: TraceEntry):
        self._trace_entries.append(trace)
            
    def save_traces_as_file(self, filename: str):
        with open(filename, "wt") as file:
            file.write("timestamp,core_id,kernel_id,command_id\n")
            for entry in self._trace_entries:
                file.write(entry.to_csv_entry() + "\n")
                
    def clear_traces(self):
        self._trace_entries.clear()
        
    @property
    def is_empty(self) -> bool:
        return len(self._trace_entries) == 0
