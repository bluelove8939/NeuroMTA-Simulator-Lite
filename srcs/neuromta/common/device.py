import sys
from typing import Sequence, Callable

from neuromta.common.core import Core, Command, DEFAULT_CORE_ID


__all__ = [
    "CommandTrace",
    "Device",
]


class CommandTrace:
    def __init__(self, timestamp: int, core_id: str, kernel_id: str, command_id: str):
        self.timestamp = timestamp
        self.core_id = core_id
        self.kernel_id = kernel_id
        self.command_id = command_id
        
    def __str__(self):
        return f"{self.timestamp},{self.core_id},{self.kernel_id},{self.command_id}"


class Device:
    def __init__(self):
        self._cores:        dict[str, Core] = None
        self._timestamp:    int             = 0

        self.verbose:       bool = False
        self.create_trace:  bool = False
        
        self._traces: list[CommandTrace] = []
        
    def change_sim_model_options(self, use_cycle_model: bool = None, use_functional_model: bool = None):
        for core in self._cores.values():
            core.change_sim_model_options(use_cycle_model, use_functional_model)
        
    def _register_core(self, name: str, core: Core | Sequence[Core]):
        if isinstance(core, Sequence):
            for idx, item in enumerate(core):
                if isinstance(item, (Core, Sequence)):
                    self._register_core(f"{name}[{idx}]", item)
        elif isinstance(core, Core):
            if core.core_id == DEFAULT_CORE_ID:
                core.core_id = name
            if core.core_id in self._cores.keys():
                raise Exception(f"[ERROR] Core with ID '{core.core_id}' already exists. Please use a unique core ID.")
            self._cores[core.core_id] = core
            core.register_command_debug_hook(self.default_command_debug_hook)

    def initialize(self, create_trace: bool = None):
        self._cores = {}
        
        for name, core in self.__dict__.items():
            if isinstance(core, (Core, Sequence)):
                self._register_core(name, core)

        self._timestamp = 0
        
        if create_trace is not None and isinstance(create_trace, bool):
            self.create_trace = create_trace
        
        return self
        
    def run_kernels(self, max_steps: int = -1):
        if not self.is_initialized:
            raise Exception("[ERROR] Device is not initialized. Please call initialize() before using this method.")
        
        while not self.is_idle:
            cycles = self.get_remaining_cycles()
            self.update_cycle_time(cycles)
            
            self._timestamp += cycles
            
            if max_steps > 0 and self._timestamp >= max_steps:
                print(f"[INFO] Reached maximum steps: {max_steps}. Stopping execution.")
                break

    def get_remaining_cycles(self) -> int:
        if not self.is_initialized:
            raise Exception("[ERROR] Device is not initialized. Please call initialize() before using this method.")
        
        return min(map(lambda x: x.get_remaining_cycles(), self._cores.values()))
    
    def update_cycle_time(self, cycles: int):
        if not self.is_initialized:
            raise Exception("[ERROR] Device is not initialized. Please call initialize() before using this method.")
        
        self._timestamp += cycles
        
        for core in self._cores.values():
            core.update_cycle_time(cycles)
            
    def register_command_debug_hook(self, hook: Callable):
        if not self.is_initialized:
            raise Exception("[ERROR] Device is not initialized. Please call initialize() before using this method.")
        
        for core in self._cores.values():
            core.register_command_debug_hook(hook)
    
    def default_command_debug_hook(self, cmd: Command):
        if self.verbose:
            sys.stdout.write(f"[DEBUG] #{self.timestamp:<5d} | core: {cmd.core_id:<24s} | kernel: {cmd.kernel_id:<24s} | command: {cmd.cmd_id:<34s}\n")
        
        if self.create_trace:
            entry = CommandTrace(
                timestamp=self._timestamp,
                core_id=cmd.core_id,
                kernel_id=cmd.kernel_id,
                command_id=cmd.cmd_id
            )
            
            self._traces.append(entry)
            
    def save_traces(self, filename: str):
        if not self.create_trace:
            print(f"[WARNING] Command tracing is not enabled. No traces to save.")
            
        with open(filename, "wt") as file:
            file.write("timestamp,core_id,kernel_id,command_id\n")
            for trace in self._traces:
                file.write(str(trace) + "\n")
                
    def clear_traces(self):
        if not self.create_trace:
            print(f"[WARNING] Command tracing is not enabled. No traces to clear.")
            
        self._traces.clear()
            
    @property
    def timestamp(self) -> int:
        return self._timestamp
        
    @property
    def is_initialized(self) -> bool:
        return self._cores is not None

    @property
    def is_idle(self) -> bool:
        return all(core.is_idle for core in self._cores.values())