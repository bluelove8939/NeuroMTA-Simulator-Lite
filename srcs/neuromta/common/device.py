from typing import Sequence

from neuromta.common.core import Core, Command

class Device:
    def __init__(self):
        self._cores:        dict[str, Core] = None
        self._timestamp:    int             = 0
        
    def _register_core(self, name: str, core: Core | Sequence[Core]):
        if isinstance(core, Sequence):
            for idx, item in enumerate(core):
                if isinstance(item, (Core, Sequence)):
                    self._register_core(f"{name}[{idx}]", item)
        elif isinstance(core, Core):
            self._cores[name] = core
            core.core_id = name
            core.register_command_debug_hook(self.default_command_debug_hook)

    def initialize(self):
        self._cores = {}
        
        for name, core in self.__dict__.items():
            if isinstance(core, (Core, Sequence)):
                self._register_core(name, core)

        self._timestamp = 0
        
        return self
        
    def run_kernels(self):
        if not self.is_initialized:
            raise Exception("[ERROR] Device is not initialized. Please call initialize() before using this method.")
        
        while not self.is_idle:
            cycles = self.get_remaining_cycles()
            self.update_cycle_time(cycles)
            
            self._timestamp += cycles

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
    
    def default_command_debug_hook(self, cmd: Command):
        print(f"[DEBUG] #{self.timestamp:<5d} | core: {cmd.core.core_id:<24s} | command: {cmd.cmd_id:<24s} args:({', '.join(map(str, cmd.args))}) kwargs:{{{', '.join(f'{k}={v}' for k, v in cmd.kwargs.items())}}}")
    
    @property
    def timestamp(self) -> int:
        return self._timestamp
        
    @property
    def is_initialized(self) -> bool:
        return self._cores is not None

    @property
    def is_idle(self) -> bool:
        return all(core.is_idle for core in self._cores.values())