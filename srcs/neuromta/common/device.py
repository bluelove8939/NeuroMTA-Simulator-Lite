import sys
import multiprocessing as mp
from typing import Sequence, Callable

from neuromta.common.core import Core, Command


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
            # if core.core_id == DEFAULT_CORE_ID:
            #     # core.core_id = name
            #     raise ValueError(f"[ERROR] Default core ID is deprecated to support RPC framework.")
            if core.core_id in self._cores.keys():
                raise Exception(f"[ERROR] Core with ID '{core.core_id}' already exists. Please use a unique core ID.")
            self._cores[core.core_id] = core
            core.register_command_debug_hook(self.default_command_debug_hook)

    def initialize(self, create_trace: bool = None):
        self._cores = {}
        
        for name, core in self.__dict__.items():
            if isinstance(core, (Core, Sequence)):
                self._register_core(name, core)

        rpc_req_send_inbox = {core.core_id: core.rpc_req_recv_queue for core in self._cores.values()}
        rpc_rsp_send_inbox = {core.core_id: core.rpc_rsp_recv_queue for core in self._cores.values()}
        
        for core in self._cores.values():
            core.initialize(rpc_req_send_inbox=rpc_req_send_inbox, rpc_rsp_send_inbox=rpc_rsp_send_inbox)
        
        if create_trace is not None and isinstance(create_trace, bool):
            self.create_trace = create_trace
        
        return self
    
    def run_kernels(self, max_steps: int = -1):
        if not self.is_initialized:
            raise Exception("[ERROR] Device is not initialized. Please call initialize() before using this method.")
        
        step_cnt = 0

        while not self.is_idle:
            for core in self._cores.values():
                core.update_cycle_time()

            if max_steps > 0 and step_cnt >= max_steps:
                print(f"[INFO] Reached maximum steps: {max_steps}. Stopping execution.")
                break
            
            step_cnt += 1
            
    def run_kernels_mp(self, max_steps: int = -1):
        if not self.is_initialized:
            raise Exception("[ERROR] Device is not initialized. Please call initialize() before using this method.")

        def _single_core_run_kernels(core: Core, max_steps: int = -1):
            step_cnt = 0

            while not core.is_idle:
                core.update_cycle_time()

                if max_steps > 0 and step_cnt >= max_steps:
                    print(f"[INFO] Reached maximum steps: {max_steps}. Stopping execution.")
                    break
                
                step_cnt += 1
        
        processes: list[mp.Process] = []
        for core in self._cores.values():
            p = mp.Process(target=_single_core_run_kernels, args=(core, max_steps))
            processes.append(p)
            
        for p in processes:
            p.start()
        
        for p in processes:
            p.join()
            
    def register_command_debug_hook(self, hook: Callable):
        if not self.is_initialized:
            raise Exception("[ERROR] Device is not initialized. Please call initialize() before using this method.")
        
        for core in self._cores.values():
            core.register_command_debug_hook(hook)
    
    def default_command_debug_hook(self, core: Core, cmd: Command):
        if self.verbose:
            sys.stdout.write(f"[DEBUG] #{self.timestamp:<5d} | core: {core.core_id.__str__():<24s} | kernel: {cmd.kernel_id:<24s} | command: {cmd.cmd_id:<34s}\n")

        if self.create_trace:
            entry = CommandTrace(
                timestamp=core.timestamp,
                core_id=core.core_id,
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
        t = [core.timestamp for core in self._cores.values()]
        return max(t) if t else 0

    @property
    def is_initialized(self) -> bool:
        return self._cores is not None

    @property
    def is_idle(self) -> bool:
        return all(core.is_idle for core in self._cores.values())