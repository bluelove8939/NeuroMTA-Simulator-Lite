import sys
import time
import enum
import traceback
import multiprocessing as mp
from typing import Sequence, Callable, Any

from neuromta.framework.core import Core, Command, Kernel


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
    
    
class _InternalDeviceCoreSignal:
    MAIN_SYNC = 0  # CORE -> DEVICE >> alert that the current main process has been finished and need to be synchronized
    EXCEPTION = 1  # CORE -> DEVICE >> alert that the exception occurred for the given core
    TERMINATE = 2  # DEVICE -> CORE >> terminate the current process
    
    def __init__(self, sig_id: int, core_id: Any, payload: dict[str, Any]=None):
        self.sig_id:  int = sig_id
        self.core_id: Any = core_id
        self.payload: dict[str, Any] = {} if payload is None else payload  # additional payload for the signal, e.g., core states when TERMINATE
    
    def __getstate__(self):
        return {
            "sig_id": self.sig_id,
            "core_id": self.core_id,
            "payload": self.payload,
        }

    def __setstate__(self, state: dict[str, Any]):
        self.sig_id = state["sig_id"]
        self.core_id = state["core_id"]
        self.payload = state["payload"]
        
    def __str__(self):
        return f"InternalSignal({self.sig_id}, {self.core_id})"
    
    def __eq__(self, value):
        if isinstance(value, int):
            return self.sig_id == value
        elif isinstance(value, _InternalDeviceCoreSignal):
            return self.sig_id == value.sig_id
        return False


class Device:
    def __init__(self):
        self._cores:        dict[str, Core] = None
        
        self.verbose:       bool = False
        self.create_trace:  bool = False

        self._rpc_req_send_inbox: dict[str, mp.Queue] = {}
        self._rpc_rsp_send_inbox: dict[str, mp.Queue] = {}

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
            if core.core_id in self._cores.keys():
                raise Exception(f"[ERROR] Core with ID '{core.core_id}' already exists. Please use a unique core ID.")
            self._cores[core.core_id] = core
            core.register_command_debug_hook(self.default_command_debug_hook)

    def initialize(self, create_trace: bool = None):
        self._cores = {}
        
        for name, core in self.__dict__.items():
            if isinstance(core, (Core, Sequence)):
                self._register_core(name, core)

        self._rpc_req_send_inbox = {core.core_id: mp.Queue() for core in self._cores.values()}
        self._rpc_rsp_send_inbox = {core.core_id: mp.Queue() for core in self._cores.values()}

        for core in self._cores.values():
            core.initialize_kernel_dispatch_queue()
            core.initialize_mp_queue_inbox(rpc_req_send_inbox=self._rpc_req_send_inbox, rpc_rsp_send_inbox=self._rpc_rsp_send_inbox)
        
        if create_trace is not None and isinstance(create_trace, bool):
            self.create_trace = create_trace
        
        return self
            
    def run_kernels(self, cycle_level: bool=True, max_steps: int=-1):
        if not self.is_initialized:
            raise Exception("[ERROR] Device is not initialized. Please call initialize() before using this method.")

        core_ids = list(self._cores.keys())
        core_main_process_list:         dict[str, mp.Process]   = {}
        core_to_device_sig_queue:       mp.Queue                = mp.Queue()
        main_sync_flags:                dict[str, bool]         = {core_id: False for core_id in core_ids}

        def _single_core_run_kernels(
            device: Device, 
            core: Core, 
            core_to_device_sig_queue: mp.Queue, 
            max_steps: int = -1,
        ):
            step_cnt = 0

            core.initialize_mp_queue_inbox(rpc_req_send_inbox=device._rpc_req_send_inbox, rpc_rsp_send_inbox=device._rpc_rsp_send_inbox)

            is_main_synchronized = False
            is_exception_caught = False

            while True:
                if core.is_idle_main:
                    if not is_main_synchronized:
                        sig = _InternalDeviceCoreSignal(
                            sig_id=_InternalDeviceCoreSignal.MAIN_SYNC,
                            core_id=core.core_id,
                            payload=core.__getstate__()
                        )
                        core_to_device_sig_queue.put(sig)
                        is_main_synchronized = True
                        
                    time.sleep(0.01)  # Avoid busy waiting
                    
                if is_exception_caught:
                    time.sleep(1)  # Avoid busy waiting
                    continue
                
                try:
                    core.update_cycle_time(cycle_level=cycle_level)

                    if max_steps > 0 and step_cnt >= max_steps:
                        print(f"[INFO] Reached maximum steps: {max_steps}. Stopping execution.")
                        break
                    
                    step_cnt += 1
                except Exception as e:
                    print(f"[ERROR] Exception occurred in core {core.core_id}: {e}")
                    print(traceback.format_exc())

                    sig = _InternalDeviceCoreSignal(
                        sig_id=_InternalDeviceCoreSignal.EXCEPTION,
                        core_id=core.core_id,
                        payload=None
                    )
                    core_to_device_sig_queue.put(sig)
                    
                    # NOTE: Instead of terminating the process, give the host a chance to receive EXCEPTION signal and handle with the exception. 
                    # The host terminates all the child processes by default. If 'is_exception_caught' is set to 'True', the loop skips updating
                    # the cycle time and waits until the host device handles the exception.
                    is_exception_caught = True

        for core_id in core_ids:
            core = self._cores[core_id]
            p = mp.Process(
                target=_single_core_run_kernels, 
                args=(self, core, core_to_device_sig_queue, max_steps)
            )
            core_main_process_list[core_id] = p

        for p in core_main_process_list.values():
            p.start()
            
        cnt = 0
        
        while not all(main_sync_flags.values()):
            sig: _InternalDeviceCoreSignal = core_to_device_sig_queue.get()

            if sig == _InternalDeviceCoreSignal.MAIN_SYNC:
                core_id = sig.core_id
                payload = sig.payload
                core = self._cores.get(core_id)
                
                if core is None:
                    raise Exception(f"[ERROR] Core with ID '{core_id}' not found.")
                
                core.__setstate__(payload)
                main_sync_flags[core_id] = True
                cnt += 1
                    
            elif sig == _InternalDeviceCoreSignal.EXCEPTION:
                core_id = sig.core_id
                print(f"[ERROR] Exception occurred in core {core_id}. Stopping execution.")
                break
            
        for p in core_main_process_list.values():
            if p.is_alive():
                p.terminate()
        for p in core_main_process_list.values():
            p.join(timeout=0.1)

    def register_command_debug_hook(self, hook: Callable):
        if not self.is_initialized:
            raise Exception("[ERROR] Device is not initialized. Please call initialize() before using this method.")
        
        for core in self._cores.values():
            core.register_command_debug_hook(hook)
    
    def default_command_debug_hook(self, core: Core, kernel: Kernel, cmd: Command):
        if self.verbose:
            sys.stdout.write(f"[DEBUG] #{core.total_timestamp:<5d} | core: {core.core_id.__str__():<12s} | kernel: {kernel.kernel_id:<36s} | command: {cmd.cmd_id:<34s}\n")

        if self.create_trace:
            entry = CommandTrace(
                timestamp=core.timestamp,
                core_id=core.core_id,
                kernel_id=kernel.kernel_id,
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