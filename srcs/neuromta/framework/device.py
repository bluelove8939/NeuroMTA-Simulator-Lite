import os
import sys
import time
import signal
import traceback
import shutil
import multiprocessing as mp
from typing import Sequence, Callable, Any

from neuromta.framework.core import Core, Kernel, Command
from neuromta.framework.tracer import Tracer


__all__ = [
    "Device",
]


NEUROMTA_ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
DEFAULT_TRACE_DIR = os.path.join(NEUROMTA_ROOT_DIR, ".logs", "traces")


class _InternalDeviceCoreSignal:
    UPDATE_TIME_RSP = 0  # DEVICE -> CORE >> response to update the timestamp
    CORE_SYNC_REQ   = 1  # DEVICE -> CORE >> terminate the current process
    TERMINATE       = 2
    
    UPDATE_TIME_REQ = 3  # CORE -> DEVICE >> alert that the device the current remaining time and request for the timestamp update
    MAIN_IDLE       = 4  # CORE -> DEVICE >> alert that the current main process has been finished
    EXCEPTION       = 5  # CORE -> DEVICE >> alert that the exception occurred for the given core
    CORE_SYNC_RSP   = 6  # CORE -> DEVICE >> alert that the core has finished its execution and is ready for the core to be synchronized (send core states as a payload)

    def __init__(self, sig_id: int, core_id: Any, payload: Any=None):
        """
        Internal Device-Core Signal
        
        Description
            - To achieve full multi-processing with the NeuroMTA simulator, the device and core components must communicate effectively. This protocol defines 
            the signals exchanged between the device and core during their interactions.
            - Each signal has a unique identifier and a payload that carries the necessary information for the recipient to process the signal.
            
        Phase #1: Execution Phase
            - In this phase, the device transfers the timestamp update request to each cores with respect to the remaining core time given from the cores.
            - Each core send UPDATE_TIME_REQ to alert the current remaining time. (payload: remaining cycle time)
            - The device send UPDATE_TIME_RSP signal to every core to update the timestamp.
            - Once the core transfers MAIN_IDLE signal, the device will acknowledge the completion of the core's execution. The core still transfers the remaining
            cycle time as the payload.
            - Once the core transfers EXCEPTION signal, the device will automatically terminate all the core's execution.
            
        Phase #2: Synchronization Phase
            - After executing all the processes, every core should send its state to update the device.
            - After receiving all the MAIN_IDLE signals, the host device will transfer CORE_SYNC_REQ signals to each core to request their states.
            - Each core will respond with a CORE_SYNC_RSP signal containing its state information.
        
        Device Protocols
            - UPDATE_TIME_RSP   (0, None, <cycles to be updated>)
            - CORE_SYNC_REQ     (1, None, None)
            - TERMINATE         (2, None, None)
                
        Core Protocols
            - UPDATE_TIME_REQ   (3, <current core id>, <remaining cycle time>)
            - MAIN_IDLE         (4, <current core id>, <remaining cycle time>)
            - EXCEPTION         (5, <current core id>, None)
            - CORE_SYNC_REQ     (6, <current core id>, <core state>)
        """
    
        self.sig_id:  int = sig_id
        self.core_id: Any = core_id
        self.payload: Any = payload
    
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
    

class SingleCoreProcess(mp.Process):
    def __init__(
        self,
        device: 'Device', 
        core: Core, 
        core_to_device_sig_queue: mp.Queue,
        device_to_core_sig_queue: mp.Queue,
        max_steps: int = -1,
        save_trace: bool = False,
        save_trace_dir: str = DEFAULT_TRACE_DIR,
    ):

        super().__init__(name=None)
        
        self.device = device
        self.core = core
        self.core_to_device_sig_queue = core_to_device_sig_queue
        self.device_to_core_sig_queue = device_to_core_sig_queue
        self.max_steps = max_steps
        self.save_trace = save_trace
        self.save_trace_dir = save_trace_dir

        self.tracer = Tracer()
        if self.save_trace:
            self.tracer.register_core(self.core)
        
    def _on_sigterm(self, signum, frame):
        if self.save_trace and (not self.tracer.is_empty):
            core_id_str_expr = self.tracer.convert_valid_core_id(self.core.core_id)
            trace_path = os.path.join(self.save_trace_dir, f"{core_id_str_expr}.csv")
            self.tracer.save_traces_as_file(trace_path)
            print(f"[INFO] Trace for core {self.core.core_id} saved to \"{trace_path}\"")
        
        os._exit(143)  # 128 + 15
        
    def run(self):
        try:
            signal.signal(signal.SIGTERM, self._on_sigterm)
        except Exception:
            pass    # may cause faulty behavior for some OS including Windows ...
        
        step_cnt = 0

        self.core.initialize_mp_queue_inbox(rpc_req_send_inbox=self.device._rpc_req_send_inbox, rpc_rsp_send_inbox=self.device._rpc_rsp_send_inbox)

        is_main_finished = False
        is_exception_caught = False
        
        # Phase 1: Execution Phase
        while True:
            # STEP 1: Send timestamp update request based on the remaining cycles of the core
            try:
                remaining_cycles = self.core.get_remaining_cycles()
            except Exception as e:
                print(f"[ERROR] Failed to get remaining cycles for core {self.core.core_id}: {e}")
                print(traceback.format_exc())
                remaining_cycles = None
                is_exception_caught = True

            if self.core.is_idle_main and not is_main_finished:
                sig_id = _InternalDeviceCoreSignal.MAIN_IDLE
                is_main_finished = True
            elif is_exception_caught:
                sig_id = _InternalDeviceCoreSignal.EXCEPTION
            else:
                sig_id = _InternalDeviceCoreSignal.UPDATE_TIME_REQ
            
            core_sig = _InternalDeviceCoreSignal(
                sig_id=sig_id,
                core_id=self.core.core_id,
                payload=remaining_cycles
            )
            
            self.core_to_device_sig_queue.put(core_sig)
            
            if is_exception_caught:
                break
            
            # STEP 2: Receive device signal
            #   - If the device signal is UPDATE_TIME_RSP, update the cycle time and goto STEP 1 again
            #   - If the device signal is CORE_SYNC_REQ, terminate the while loop and goto Phase 2
            device_sig: _InternalDeviceCoreSignal = self.device_to_core_sig_queue.get()
            
            if device_sig == _InternalDeviceCoreSignal.CORE_SYNC_REQ:
                break
            elif device_sig == _InternalDeviceCoreSignal.UPDATE_TIME_RSP:
                cycle_time: int = device_sig.payload
            
                try:
                    self.core.update_cycle_time(cycle_time=cycle_time)

                    if self.max_steps > 0 and step_cnt >= self.max_steps:
                        print(f"[INFO] Reached maximum steps: {self.max_steps}. Stopping execution.")
                        break
                    
                    step_cnt += 1
                    
                except Exception as e:
                    print(f"[ERROR] Exception occurred in core {self.core.core_id}: {e}")
                    print(traceback.format_exc())
                    is_exception_caught = True
            else:
                print(f"[ERROR] Unexpected device signal: {device_sig}")
                is_exception_caught = True
        
        # Phase 2: Synchronization Phase
        core_sig = _InternalDeviceCoreSignal(
            sig_id=_InternalDeviceCoreSignal.CORE_SYNC_RSP,
            core_id=self.core.core_id,
            payload=self.core.__getstate__()
        )
        
        self.core_to_device_sig_queue.put(core_sig)
        
        device_sig = self.device_to_core_sig_queue.get()
        
        if device_sig == _InternalDeviceCoreSignal.TERMINATE:
            return
        else:
            print(f"[ERROR] Unexpected device signal: {device_sig}")
            return


class Device:
    def __init__(self):
        self._cores:        dict[str, Core] = None
        
        self._verbose:      bool = False
        self.create_trace:  bool = False

        self._rpc_req_send_inbox: dict[str, mp.Queue] = {}
        self._rpc_rsp_send_inbox: dict[str, mp.Queue] = {}
        
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
    
    def default_command_debug_hook(self, core: Core, kernel: Kernel, cmd: Command):
        if self._verbose:
            sys.stdout.write(f"[DEBUG] #{core.timestamp:<5d} | core: {core.core_id.__str__():<12s} | kernel: {kernel.kernel_id:<36s} | command: {cmd.cmd_id:<34s}\n")
            
    def run_kernels(self, verbose: bool=False, max_steps: int=-1, save_trace: bool=False, save_trace_dir: str=DEFAULT_TRACE_DIR):
        if not self.is_initialized:
            raise Exception("[ERROR] Device is not initialized. Please call initialize() before using this method.")
        
        self._verbose = verbose

        core_ids:                   list[str]                       = list(self._cores.keys())
        core_process_dict:          dict[str, SingleCoreProcess]    = {}
        core_to_device_sig_queue:   mp.Queue                        = mp.Queue()
        device_to_core_sig_q_dict:  dict[str, mp.Queue]             = {core_id: mp.Queue() for core_id in core_ids}
        main_idle_flags:            dict[str, bool]                 = {core_id: False for core_id in core_ids}
        core_sync_flags:            dict[str, bool]                 = {core_id: False for core_id in core_ids}
        exception_flag:             bool                            = False

        if save_trace:
            if os.path.isdir(save_trace_dir):
                shutil.rmtree(save_trace_dir)  # Remove existing directory
            os.makedirs(save_trace_dir, exist_ok=True)

        for core_id in core_ids:
            core = self._cores[core_id]
            p = SingleCoreProcess(
                device=self, core=core, core_to_device_sig_queue=core_to_device_sig_queue, device_to_core_sig_queue=device_to_core_sig_q_dict[core_id], 
                max_steps=max_steps, save_trace=save_trace, save_trace_dir=save_trace_dir
            )
            core_process_dict[core_id] = p

        for p in core_process_dict.values():
            p.start()
        
        while True:
            timestamp_update_req_flag: dict[str, bool] = {core_id: False for core_id in core_ids}
            remaining_cycles_min: int = None
            
            while not all(timestamp_update_req_flag.values()):
                sig: _InternalDeviceCoreSignal = core_to_device_sig_queue.get()

                if sig == _InternalDeviceCoreSignal.MAIN_IDLE:
                    core_id = sig.core_id
                    main_idle_flags[core_id] = True
                    timestamp_update_req_flag[core_id] = True
                    
                    if remaining_cycles_min is None:
                        remaining_cycles_min = sig.payload
                    else:
                        if sig.payload is not None:  # if the payload is None, it means that the core is not requesting a timestamp update
                            remaining_cycles_min = min(remaining_cycles_min, sig.payload)

                elif sig == _InternalDeviceCoreSignal.EXCEPTION:
                    core_id = sig.core_id
                    print(f"[ERROR] Exception occurred in core {core_id}. Stopping execution.")
                    exception_flag = True
                    break
                
                elif sig == _InternalDeviceCoreSignal.UPDATE_TIME_REQ:
                    core_id = sig.core_id
                    timestamp_update_req_flag[core_id] = True
                    
                    if remaining_cycles_min is None:
                        remaining_cycles_min = sig.payload
                    else:
                        if sig.payload is not None:  # if the payload is None, it means that the core is not requesting a timestamp update
                            remaining_cycles_min = min(remaining_cycles_min, sig.payload)
            
            if exception_flag:
                break
            
            if all(main_idle_flags.values()):
                break
            
            if remaining_cycles_min == 0 or remaining_cycles_min is None:
                remaining_cycles_min = 1
            
            for core_id, q in device_to_core_sig_q_dict.items():
                device_sig = _InternalDeviceCoreSignal(
                    sig_id=_InternalDeviceCoreSignal.UPDATE_TIME_RSP,
                    core_id=core_id,
                    payload=remaining_cycles_min
                )
                q.put(device_sig)
        
        if not exception_flag:
            for core_id in core_ids:
                device_sig = _InternalDeviceCoreSignal(
                    sig_id=_InternalDeviceCoreSignal.CORE_SYNC_REQ,
                    core_id=core_id,
                )
                device_to_core_sig_q_dict[core_id].put(device_sig)
                
            while not all(core_sync_flags.values()):
                sig: _InternalDeviceCoreSignal = core_to_device_sig_queue.get()
                
                if sig == _InternalDeviceCoreSignal.CORE_SYNC_RSP:
                    core_id = sig.core_id
                    payload = sig.payload
                    core = self._cores.get(core_id)
                    
                    if core is None:
                        raise Exception(f"[ERROR] Core with ID '{core_id}' not found.")

                    core.__setstate__(payload)
                    core_sync_flags[core_id] = True
                    
                else:
                    print(f"[ERROR] Failed to synchronize the core states since the device received invalid signal '{sig}' from the core. (the simulation result may be inconsistent)")
                    break
        else:
            print(f"[ERROR] Skip synchronizing core states since unexpected exception occurred.")

        for p in core_process_dict.values():
            if p.is_alive():
                p.terminate()
                
        for p in core_process_dict.values():
            p.join(timeout=0.1)

    def register_command_debug_hook(self, hook: Callable):
        if not self.is_initialized:
            raise Exception("[ERROR] Device is not initialized. Please call initialize() before using this method.")
        
        for core in self._cores.values():
            core.register_command_debug_hook(hook)
            
    @property
    def verbose(self) -> bool:
        return self._verbose
    
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