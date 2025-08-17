import enum
import math
import multiprocessing as mp
from typing import Callable, Sequence, Any


__all__ = [
    "set_global_context",
    "get_global_core_context",
    "get_global_kernel_context",
    "get_global_pid",
    
    "Command",
    
    "Kernel",
    
    "CoreCycleModel",
    "Core",
    
    "core_kernel_method",
    "core_command_method",
    "new_parallel_thread",
    "start_parallel_thread",
    "end_parallel_thread",
    
    # "DEFAULT_CORE_ID"
]


MAX_COMMAND_NUM_PER_KERNEL = 2 ** 20
# DEFAULT_CORE_ID = "__DEFAULT"
# NEGLECT = "__NEGLECT"


#################################################
# Global Context Management
#################################################

class GlobalContextMode(enum.Enum):
    IDLE    = enum.auto()
    COMPILE = enum.auto()
    EXECUTE = enum.auto()
    
_context_mode: GlobalContextMode = GlobalContextMode.IDLE
_core_context: 'Core' = None
_kernel_context: 'Kernel' = None
_parent_kernel_callstack: list[tuple['GlobalContextMode', 'Core', 'Kernel']] = []
    
class new_global_context:
    def __init__(self, context_mode: GlobalContextMode, core_context: 'Core' = None, kernel_context: 'Kernel' = None):
        self.context_mode = context_mode
        self.core_context = core_context
        self.kernel_context = kernel_context
        
        self._history_context_mode   = None
        self._history_core_context   = None
        self._history_kernel_context = None

    def __enter__(self):
        self._history_context_mode   = get_global_context_mode()
        self._history_core_context   = get_global_core_context()
        self._history_kernel_context = get_global_kernel_context()
        
        set_global_context(self.context_mode, self.core_context, self.kernel_context)

    def __exit__(self, exc_type, exc_value, traceback):
        set_global_context(self._history_context_mode, self._history_core_context, self._history_kernel_context)

def set_global_context(context_mode: GlobalContextMode, core: 'Core', kernel: 'Kernel | str'):
    global _core_context, _kernel_context, _context_mode
    
    if isinstance(context_mode, str):
        context_mode = GlobalContextMode.__members__.get(context_mode.upper())
    if context_mode == GlobalContextMode.COMPILE and not isinstance(kernel, Kernel):
        raise Exception(f"[ERROR] Cannot set global context to COMPILE mode with a non-Kernel object: {kernel}")
    
    _context_mode = context_mode
    _core_context = core
    _kernel_context = kernel
    
def get_global_context_mode() -> GlobalContextMode:
    global _context_mode
    return _context_mode

def get_global_core_context() -> 'Core':
    global _core_context
    return _core_context

def get_global_kernel_context() -> 'Kernel':
    global _kernel_context
    return _kernel_context

def get_global_pid() -> str:
    cid = get_global_core_context().core_id
    kctx = get_global_kernel_context()
    kid = kctx.kernel_id if isinstance(kctx, Kernel) else kctx
    return f"{cid}.{kid}"   # this is the global PID format: <core_id>.<root kernel_id>

def store_global_parent_kernel_callstack():
    global _parent_kernel_callstack
    
    context_mode = get_global_context_mode()
    core_context = get_global_core_context()
    kernel_context = get_global_kernel_context()
    
    if context_mode != GlobalContextMode.COMPILE:
        raise Exception(f"[ERROR] Cannot add to parent kernel callstack since the global context mode is not COMPILE, but {context_mode.name}")

    history = (context_mode, core_context, kernel_context)
    _parent_kernel_callstack.append(history)

def restore_global_parent_kernel_callstack() -> 'Kernel':
    global _parent_kernel_callstack
    
    if len(_parent_kernel_callstack) == 0:
        raise Exception("[ERROR] Cannot pop from parent kernel callstack since it is empty")

    history = _parent_kernel_callstack.pop()
    set_global_context(*history)
    
def get_global_current_parent_kernel_callstack() -> tuple[GlobalContextMode, 'Core', 'Kernel']:
    global _parent_kernel_callstack
    
    if len(_parent_kernel_callstack) == 0:
        raise Exception("[ERROR] Cannot get current parent kernel callstack since it is empty")
    
    return _parent_kernel_callstack[-1]


#################################################
# Decorators for Command and Kernel Methods
#################################################

def core_kernel_method(_func: Callable):
    def __core_kernel_method_wrapper(_core: 'Core', *_args, **_kwargs) -> Kernel:
        if not isinstance(_core, Core):
            raise Exception(f"[ERROR] Command method '{_func.__name__}' can only be called on an instance of Core")
        
        kernel = Kernel(
            _func.__name__,     # the kernel ID is the name of the function
            _func,              # the behavioral model is the function itself
            _core,              # the core on which this kernel is registered
            *_args,             # the arguments of the kernel
            **_kwargs           # the keyword arguments of the kernel
        )
        
        if get_global_context_mode() == GlobalContextMode.IDLE:
            _core.dispatch_kernel(kernel)
        elif get_global_context_mode() == GlobalContextMode.COMPILE:
            kernel_context = get_global_kernel_context()
            
            if kernel_context is None:
                raise Exception(f"[ERROR] Cannot register kernel '{kernel.kernel_id}' to the compiled kernel since it is called outside of a low-level kernel function")
            
            kernel_context.add_execution_step(kernel)
        else:
            print(f"[WARNING] Kernel method '{_func.__name__}' is called outside of the compile or idle context. It implies that the kernel is called inside the command execution context, which is strictly prohibited. This is mainly because of the faulty implementation of the command method.")
            raise Exception(f"[ERROR] Kernel method '{_func.__name__}' is called outside of the compile or idle context.")
        
        return kernel
        
    return __core_kernel_method_wrapper

def core_command_method(_func: Callable):
    def __core_command_method_wrapper(_core: 'Core', *_args, **_kwargs) -> Command:
        if get_global_context_mode() in (GlobalContextMode.IDLE, GlobalContextMode.EXECUTE):
            return _func(_core, *_args, **_kwargs)

        if not isinstance(_core, Core):
            raise Exception(f"[ERROR] Command method '{_func.__name__}' can only be called on an instance of Core")
        
        kernel_context = get_global_kernel_context()
        
        if kernel_context is None:
            raise Exception(f"[ERROR] Cannot register command '{_func.__name__}' to the compiled kernel since it is called outside of a low-level kernel function")
        elif not isinstance(kernel_context, Kernel):
            raise Exception(f"[ERROR] Cannot register command '{_func.__name__}' to the compiled kernel since it is called outside of a low-level kernel function. The current kernel context is not an instance of Kernel, but {type(kernel_context).__name__}")

        cmd = Command(
            kernel_context.kernel_id,   # the kernel context in which this command is registered (kernel ID given instead of the Kernel object itself. This is because the Command should be hashable ...)
            _func.__name__,             # the command ID is the name of the function
            *_args,                     # the arguments of the command
            **_kwargs                   # the keyword arguments of the command
        )
        
        if get_global_context_mode() == GlobalContextMode.COMPILE:
            kernel_context.add_execution_step(cmd)
        else:
            print(f"[WARNING] Command method '{_func.__name__}' is called outside of the compile or idle context. It implies that the command is called inside the command execution context, which is strictly prohibited. This is mainly because of the faulty implementation of the command method.")
            raise Exception(f"[ERROR] Command method '{_func.__name__}' is called outside of the compile or idle context.")
        
        return cmd
    return __core_command_method_wrapper

class new_parallel_thread:
    def __enter__(self):
        start_parallel_thread()
    
    def __exit__(self, exc_type, exc_value, traceback):
        end_parallel_thread()

def start_parallel_thread():
    if get_global_context_mode() != GlobalContextMode.COMPILE:
        raise Exception("[ERROR] Cannot create a parallel kernel since the global context mode is not COMPILE")
    
    kernel_context = get_global_kernel_context()
    parallel_kernel = kernel_context.add_parallel_kernel_step()
    
    store_global_parent_kernel_callstack()
    set_global_context(GlobalContextMode.COMPILE, get_global_core_context(), parallel_kernel)
    
def end_parallel_thread():
    if get_global_context_mode() != GlobalContextMode.COMPILE:
        raise Exception("[ERROR] Cannot end parallel kernel since the global context mode is not COMPILE")

    restore_global_parent_kernel_callstack()


#################################################
# Implementation
#################################################

class RPCMessage:
    def __init__(self, msg_type: int, src_core_id: str, dst_core_id: str, kernel_id: str, cmd_id: str, *args, **kwargs):
        self.msg_type = msg_type            # 0 for request, 1 for response
        self.src_core_id = src_core_id
        self.dst_core_id = dst_core_id
        self.kernel_id = kernel_id
        self.cmd_id = cmd_id
        self.args = args
        self.kwargs = kwargs
    
    def __getstate__(self):
        return {
            "msg_type": self.msg_type,
            "src_core_id": self.src_core_id,
            "dst_core_id": self.dst_core_id,
            "kernel_id": self.kernel_id,
            "cmd_id": self.cmd_id,
            "args": self.args,
            "kwargs": self.kwargs
        }
        
    def __setstate__(self, state):
        self.msg_type = state["msg_type"]
        self.src_core_id = state["src_core_id"]
        self.dst_core_id = state["dst_core_id"]
        self.kernel_id = state["kernel_id"]
        self.cmd_id = state["cmd_id"]
        self.args = state["args"]
        self.kwargs = state["kwargs"]

class Command:
    def __init__(
        self, 
        kernel_id: str,
        cmd_id: str,
        *args,
        **kwargs
    ):
        self._kernel_id         = kernel_id
        self.cmd_id             = cmd_id
        self.args               = args
        self.kwargs             = kwargs
        
        self._cached_cycle: int = None
        self._cached_cycle_slack: int = 0
        
    def get_remaining_cycles(self, core: 'Core') -> int:
        if self._cached_cycle is None:
            self._cached_cycle = self.run_cycle_model(core)
            
            if self._cached_cycle is None:
                raise Exception(f"[ERROR] Cycle model '{self.cycle_model}' returned None for command '{self.cmd_id}'")
            
            self._cached_cycle = max(1, self._cached_cycle)  # ensure at least 1 cycle

        return max(0, self._cached_cycle - self._cached_cycle_slack)

    def update_cycle_time(self, core: 'Core', cycle_time: int):
        if cycle_time < 0:
            raise ValueError(f"[ERROR] Cycle time cannot be negative: {cycle_time}")

        self._cached_cycle_slack += cycle_time
        
        if self.is_finished(core):
            flag = self.run_behavioral_model(core)
            
            if isinstance(flag, bool) and flag == False:
                self._cached_cycle_slack -= cycle_time

        if self.is_finished(core):
            core.run_command_debug_hook(cmd=self)

    def run_behavioral_model(self, core: 'Core'):
        with new_global_context(GlobalContextMode.EXECUTE, core, self._kernel_id):
            model = core.get_behavioral_model(self.cmd_id)
            return model(*self.args, **self.kwargs)
        
    def run_cycle_model(self, core: 'Core') -> int:
        with new_global_context(GlobalContextMode.EXECUTE, core, self._kernel_id):
            model = core.get_cycle_model(self.cmd_id)

            if model is None:
                return 1
            elif isinstance(model, int):
                return model
            elif callable(model):
                return model(*self.args, **self.kwargs)
            
            return None
    
    def force_finish(self):
        self._cached_cycle_slack = self._cached_cycle
        
    def to_rpc_message(self, src_core_id: str, dst_core_id: str) -> RPCMessage:
        return RPCMessage(
            src_core_id=src_core_id,
            dst_core_id=dst_core_id,
            kernel_id=self.kernel_id,
            cmd_id=self.cmd_id,
            args=self.args,
            kwargs=self.kwargs
        )
    
    @property
    def kernel_id(self) -> str:
        return self._kernel_id
        
    def is_finished(self, core: 'Core') -> bool:
        remaining_cycles = self.get_remaining_cycles(core)
        if remaining_cycles is None:
            return False
        return remaining_cycles <= 0

    def __str__(self):
        return f"Command[cmd_id={self.cmd_id}](args=({', '.join(map(str, self.args))}), kwargs={{{', '.join(f'{k}={v}' for k, v in self.kwargs.items())}}})"


class ThreadGroup(list['Kernel']):
    def __init__(self):
        super().__init__()
    
    def append(self, kernel: 'Kernel'):
        if not isinstance(kernel, Kernel):
            raise TypeError(f"[ERROR] Cannot add kernel '{kernel}' to the parallel kernel group since it is not an instance of Kernel")
        return super().append(kernel)

    def get_remaining_cycles(self, core: 'Core') -> int:
        return min(kernel.get_remaining_cycles(core) for kernel in self)

    def update_cycle_time(self, core: 'Core', cycle_time: int):
        for kernel in self:
            kernel.update_cycle_time(core, cycle_time)
    
    def is_finished(self, core: 'Core') -> bool:
        return all(kernel.is_finished(core) for kernel in self)

class Kernel:
    def __init__(self, kernel_id: str, func: Callable, core: 'Core', *args, **kwargs):
        self.kernel_id = kernel_id
        self.func = func
        self.core = core
        self.args = args
        self.kwargs = kwargs
        
        self.root_kernel = self
        
        self._is_compiled = False
        self._is_parallel = False
        
        self._execution_steps: list[Command | Kernel | ThreadGroup] = []
        self._execution_cursor: int = 0
        
    def set_parallel(self):
        self._is_compiled = True
        self._is_parallel = True
        return self
        
    def add_execution_step(self, step: 'Command | Kernel'):
        if get_global_context_mode() != GlobalContextMode.COMPILE:
            raise Exception(f"[ERROR] Cannot add execution step '{step}' to the kernel '{self.kernel_id}' since it is not in compile mode")
        if not isinstance(step, (Command, Kernel)):
            raise TypeError(f"[ERROR] Execution step must be an instance of Command or Kernel, but got {type(step).__name__}")
        
        self._execution_steps.append(step)
        
        if isinstance(step, Kernel):
            step.root_kernel = self.root_kernel
            
    def add_parallel_kernel_step(self) -> 'Kernel':
        if get_global_context_mode() != GlobalContextMode.COMPILE:
            raise Exception(f"[ERROR] Cannot add parallel kernel step to the kernel '{self.kernel_id}' since it is not in compile mode")
        
        if len(self._execution_steps) == 0:
            self._execution_steps.append(ThreadGroup())
        elif not isinstance(self._execution_steps[-1], ThreadGroup):
            self._execution_steps.append(ThreadGroup())

        parallel_kernel_idx = len(self._execution_steps[-1])

        parallel_kernel = Kernel(kernel_id=f"{self.kernel_id}.{parallel_kernel_idx}", func=None, core=None).set_parallel()
        parallel_kernel.root_kernel = self.root_kernel

        self._execution_steps[-1].append(parallel_kernel)

        return parallel_kernel

    def compile(self):
        with new_global_context(GlobalContextMode.COMPILE, self.core, self):
            self.func(self.core, *self.args, **self.kwargs)
        self._is_compiled = True
        
    def get_remaining_cycles(self, core: 'Core') -> int:
        if not self.is_compiled:
            self.compile()
        if self.is_finished(core):
            return None
        
        return self.current_step(core).get_remaining_cycles(core)

    def update_cycle_time(self, core: 'Core', cycle_time: int):
        if not self.is_compiled:
            self.compile()
            
        if self.is_finished(core):
            return
        
        self.current_step(core).update_cycle_time(core, cycle_time)
        
        if self.current_step(core).is_finished(core):
            self._execution_cursor += 1
            
    def current_step(self, core: 'Core') -> 'Command | Kernel | ThreadGroup':
        if not self.is_compiled:
            self.compile()
        if self.is_finished(core):
            return None
        
        return self._execution_steps[self._execution_cursor]
        
    @property
    def is_compiled(self) -> bool:
        return self._is_compiled
    
    def is_finished(self, core: 'Core') -> bool:
        if not self.is_compiled:
            self.compile()
        return self._execution_cursor >= len(self._execution_steps)


class CoreCycleModel:
    def __init__(self):
        pass

class Core:
    def __init__(self, core_id: str, cycle_model: CoreCycleModel=None):
        self.core_id = core_id

        self._cycle_model: CoreCycleModel = cycle_model

        self._dispatched_main_kernels: dict[str, Kernel] = {}
        self._dispatched_rpc_kernels:  dict[str, Kernel] = {}
        
        self._rpc_req_recv_queue = mp.Queue()                   # queue to receive RPC request messages
        self._rpc_rsp_recv_queue = mp.Queue()                   # queue to receive RPC response messages
        self._rpc_req_send_inbox: dict[str, mp.Queue] = None    # inbox to send RPC request messages (will be initialized by initialize() method)
        self._rpc_rsp_send_inbox: dict[str, mp.Queue] = None    # inbox to send RPC response messages (will be initialized by initialize() method)
        
        self._registered_command_debug_hooks: dict[str, Callable[[Command], None]] = {}
        
        self._use_cycle_model = True
        self._use_functional_model = True

        self._timestamp = 0

    def initialize(self, rpc_req_send_inbox: dict[str, mp.Queue] = None, rpc_rsp_send_inbox: dict[str, mp.Queue] = None):
        self._dispatched_main_kernels.clear()
        
        self._rpc_req_send_inbox = rpc_req_send_inbox
        self._rpc_rsp_send_inbox = rpc_rsp_send_inbox

        return self
        
    def change_sim_model_options(self, use_cycle_model: bool = None, use_functional_model: bool = None):
        self._use_cycle_model = use_cycle_model if use_cycle_model is not None else self._use_cycle_model
        self._use_functional_model = use_functional_model if use_functional_model is not None else self._use_functional_model

    def dispatch_kernel(self, kernel: Kernel):
        if not isinstance(kernel, Kernel):
            raise Exception(f"[ERROR] Cannot dispatch kernel '{kernel}' to the core since it is not an instance of CompiledKernel")
        
        kernel_name = kernel.kernel_id
        i = 0
        
        while kernel.kernel_id in self._dispatched_main_kernels.keys():
            kernel.kernel_id = f"{kernel_name}_{i}"
            i += 1
            
        self._dispatched_main_kernels[kernel.kernel_id] = kernel

    def update_cycle_time(self):
        if self.is_idle:
            return
        
        self._rpc_req_msg_process_routine()  # dispatch RPC kernel if the RPC request queue is not empty
        
        is_rpc_handle_mode = len(self._dispatched_rpc_kernels) > 0
        
        if is_rpc_handle_mode:
            dispatched_kernels = self._dispatched_rpc_kernels   # the core is now handling with RPC message
        else:
            dispatched_kernels = self._dispatched_main_kernels  # the core is now handling with the main kernels
        
        target_cycle_time = math.inf
        
        for kernel in dispatched_kernels.values():
            cycles = kernel.get_remaining_cycles(self)
            if cycles is not None:
                target_cycle_time = min(target_cycle_time, cycles)
        
        target_cycle_time = max(1, target_cycle_time)  # ensure at least 1 cycle
        
        kernel_ids = list(dispatched_kernels.keys())

        for kernel_id in kernel_ids:
            kernel = dispatched_kernels[kernel_id]
            kernel.update_cycle_time(self, target_cycle_time)

            if kernel.is_finished(self):
                if is_rpc_handle_mode:
                    self._rpc_rsp_msg_generation_routine(kernel)  # generate RPC response if the current ongoing RPC message is properly handled
                del dispatched_kernels[kernel_id]
                
        self._timestamp += target_cycle_time

    def register_command_debug_hook(self, hook: Callable[[Command], None]) -> str:
        def create_hook_id(i: int) -> str:
            return f"hook_{i}"
        
        MAX_HOOK_NUM = 1000
        
        for i in range(MAX_HOOK_NUM):
            hook_id = create_hook_id(i)
            if hook_id not in self._registered_command_debug_hooks:
                self._registered_command_debug_hooks[hook_id] = hook
                return hook_id
        
        raise Exception(f"[ERROR] Cannot register command debug hook since the maximum number of hooks ({MAX_HOOK_NUM}) is reached. Please remove some hooks before adding new ones.")
            
    def unregister_command_debug_hook(self, hook_id: str):
        if hook_id in self._registered_command_debug_hooks:
            del self._registered_command_debug_hooks[hook_id]
        else:
            raise Exception(f"[ERROR] Hook ID '{hook_id}' is not registered")
        
    def run_command_debug_hook(self, cmd: Command):
        for hook_id, hook in self._registered_command_debug_hooks.items():
            try:
                hook(self, cmd)
            except Exception as e:
                print(f"[ERROR] Command debug hook '{hook_id}' failed with error: {e}")
                
    def get_cycle_model(self, cmd_id: str) -> Callable:
        return getattr(self._cycle_model, cmd_id) if (self._use_cycle_model and hasattr(self._cycle_model, cmd_id)) else 1

    def get_behavioral_model(self, cmd_id: str) -> Callable:
        if not hasattr(self, cmd_id):
            raise Exception(f"[ERROR] Command '{cmd_id}' is not registered in the core '{self.core_id}'")
        return getattr(self, cmd_id)
    
    @core_command_method
    def _atomic_send_rpc_req_msg(self, msg: RPCMessage):
        self._rpc_req_send_inbox[msg.dst_core_id].put(msg)
        
    @core_command_method
    def _atomic_send_rpc_rsp_msg(self, msg: RPCMessage):
        rsp_msg = RPCMessage(
            msg_type=1,  # response message
            src_core_id=self.core_id,
            dst_core_id=msg.src_core_id,
            kernel_id=msg.kernel_id,
            cmd_id=msg.cmd_id,
            *msg.args,
            **msg.kwargs
        )
        self._rpc_rsp_send_inbox[msg.src_core_id].put(rsp_msg)
        
    @core_command_method
    def _atomic_wait_rpc_rsp_msg(self):
        if self.rpc_rsp_recv_queue.empty():
            return False

        self.rpc_rsp_recv_queue.get()  # TODO: the response message just eliminated and not handled?
        return True
        
    def _rpc_req_msg_process_routine(self):
        if self.rpc_req_recv_queue.empty():
            return

        msg: RPCMessage = self.rpc_req_recv_queue.get()
        rpc_kernel_id = msg.src_core_id  # RPC kernel ID is the same with the source core ID of the RPC message

        if not isinstance(msg, RPCMessage):
            raise Exception(f"[ERROR] Received message is not an instance of RPCMessage: {type(msg).__name__}")
        if msg.msg_type != 0:
            raise Exception(f"[ERROR] Received message is not a request message: {msg.msg_type}. This exception may caused by the faulty implementation of RPC.")
        
        func = getattr(self, msg.cmd_id, None)
        
        if func is None:
            raise Exception(f"[ERROR] Command '{msg.cmd_id}' is not registered in the core '{self.core_id}' for RPC processing")
        elif func.__name__ == "__core_command_method_wrapper":
            cmd = Command(core_id=self.core_id, cmd_id=msg.cmd_id, *msg.args, **msg.kwargs)
            kernel = Kernel(kernel_id=rpc_kernel_id, func=func, core=self, *msg.args, **msg.kwargs)
        elif func.__name__ == "__core_kernel_method_wrapper":
            kernel = Kernel(kernel_id=rpc_kernel_id, func=func, core=self, *msg.args, **msg.kwargs)
        else:
            raise Exception(f"[ERROR] Command '{msg.cmd_id}' is not a valid command for RPC processing. It must be a core command or a kernel method.")
        
        self._dispatched_rpc_kernels[rpc_kernel_id] = kernel
        
    def _rpc_rsp_msg_generation_routine(self, kernel: Kernel):
        if len(self._dispatched_rpc_kernels) > 0:
            return
        
        rsp_msg = RPCMessage(
            msg_type=1,  # response message
            src_core_id=self.core_id,
            dst_core_id=kernel.kernel_id,
            kernel_id=kernel.kernel_id,
            cmd_id="response",
        )

        self._rpc_rsp_send_inbox[kernel.core.core_id].put(rsp_msg)

    @property
    def is_idle(self) -> bool:
        for kernel in self._dispatched_main_kernels.values():
            if not kernel.is_finished(self):
                return False
        
        for kernel in self._dispatched_rpc_kernels.values():
            if not kernel.is_finished(self):
                return False
            
        return True
    
    @property
    def use_cycle_model(self) -> bool:
        return self._use_cycle_model
    
    @property
    def use_functional_model(self) -> bool:
        return self._use_functional_model
    
    @property
    def rpc_req_recv_queue(self) -> mp.Queue:
        return self._rpc_req_recv_queue

    @property
    def rpc_rsp_recv_queue(self) -> mp.Queue:
        return self._rpc_rsp_recv_queue

    @property
    def timestamp(self) -> int:
        return self._timestamp
