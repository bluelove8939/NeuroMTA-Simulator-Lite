import enum
from typing import Callable, Sequence, Any


__all__ = [
    # "set_global_core_context",
    # "set_global_kernel_context",
    "set_global_context",
    "get_global_core_context",
    "get_global_kernel_context",
    "get_global_pid",
    
    "Command",
    
    "Kernel",
    # "ParallelCompiledKernel",
    
    "CoreCycleModel",
    "CoreFunctionalModel",
    "Core",
    
    # "register_command_to_compiled_kernel",
    "core_kernel_method",
    "core_command_method",
    "start_parallel_kernel",
    "end_parallel_kernel",
    
    "DEFAULT_CORE_ID"
]


MAX_COMMAND_NUM_PER_KERNEL = 2 ** 20
DEFAULT_CORE_ID = "__DEFAULT"
NEGLECT = "__NEGLECT"


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

def set_global_context(context_mode: GlobalContextMode, core: 'Core', kernel: 'Kernel'):
    global _core_context, _kernel_context, _context_mode
    
    if isinstance(context_mode, str):
        context_mode = GlobalContextMode.__members__.get(context_mode.upper())
    
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
    kid = get_global_kernel_context().root_kernel.kernel_id   # the root kernel ID is used for global context PID
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
        if get_global_context_mode() == GlobalContextMode.IDLE:
            return _func(_core, *_args, **_kwargs)
        
        if not isinstance(_core, Core):
            raise Exception(f"[ERROR] Command method '{_func.__name__}' can only be called on an instance of Core")
        
        kernel_context = get_global_kernel_context()
        
        if kernel_context is None:
            raise Exception(f"[ERROR] Cannot register command '{_func.__name__}' to the compiled kernel since it is called outside of a low-level kernel function")
        
        _cycle_model        = getattr(_core._cycle_model, _func.__name__     )  if (_core.use_cycle_model      and hasattr(_core._cycle_model, _func.__name__     )) else 1
        _functional_model   = getattr(_core._functional_model, _func.__name__)  if (_core.use_functional_model and hasattr(_core._functional_model, _func.__name__)) else None

        cmd = Command(
            _core,                      # the core on which this command is registered
            kernel_context,             # the kernel context in which this command is registered
            _func.__name__,             # the command ID is the name of the function
            _func,                      # the behavioral model is the function itself
            _cycle_model,               # the cycle model is the first argument
            _functional_model,          # the functional model is the second argument (if exists)
            False,                      # is_conditional is False by default
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

def core_conditional_command_method(_func: Callable):
    def __core_conditional_command_method_wrapper(_core, *_args, **_kwargs) -> Command:
        if get_global_context_mode() == GlobalContextMode.IDLE:
            return _func(_core, *_args, **_kwargs)
        
        if not isinstance(_core, Core):
            raise Exception(f"[ERROR] Conditional command method '{_func.__name__}' can only be called on an instance of Core")
        
        kernel_context = get_global_kernel_context()
        
        if kernel_context is None:
            raise Exception(f"[ERROR] Cannot register command '{_func.__name__}' to the compiled kernel since it is called outside of a low-level kernel function")
        
        cmd = Command(
            _core,                      # the core on which this command is registered
            kernel_context,             # the kernel context in which this command is registered
            _func.__name__,             # the command ID is the name of the function
            _func,                      # the behavioral model is the function itself
            1,                          # the cycle model is 1 (indicating it is a conditional command)
            None,                       # the functional model is None (indicating it is a conditional command)
            True,                       # is_conditional is True
            *_args,                     # the arguments of the command
            **_kwargs                   # the keyword arguments of the command
        )
        
        if get_global_context_mode() == GlobalContextMode.COMPILE:
            kernel_context.add_execution_step(cmd)
        else:
            print(f"[WARNING] Command method '{_func.__name__}' is called outside of the compile or idle context. It implies that the command is called inside the command execution context, which is strictly prohibited. This is mainly because of the faulty implementation of the command method.")
            raise Exception(f"[ERROR] Command method '{_func.__name__}' is called outside of the compile or idle context.")
        
        return cmd
    return __core_conditional_command_method_wrapper

def start_parallel_kernel():
    if get_global_context_mode() != GlobalContextMode.COMPILE:
        raise Exception("[ERROR] Cannot create a parallel kernel since the global context mode is not COMPILE")
    
    kernel_context = get_global_kernel_context()
    parallel_kernel = kernel_context.add_parallel_kernel_step()
    
    store_global_parent_kernel_callstack()
    set_global_context(GlobalContextMode.COMPILE, get_global_core_context(), parallel_kernel)
    
def end_parallel_kernel():
    if get_global_context_mode() != GlobalContextMode.COMPILE:
        raise Exception("[ERROR] Cannot end parallel kernel since the global context mode is not COMPILE")

    restore_global_parent_kernel_callstack()


#################################################
# Implementation
#################################################

class Command:
    def __init__(
        self, 
        core: 'Core',
        kernel: 'Kernel',
        cmd_id: str,
        behavioral_model: Callable,
        cycle_model: Callable,
        functional_model: Callable,
        is_conditional: bool = False,
        *args,
        **kwargs
    ):
        self.core               = core
        self.kernel             = kernel
        self.cmd_id             = cmd_id
        self.behavioral_model   = behavioral_model
        self.cycle_model        = cycle_model
        self.functional_model   = functional_model
        self.is_conditional     = is_conditional
        self.args               = args
        self.kwargs             = kwargs
        
        self._cached_cycle: int = None
        self._cached_cycle_slack: int = 0
        
    def get_remaining_cycles(self) -> int:
        if self.is_conditional:
            return None     # Conditional commands do not have a fixed cycle count
        
        if self._cached_cycle is None:
            self._cached_cycle = self.run_cycle_model()
            
            if self._cached_cycle is None:
                raise Exception(f"[ERROR] Cycle model '{self.cycle_model}' returned None for command '{self.cmd_id}'")
            
            self._cached_cycle = max(1, self._cached_cycle)  # ensure at least 1 cycle

        return max(0, self._cached_cycle - self._cached_cycle_slack)

    def update_cycle_time(self, cycle_time: int):
        if cycle_time < 0:
            raise ValueError(f"[ERROR] Cycle time cannot be negative: {cycle_time}")

        if self.is_conditional:
            flag = self.run_behavioral_model()
        
            if not isinstance(flag, bool):
                raise Exception(f"[ERROR] Behavioral model for conditional command '{self.cmd_id}' should return a boolean value indicating whether the command is finished or not, but got {type(flag).__name__}")
            
            if flag:
                self.force_finish()
        else:
            self._cached_cycle_slack += cycle_time
            
            if self.is_finished:
                flag = self.run_behavioral_model()
                
                if isinstance(flag, bool) and flag == False:
                    self._cached_cycle_slack -= cycle_time
                else:
                    self.run_functional_model()
                    
        if self.is_finished:
            self.core.run_command_debug_hook(cmd=self)

    def run_behavioral_model(self):
        with new_global_context(GlobalContextMode.EXECUTE, self.core, self.kernel):
            return self.behavioral_model(self.core, *self.args, **self.kwargs)
    
    def run_functional_model(self):
        if self.functional_model is None:
            return
        
        with new_global_context(GlobalContextMode.EXECUTE, self.core, self.kernel):
            return self.functional_model(*self.args, **self.kwargs)
        
    def run_cycle_model(self) -> int:
        with new_global_context(GlobalContextMode.EXECUTE, self.core, self.kernel):
            if self.cycle_model is None:
                return 1
            elif isinstance(self.cycle_model, int):
                return self.cycle_model
            elif callable(self.cycle_model):
                return self.cycle_model(*self.args, **self.kwargs)
            
            return None
    
    def force_finish(self):
        self._cached_cycle_slack = self._cached_cycle
    
    @property
    def core_id(self) -> str:
        return self.core.core_id
    
    @property
    def kernel_id(self) -> str:
        return self.kernel.kernel_id
        
    @property
    def is_finished(self) -> bool:
        remaining_cycles = self.get_remaining_cycles()
        if remaining_cycles is None:
            return False
        return remaining_cycles <= 0

    def __str__(self):
        return f"Command[cmd_id={self.cmd_id}](args=({', '.join(map(str, self.args))}), kwargs={{{', '.join(f'{k}={v}' for k, v in self.kwargs.items())}}})"


class ParallelKernelGroup(list['Kernel']):
    def __init__(self):
        super().__init__()
    
    def append(self, kernel: 'Kernel'):
        if not isinstance(kernel, Kernel):
            raise TypeError(f"[ERROR] Cannot add kernel '{kernel}' to the parallel kernel group since it is not an instance of Kernel")
        return super().append(kernel)
        
    def get_remaining_cycles(self) -> int:
        return min(kernel.get_remaining_cycles() for kernel in self)

    def update_cycle_time(self, cycle_time: int):
        for kernel in self:
            kernel.update_cycle_time(cycle_time)
    
    @property
    def is_finished(self) -> bool:
        return all(kernel.is_finished for kernel in self)

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
        
        self._execution_steps: list[Command | Kernel | ParallelKernelGroup] = []
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
            self._execution_steps.append(ParallelKernelGroup())
        elif not isinstance(self._execution_steps[-1], ParallelKernelGroup):
            self._execution_steps.append(ParallelKernelGroup())

        parallel_kernel_idx = len(self._execution_steps[-1])

        parallel_kernel = Kernel(kernel_id=f"{self.kernel_id}.{parallel_kernel_idx}", func=None, core=None).set_parallel()
        parallel_kernel.root_kernel = self.root_kernel

        self._execution_steps[-1].append(parallel_kernel)

        return parallel_kernel

    def compile(self):
        with new_global_context(GlobalContextMode.COMPILE, self.core, self):
            self.func(self.core, *self.args, **self.kwargs)
        self._is_compiled = True
        
    def get_remaining_cycles(self) -> int:
        if not self.is_compiled:
            self.compile()
        if self.is_finished:
            return None
        
        return self.current_step.get_remaining_cycles()

    def update_cycle_time(self, cycle_time: int):
        if not self.is_compiled:
            self.compile()
            
        if self.is_finished:
            return
        
        self.current_step.update_cycle_time(cycle_time)
        
        if self.current_step.is_finished:
            self._execution_cursor += 1
            
    @property
    def current_step(self) -> 'Command | Kernel | ParallelKernelGroup':
        if not self.is_compiled:
            self.compile()
        if self.is_finished:
            return None
        
        return self._execution_steps[self._execution_cursor]
        
    @property
    def is_compiled(self) -> bool:
        return self._is_compiled
    
    @property
    def is_finished(self) -> bool:
        if not self.is_compiled:
            self.compile()
        return self._execution_cursor >= len(self._execution_steps)


class _CoreModel:
    def __init__(self):
        pass
    
class CoreCycleModel(_CoreModel):
    def __init__(self):
        super().__init__()

class CoreFunctionalModel(_CoreModel):
    def __init__(self):
        super().__init__()
    
    
class Core:
    def __init__(self, core_id: str, cycle_model: CoreCycleModel=None, functional_model: CoreFunctionalModel=None):
        self.core_id = core_id
        
        self._cycle_model:      CoreCycleModel      = cycle_model
        self._functional_model: CoreFunctionalModel = functional_model

        self._dispatched_kernels: dict[str, Kernel] = {}
        self._registered_command_debug_hooks: dict[str, Callable[[Command], None]] = {}
        
        self.use_cycle_model = True
        self.use_functional_model = True
        
    def change_sim_model_options(self, use_cycle_model: bool = None, use_functional_model: bool = None):
        self.use_cycle_model = use_cycle_model if use_cycle_model is not None else self.use_cycle_model
        self.use_functional_model = use_functional_model if use_functional_model is not None else self.use_functional_model

    def dispatch_kernel(self, kernel: Kernel):
        if not isinstance(kernel, Kernel):
            raise Exception(f"[ERROR] Cannot dispatch kernel '{kernel}' to the core since it is not an instance of CompiledKernel")
        
        kernel_name = kernel.kernel_id
        i = 0
        
        while kernel.kernel_id in self._dispatched_kernels.keys():
            kernel.kernel_id = f"{kernel_name}_{i}"
            i += 1
            
        self._dispatched_kernels[kernel.kernel_id] = kernel
    
    def get_remaining_cycles(self) -> int:
        if len(self._dispatched_kernels) == 0:
            return 1
        
        remaining_cycles = []
        
        for kernel in self._dispatched_kernels.values():
            cycles = kernel.get_remaining_cycles()
            if cycles is not None:
                remaining_cycles.append(cycles)
        
        if len(remaining_cycles) == 0:
            return 1
        
        return max(1, min(remaining_cycles))

    def update_cycle_time(self, cycle_time: int):
        kernel_ids = list(self._dispatched_kernels.keys())

        for kernel_id in kernel_ids:
            kernel = self._dispatched_kernels[kernel_id]
            kernel.update_cycle_time(cycle_time)
            
            if kernel.is_finished:
                del self._dispatched_kernels[kernel_id]
            
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
                hook(cmd)
            except Exception as e:
                print(f"[ERROR] Command debug hook '{hook_id}' failed with error: {e}")
    
    @property
    def is_idle(self) -> bool:
        if len(self._dispatched_kernels) == 0:
            return True
        
        for kernel in self._dispatched_kernels.values():
            if not kernel.is_finished:
                return False
        return True