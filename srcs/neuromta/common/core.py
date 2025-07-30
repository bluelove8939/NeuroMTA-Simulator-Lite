from typing import Callable, Sequence, Any


__all__ = [
    "set_global_core_context",
    "set_global_compiled_kernel_context",
    "get_global_core_context",
    "get_global_compiled_kernel_context",
    "get_global_context",
    
    # "create_new_parallel_kernel",
    # "merge_parallel_kernels",
    
    "Command",
    "CommandConditional",
    
    "CompiledKernel",
    "ParallelCompiledKernel",
    
    "CoreCycleModel",
    "CoreFunctionalModel",
    "Core",
    
    "register_command_to_compiled_kernel",
    "core_kernel_method",
    "core_command_method",
    
    "DEFAULT_CORE_ID"
]


MAX_COMMAND_NUM_PER_KERNEL = 8192
DEFAULT_CORE_ID = "__DEFAULT"
NEGLECT = "__NEGLECT"


#################################################
# Global Context Management
#################################################

_core_context: 'Core' = None
_compiled_kernel_context: 'CompiledKernel' = None

def set_global_core_context(core: 'Core'):
    global _core_context
    _core_context = core

def get_global_core_context() -> 'Core':
    global _core_context
    if _core_context is None:
        raise Exception("[ERROR] Global core context is not set. Please set it using set_global_core_context() before using it.")
    return _core_context

def set_global_compiled_kernel_context(kernel: 'CompiledKernel'):
    global _compiled_kernel_context
    _compiled_kernel_context = kernel

def get_global_compiled_kernel_context() -> 'CompiledKernel':
    global _compiled_kernel_context
    if _compiled_kernel_context is None:
        raise Exception("[ERROR] Global compiled kernel context is not set. Please set it using set_global_compiled_kernel_context() before using it.")
    return _compiled_kernel_context

def get_global_context() -> str:
    cid = get_global_core_context().core_id
    kid = get_global_compiled_kernel_context().kernel_id
    return f"{cid}.{kid}"


#################################################
# Decorators for Command and Kernel Methods
#################################################

def register_command_to_compiled_kernel(command: 'Command'):
    global _compiled_kernel_context
    global MAX_COMMAND_NUM_PER_KERNEL
    
    if _compiled_kernel_context is None:
        raise Exception(f"[ERROR] Cannot register command '{type(command).__name__}' to the compiled kernel since it is called outside of a low-level kernel function")
    
    _compiled_kernel_context.add_command(command)

def core_kernel_method(_func: Callable):
    def __core_kernel_method_wrapper(_core, *_args, **_kwargs) -> CompiledKernel:
        global _compiled_kernel_context
        
        if not isinstance(_core, Core):
            raise Exception(f"[ERROR] Kernel method '{_func.__name__}' can only be called on an instance of Core or a function that receiving Core as the first argument")
        
        create_new_kernel_flag = _compiled_kernel_context is None
        
        if create_new_kernel_flag:
            _compiled_kernel_context = CompiledKernel(kernel_id=_func.__name__)
        elif not isinstance(_compiled_kernel_context, CompiledKernel):
            raise Exception(f"[ERROR] Low-level kernel function '{_func.__name__}' can only be called within another low-level kernel function or outside of any kernel context")
        
        _r = _func(_core, *_args, **_kwargs)
        
        if _r is not None:
            raise Exception(f"[ERROR] Low-level kernel function '{_func.__name__}' should not return a value")
        
        kernel = _compiled_kernel_context
        
        if create_new_kernel_flag:
            kernel.register_host_core(_core)
            _compiled_kernel_context = None
        
        return kernel
    return __core_kernel_method_wrapper

def core_command_method(_func: Callable):
    def __core_command_method_wrapper(_core, *_args, **_kwargs) -> Command:
        global _compiled_kernel_context
        
        if not isinstance(_core, Core):
            raise Exception(f"[ERROR] Command method '{_func.__name__}' can only be called on an instance of Core")
        
        if _compiled_kernel_context is None:
            return _func(_core, *_args, **_kwargs)
        
        if _core.use_cycle_model and hasattr(_core._cycle_model, _func.__name__):
            _cycle_model = getattr(_core._cycle_model, _func.__name__)
        else:
            _cycle_model = 1
        
        if _core.use_functional_model and hasattr(_core._functional_model, _func.__name__):
            _functional_model = getattr(_core._functional_model, _func.__name__)
        else:
            _functional_model = None
                
        cmd = Command(
            _core,                      # the core on which this command is registered
            _compiled_kernel_context,   # the kernel context in which this command is registered
            _func.__name__,             # the command ID is the name of the function
            _func,                      # the behavioral model is the function itself
            _cycle_model,               # the cycle model is the first argument
            _functional_model,          # the functional model is the second argument (if exists)
            *_args,                     # the arguments of the command
            **_kwargs                   # the keyword arguments of the command
        )
        
        register_command_to_compiled_kernel(cmd)
        
        return cmd
    return __core_command_method_wrapper

def core_conditional_command_method(_func: Callable):
    def __core_conditional_command_method_wrapper(_core, *_args, **_kwargs) -> CommandConditional:
        global _compiled_kernel_context
        
        if not isinstance(_core, Core):
            raise Exception(f"[ERROR] Conditional command method '{_func.__name__}' can only be called on an instance of Core")
        
        if _compiled_kernel_context is None:
            return _func(_core, *_args, **_kwargs)
        
        cmd = CommandConditional(
            _core, 
            _compiled_kernel_context, 
            _func.__name__, 
            _func, 
            *_args, 
            **_kwargs
        )
        
        register_command_to_compiled_kernel(cmd)
        
        return cmd
    return __core_conditional_command_method_wrapper




#################################################
# Implementation
#################################################

class Command:
    def __init__(
        self, 
        core: 'Core',
        kernel: 'CompiledKernel',
        cmd_id: str,
        behavioral_model: Callable,
        cycle_model: Callable,
        functional_model: Callable,
        *args,
        **kwargs
    ):
        self.core               = core
        self.kernel             = kernel
        self.cmd_id             = cmd_id
        self.behavioral_model   = behavioral_model
        self.cycle_model        = cycle_model
        self.functional_model   = functional_model
        self.args               = args
        self.kwargs             = kwargs
        
        self._cached_cycle: int = None
        self._cached_cycle_slack: int = 0
        
    def get_remaining_cycles(self) -> int:
        if self._cached_cycle is None:
            if isinstance(self.cycle_model, int):
                self._cached_cycle = self.cycle_model
            elif callable(self.cycle_model):
                self._cached_cycle = self.cycle_model(*self.args, **self.kwargs)
            
            if self._cached_cycle is None:
                raise Exception(f"[ERROR] Cycle model '{self.cycle_model}' returned None for command '{self.cmd_id}'")
            
            self._cached_cycle = max(1, self._cached_cycle)  # ensure at least 1 cycle

        return max(0, self._cached_cycle - self._cached_cycle_slack)

    def update_cycle_time(self, cycle_time: int):
        if cycle_time < 0:
            raise ValueError(f"[ERROR] Cycle time cannot be negative: {cycle_time}")

        self._cached_cycle_slack += cycle_time
        
        if self.get_remaining_cycles() <= 0:
            flag = self.run_behavioral_model()
            
            if isinstance(flag, bool) and flag == False:
                self._cached_cycle_slack -= cycle_time
            else:
                self.run_functional_model()
                
        if self.is_finished:
            self.core.run_command_debug_hook(cmd=self)

    def run_behavioral_model(self):
        set_global_core_context(self.core)
        set_global_compiled_kernel_context(self.kernel)
        return self.behavioral_model(self.core, *self.args, **self.kwargs)
    
    def run_functional_model(self):
        if self.functional_model is None:
            return
        
        set_global_core_context(self.core)
        set_global_compiled_kernel_context(self.kernel)
        return self.functional_model(*self.args, **self.kwargs)
    
    @property
    def core_id(self) -> str:
        return self.core.core_id
    
    @property
    def kernel_id(self) -> str:
        return self.kernel.kernel_id
        
    @property
    def is_finished(self) -> bool:
        return self.get_remaining_cycles() <= 0

    def __str__(self):
        return f"Command[cmd_id={self.cmd_id}](args=({', '.join(map(str, self.args))}), kwargs={{{', '.join(f'{k}={v}' for k, v in self.kwargs.items())}}})"
    
class CommandConditional(Command):
    def __init__(
        self, 
        core: 'Core',
        kernel: 'CompiledKernel',
        cmd_id: str,
        behavioral_model: Callable,
        *args,
        **kwargs
    ):
        super().__init__(
            core, 
            kernel, 
            cmd_id, 
            behavioral_model, 
            1,  # Use None for cycle model to indicate no cycle model is used
            None,  # Use None for functional model to indicate no functional model is used
            *args, 
            **kwargs
        )
        
        self._cached_cycle = 1
        self._cached_cycle_slack = 0
        
    def force_finish(self):
        self._cached_cycle_slack = self._cached_cycle
        
    def get_remaining_cycles(self) -> int:
        return max(0, self._cached_cycle - self._cached_cycle_slack)

    def update_cycle_time(self, cycle_time: int):
        if cycle_time < 0:
            raise ValueError(f"[ERROR] Cycle time cannot be negative: {cycle_time}")
                
        flag = self.run_behavioral_model()
        
        if not isinstance(flag, bool):
            raise Exception(f"[ERROR] Behavioral model for CommandWithCallback '{self.cmd_id}' should return a boolean value indicating whether the command is finished or not, but got {type(flag).__name__}")
        
        if flag:
            self.force_finish()
            self.core.run_command_debug_hook(cmd=self)


class CompiledKernel:
    def __init__(self, kernel_id: str):
        self.kernel_id = kernel_id
        self._host_core: 'Core' = None
        
        self._command_queue: list[Command] = []
        self._command_queue_cursor: int = 0
        
        self._p_kernel_group_queue: list[list[ParallelCompiledKernel]] = [[]]
        self._p_kernel_group_queue_cursor: int = 0
        
    def add_command(self, cmd: Command):
        if not isinstance(cmd, Command):
            raise Exception(f"[ERROR] Cannot add command '{cmd}' to the compiled kernel since it is not an instance of Command")
        
        if len(self._command_queue) >= MAX_COMMAND_NUM_PER_KERNEL:
            raise Exception(
                f"[ERROR] Cannot register command '{cmd.cmd_id}' to the compiled kernel since it exceeds the maximum number of commands per kernel ({MAX_COMMAND_NUM_PER_KERNEL}). "
                f"You may created a kernel with infinite while loop that cannot be compiled by the tracing. If you did not intend to create such a kernel, please check your kernel function implementation. "
                f"Otherwise, you can increase the maximum number of commands per kernel by changing the value of MAX_COMMAND_NUM_PER_KERNEL.")
        
        self._command_queue.append(cmd)
        
    def register_host_core(self, core: 'Core'):
        if not isinstance(core, Core):
            raise Exception(f"[ERROR] Cannot register host core '{core}' to the compiled kernel since it is not an instance of Core")
        
        self._host_core = core
        self._host_core.dispatch_kernel(self)
    
    def update_cycle_time(self, cycle_time: int):
        if self.is_finished:
            return
        
        item = self._command_queue[self._command_queue_cursor]
        item.update_cycle_time(cycle_time)
        
        if item.is_finished:
            self._command_queue_cursor += 1
            
    def create_parallel_kernel(self) -> 'ParallelCompiledKernel':
        new_kernel = ParallelCompiledKernel(orig_kernel=self, thread_id=len(self._p_kernel_group_queue[-1]))
        self._p_kernel_group_queue[-1].append(new_kernel)
        return new_kernel
    
    def register_cached_parallel_kernels(self):
        self._p_kernel_group_queue.append([])
        
    @property
    def is_finished(self) -> bool:
        if self._command_queue_cursor >= len(self._command_queue):
            return True
        
        cmd = self.current_command
        return cmd is None or cmd.is_finished
        
    @property
    def current_command(self) -> Command | None:
        if self._command_queue_cursor >= len(self._command_queue):
            return None
        return self._command_queue[self._command_queue_cursor]   
    
    @property
    def host_core(self) -> 'Core':
        return self._host_core
    
    @property
    def is_dispatched(self) -> bool:
        return self._host_core is not None

    def __str__(self):
        return f"CompiledKernel({self.kernel_id})"
    
class ParallelCompiledKernel(CompiledKernel):
    def __init__(self, orig_kernel: CompiledKernel, thread_id: int):
        super().__init__(kernel_id=f"{orig_kernel.kernel_id}.t{thread_id}")
        self._orig_kernel = orig_kernel
        self._thread_id = thread_id
        
    @property
    def orig_kernel(self) -> CompiledKernel:
        return self._orig_kernel

    @property
    def thread_id(self) -> int:
        return self._thread_id

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

        self._dispatched_kernels: dict[str, CompiledKernel] = {}
        self._registered_command_debug_hooks: dict[str, Callable[[Command], None]] = {}
        
        self.use_cycle_model = True
        self.use_functional_model = True

    def dispatch_kernel(self, kernel: CompiledKernel):
        if not isinstance(kernel, CompiledKernel):
            raise Exception(f"[ERROR] Cannot dispatch kernel '{kernel}' to the core since it is not an instance of CompiledKernel")
        
        kernel_name = kernel.kernel_id
        i = 0
        
        while kernel.kernel_id in self._dispatched_kernels.keys():
            kernel.kernel_id = f"{kernel_name}_{i}"
            i += 1
            
        self._dispatched_kernels[kernel.kernel_id] = kernel
        
    def get_current_commands(self) -> Sequence[Command]:
        ret = []
        
        for kernel in self._dispatched_kernels.values():
            cmd = kernel.current_command
            if cmd is not None:
                ret.append(cmd)

        return ret
    
    def get_remaining_cycles(self) -> int:
        cmds = self.get_current_commands()
        
        if len(cmds) == 0:
            return 1

        return max(1, min(map(lambda x: 0 if isinstance(x, CommandConditional) else x.get_remaining_cycles(), cmds)))

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
                
    @core_conditional_command_method
    def _merge_parallel_kernels(self, *parallel_kernels: ParallelCompiledKernel) -> bool:
        for p_kernel in parallel_kernels:
            if not p_kernel.is_dispatched:
                p_kernel.register_host_core(self)
        
        for p_kernel in parallel_kernels:
            if not isinstance(p_kernel, ParallelCompiledKernel):
                raise Exception(f"[ERROR] Cannot merge parallel kernel '{p_kernel}' since it is not an instance of ParallelCompiledKernel")
            
            if not p_kernel.is_finished:
                return False
        
        orig_kernel = p_kernel.orig_kernel
        orig_kernel._p_kernel_group_queue_cursor += 1
        return True
    
    @property
    def is_idle(self) -> bool:
        if len(self._dispatched_kernels) == 0:
            return True
        return len(self.get_current_commands()) == 0
    
    def create_new_parallel_kernel(self):
        global _compiled_kernel_context
        
        if isinstance(_compiled_kernel_context, ParallelCompiledKernel):
            new_kernel = _compiled_kernel_context.orig_kernel.create_parallel_kernel()
            _compiled_kernel_context = new_kernel
        elif isinstance(_compiled_kernel_context, CompiledKernel):
            new_kernel = _compiled_kernel_context.create_parallel_kernel()
            _compiled_kernel_context = new_kernel
        else:
            raise Exception(f"[ERROR] Cannot create a new parallel kernel since the global compiled kernel context is not an instance of CompiledKernel or ParallelCompiledKernel, but {type(_compiled_kernel_context).__name__}")

    def merge_parallel_kernels(self):
        global _compiled_kernel_context
        
        if not isinstance(_compiled_kernel_context, ParallelCompiledKernel):
            raise Exception(f"[ERROR] Cannot merge parallel kernels since the global compiled kernel context is not an instance of ParallelCompiledKernel, but {type(_compiled_kernel_context).__name__}")
        
        orig_kernel = _compiled_kernel_context.orig_kernel
        p_kernel_group = orig_kernel._p_kernel_group_queue[-1]
        orig_kernel.register_cached_parallel_kernels()
        
        _compiled_kernel_context = orig_kernel
        self._merge_parallel_kernels(*p_kernel_group)
