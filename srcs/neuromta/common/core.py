import enum
from typing import Callable, Sequence, Any

from neuromta.common.custom_types import DataType


__all__ = [
    "Command",
    "CompiledCoreKernel",
    "Core",
    
    "register_command_to_compiled_kernel",
    "core_kernel_method",
    "core_command_method",
]


_compiled_kernel_context: 'CompiledCoreKernel' = None
MAX_COMMAND_NUM_PER_KERNEL = 8192
UNKNOWN = "UNKNOWN"


class Command:
    def __init__(
        self, 
        core: 'Core',
        cmd_id: str,
        cycle_model: Any,
        behavioral_model: Callable,
        *args,
        **kwargs
    ):
        self.core               = core
        self.cmd_id             = cmd_id
        self.cycle_model        = cycle_model
        self.behavioral_model   = behavioral_model
        self.args               = args
        self.kwargs             = kwargs
        
        self._cached_cycle: int = None
        self._cached_cycle_slack: int = 0
        
    def get_remaining_cycles(self) -> int:
        if self._cached_cycle is None:
            if isinstance(self.cycle_model, int):
                self._cached_cycle = self.cycle_model
            elif hasattr(self.core, self.cycle_model):
                cm = getattr(self.core, self.cycle_model)
                
                if callable(cm):
                    self._cached_cycle = cm(*self.args, **self.kwargs)
                else:
                    self._cached_cycle = cm
            else:
                raise Exception(f"[ERROR] Cycle model '{self.cycle_model}' is not defined in core '{self.core.__class__.__name__}'")
            
            self._cached_cycle = max(1, self._cached_cycle)  # ensure at least 1 cycle

        return max(0, self._cached_cycle - self._cached_cycle_slack)

    def update_cycle_time(self, cycle_time: int):
        if cycle_time < 0:
            raise ValueError(f"[ERROR] Cycle time cannot be negative: {cycle_time}")

        self._cached_cycle_slack += cycle_time
        
        if (self._cached_cycle_slack + cycle_time) >= self._cached_cycle:
            flag = self.run_behavioral_model()
            
            if isinstance(flag, bool) and flag == False:
                self._cached_cycle_slack -= cycle_time
                
        if self.is_finished:
            self.core.run_command_debug_hook(cmd=self)

    def run_behavioral_model(self):
        return self.behavioral_model(self.core, *self.args, **self.kwargs)
        
    @property
    def is_finished(self) -> bool:
        return self.get_remaining_cycles() <= 0

    def __str__(self):
        return f"Command[cmd_id={self.cmd_id}](args=({', '.join(map(str, self.args))}), kwargs={{{', '.join(f'{k}={v}' for k, v in self.kwargs.items())}}})"


class CompiledCoreKernel:
    def __init__(self, kernel_id: str):
        self.kernel_id = kernel_id
        self._command_subkernel_queue: list[Command | CompiledCoreKernel] = []
        self._cursor: int = 0
        
    def add_command(self, cmd: Command):
        if not isinstance(cmd, Command):
            raise Exception(f"[ERROR] Cannot add command '{cmd}' to the compiled kernel since it is not an instance of Command")
        
        if len(self._command_subkernel_queue) >= MAX_COMMAND_NUM_PER_KERNEL:
            raise Exception(
                f"[ERROR] Cannot register command '{cmd.cmd_id}' to the compiled kernel since it exceeds the maximum number of commands per kernel ({MAX_COMMAND_NUM_PER_KERNEL}). "
                f"You may created a kernel with infinite while loop that cannot be compiled by the tracing. If you did not intend to create such a kernel, please check your kernel function implementation. "
                f"Otherwise, you can increase the maximum number of commands per kernel by changing the value of MAX_COMMAND_NUM_PER_KERNEL.")
        
        self._command_subkernel_queue.append(cmd)
        
    def add_subkernel(self, subkernel: 'CompiledCoreKernel'):
        if not isinstance(subkernel, CompiledCoreKernel):
            raise Exception(f"[ERROR] Cannot add subkernel '{subkernel}' to the compiled kernel since it is not an instance of CompiledKernel")
        
        if len(self._command_subkernel_queue) >= MAX_COMMAND_NUM_PER_KERNEL:
            raise Exception(
                f"[ERROR] Cannot register subkernel to the compiled kernel since it exceeds the maximum number of commands per kernel ({MAX_COMMAND_NUM_PER_KERNEL}). "
                f"You may created a kernel with infinite while loop that cannot be compiled by the tracing. If you did not intend to create such a kernel, please check your kernel function implementation. "
                f"Otherwise, you can increase the maximum number of commands per kernel by changing the value of MAX_COMMAND_NUM_PER_KERNEL.")
        
        self._command_subkernel_queue.append(subkernel)
    
    def update_cycle_time(self, cycle_time: int):
        if self.is_finished:
            return
        
        item = self._command_subkernel_queue[self._cursor]
        item.update_cycle_time(cycle_time)
        
        if item.is_finished:
            self._cursor += 1
        
    @property
    def is_finished(self) -> bool:
        if self._cursor >= len(self._command_subkernel_queue):
            return True
        
        cmd = self.current_command
        return cmd is None or cmd.is_finished
        
    @property
    def current_command(self) -> Command | None:
        if self._cursor >= len(self._command_subkernel_queue):
            return None
        
        item = self._command_subkernel_queue[self._cursor]
        
        if isinstance(item, CompiledCoreKernel):
            return item.current_command
        elif isinstance(item, Command):
            return item
        
        return None       

    def __str__(self):
        command_str_expr = "\n".join(map(lambda x: '    ' + str(x), self._command_subkernel_queue))
        return "CompiledKernel(\n" + command_str_expr + "\n)"
    
    
class Core:
    def __init__(self, core_id: str=UNKNOWN):
        self.core_id = core_id
        self._dispatched_kernels: list[CompiledCoreKernel] = []
        self._registered_command_debug_hooks: dict[str, Callable[[Command], None]] = {}

    def dispatch_kernel(self, kernel: CompiledCoreKernel):
        if not isinstance(kernel, CompiledCoreKernel):
            raise Exception(f"[ERROR] Cannot dispatch kernel '{kernel}' to the core since it is not an instance of CompiledKernel")
        
        self._dispatched_kernels.append(kernel)
        
    def get_current_commands(self) -> Sequence[Command]:
        ret = []
        
        for kernel in self._dispatched_kernels:
            cmd = kernel.current_command
            if cmd is not None:
                ret.append(cmd)

        return ret
    
    def get_remaining_cycles(self) -> int:
        cmds = self.get_current_commands()
        
        if len(cmds) == 0:
            return 1
        
        return max(1, min(map(lambda x: x.get_remaining_cycles(), cmds)))
    
    def update_cycle_time(self, cycle_time: int):
        for kernel in self._dispatched_kernels:
            kernel.update_cycle_time(cycle_time)
            
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
        return len(self.get_current_commands()) == 0
    
    
def register_command_to_compiled_kernel(command: Command):
    global _compiled_kernel_context
    global MAX_COMMAND_NUM_PER_KERNEL
    
    if _compiled_kernel_context is None:
        raise Exception(f"[ERROR] Cannot register command '{type(command).__name__}' to the compiled kernel since it is called outside of a low-level kernel function")
    
    _compiled_kernel_context.add_command(command)

        
def core_kernel_method(_func: Callable):
    def __core_kernel_method_wrapper(*_args, **_kwargs) -> CompiledCoreKernel:
        global _compiled_kernel_context
        
        _compiled_kernel_context_history = _compiled_kernel_context
        
        if not (isinstance(_compiled_kernel_context_history, CompiledCoreKernel) or _compiled_kernel_context_history is None):
            raise Exception(f"[ERROR] Low-level kernel function '{_func.__name__}' can only be called within another low-level kernel function or outside of any kernel context")
        
        _compiled_kernel_context = CompiledCoreKernel(kernel_id=_func.__name__)
        _r = _func(*_args, **_kwargs)
        
        if _r is not None:
            raise Exception(f"[ERROR] Low-level kernel function '{_func.__name__}' should not return a value")
        
        if isinstance(_compiled_kernel_context_history, CompiledCoreKernel):
            _compiled_kernel_context_history.add_subkernel(_compiled_kernel_context)
        
        kernel = _compiled_kernel_context
        _compiled_kernel_context = _compiled_kernel_context_history
        
        return kernel
    return __core_kernel_method_wrapper


def core_command_method(_cycle_model: Any) -> Callable:
    def __core_command_method_decorator(_func: Callable):
        def __core_command_method_wrapper(_core, *_args, **_kwargs) -> Command:
            global _compiled_kernel_context
            
            if not isinstance(_core, Core):
                raise Exception(f"[ERROR] Command method '{_func.__name__}' can only be called on an instance of Core")
            
            if _compiled_kernel_context is None:
                return _func(_core, *_args, **_kwargs)
                    
            cmd = Command(
                _core,
                _func.__name__,
                _cycle_model,
                _func,
                *_args,
                **_kwargs
            )
            
            register_command_to_compiled_kernel(cmd)
            
            return cmd
        return __core_command_method_wrapper
    return __core_command_method_decorator
        
    
if __name__ == "__main__":  
    class CustomCore(Core):
        def __init__(self):
            super().__init__()
            
            self.reg = 0
        
        @core_command_method(1)
        def command1(self, arg1: int, arg2: str):
            self.reg += 1
            print(f"Executing command1 with arg1={arg1} and arg2='{arg2}'")
            
        @core_command_method(1)
        def command2(self):
            # self.reg += 1
            # print(f"Executing command2")
            print("Executing command2")
            
            if self.reg >= 5:
                return True
            return False
    
       
    @core_kernel_method
    def example_kernel_1(core: CustomCore):
        for i in range(10):
            core.command1(arg1=0, arg2="kernel1 triggering semaphore")
        
    @core_kernel_method
    def example_kernel_2(core: CustomCore):
        core.command2()
        core.command1(arg1=0, arg2="kernel2 terminated")
        
    
    core = CustomCore()
    kernel1 = example_kernel_1(core)
    kernel2 = example_kernel_2(core)
    
    print(kernel1)
    print(kernel2)
    
    # print(kernel)
    
    core.dispatch_kernel(kernel1)
    core.dispatch_kernel(kernel2)
    
    total_cycles = 0
    
    max_iter = 20
    
    while not core.is_idle:
        print(f"# {total_cycles}")
        cycles = core.get_remaining_cycles()
        core.update_cycle_time(cycles)
        
        total_cycles += cycles
        max_iter -= 1
        
        if max_iter <= 0:
            print("Max iterations reached, stopping execution.")
            break
        
    print(f"Total cycles executed: {total_cycles}")
