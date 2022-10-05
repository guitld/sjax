import dataclasses
import collections
from typing import Dict, NamedTuple, Callable
import jax
import jax.numpy as jnp
import numpy as np

@dataclasses.dataclass
class Frame:
    """
    Tracks what's going on during a call of a transformed function
    """
    params: Dict[str, jnp.ndarray]
    is_initialising: bool = False

    # Keeps track of how many modules of each class have been created
    # by appending a number to it's name
    # Used to assign new modules unique names
    module_counts: Dict[str, int] = dataclasses.field(
        default_factory=lambda: collections.defaultdict(lambda: 0)
    )

    # Keep track of the path to the current module method call
    # Module methods will add themselves to this stack when called
    # Used to give each parameter a unique name corresponding to the method's scope
    call_stack: list = dataclasses.field(default_factory=list)
    
    def create_param_path(self, identifier) -> str:
        """
        Creates a unique path for this parameter
        """
        return '/'.join(['~'] + self.call_stack + [identifier])
    
    def create_unique_module_name(self, module_name: str) -> str:
        """
        Assigns a unique name to the module by appending its number to its name
        """
        number = self.module_counts[module_name]
        self.module_counts[module_name] += 1
        return f"{module_name}_{number}"
    
frame_stack = []

# Keep track of the PRNGs created to each module initialization.
# It uses a np.rand as an initial seed
rng_stack = [jax.random.PRNGKey(np.random.randint(1e8))]

def current_frame():
    return frame_stack[-1]

def set_seed(seed):
    rng_stack.clear()
    rng_stack.append(jax.random.PRNGKey(seed))

def current_rng():
    return rng_stack[-1]

def new_rng():
    rng_stack.append(jax.random.split(current_rng(), 1).flatten())

def parameter_shapes(params):
    return jax.tree_util.tree_map(lambda p: p.shape, params)

class Transformed(NamedTuple):
  init: Callable
  apply: Callable
  
def transform(f) -> Transformed:

  def init_f(*args, **kwargs):
    frame_stack.append(Frame({}, is_initialising=True))
    f(*args, **kwargs)
    frame = frame_stack.pop()
    return frame.params

  def apply_f(params, *args, **kwargs):
    frame_stack.append(Frame(params))
    outs = f(*args, **kwargs)
    frame_stack.pop()
    return outs

  return Transformed(init_f, apply_f)


class Module:
    def __init__(self):
        # Assign a unique name to this instance of the module
        self._unique_name = current_frame().create_unique_module_name(
            self.__class__.__name__)

def module_method(f):
    """
    Decorator for Module methods
    """
    def wrapped(self, *args, **kwargs):
        """
        A version of f that lets the frame know it's being called
        """
        # Self is the instance to which this method is attached
        module_name = self._unique_name
        call_stack = current_frame().call_stack
        call_stack.append(module_name)
        call_stack.append(f.__name__)
        outs = f(self, *args, **kwargs)
        assert call_stack.pop() == f.__name__
        assert call_stack.pop() == module_name
        return outs
    
    return wrapped

def get_param(identifier, shape):
    frame = current_frame()
    param_path = frame.create_param_path(identifier)

    if frame.is_initialising:
        frame.params[param_path] = jax.random.normal(current_rng(), shape=shape)
        new_rng()

    return frame.params[param_path]