from omni.isaac.core.simulation_context import SimulationContext

from contextlib import contextmanager
import functools


def require_sim_initialized(func):

    @functools.wraps(func)
    def _func(*args, **kwargs):
        if SimulationContext.instance()._physics_sim_view is None:
            raise RuntimeError("SimulationContext not initialzed.")
        return func(*args, **kwargs)
    
    return _func


@contextmanager
def disable_warnings(physics_sim_view):
    try:
        physics_sim_view.enable_warnings(False)
        yield
    finally:
        physics_sim_view.enable_warnings(True)
        