import sys
import logging
from os import devnull
from math import ceil
from psutil import cpu_count, virtual_memory
from contextlib import contextmanager, redirect_stderr, redirect_stdout

ERRORS_TO_HANDLE = [AttributeError, ValueError, TypeError, KeyError]
try:
    from numba.core.errors import TypingError

    ERRORS_TO_HANDLE.append(TypingError)
except ImportError:
    pass
ERRORS_TO_HANDLE = tuple(ERRORS_TO_HANDLE)

SAMPLE_SIZE = 1000
N_REPEATS = 3


@contextmanager
def suppress_stdout_stderr_logging():
    """
    A context manager that redirects stdout and stderr to devnull
    Used for avoiding repeated prints of the data during sample/test applies of Swifter
    """
    previous_level = logging.root.manager.disable

    logging.disable(logging.CRITICAL)
    try:
        with open(devnull, "w") as fnull:
            with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
                yield (err, out)
    finally:
        logging.disable(previous_level)


class _SwifterBaseObject:
    def __init__(self, base_obj, npartitions=None):
        self._obj = base_obj
        self._nrows = self._obj.shape[0]
        self._ray_memory = None
        self.set_npartitions(npartitions=npartitions)

    @staticmethod
    def _validate_apply(expr, error_message):
        if not expr:
            raise ValueError(error_message)

    def set_npartitions(self, npartitions=None):
        """
        Set the number of partitions to use for dask/modin
        """
        if npartitions is None:
            self._npartitions = cpu_count() * 2
        else:
            self._npartitions = npartitions

        if "modin.pandas" in sys.modules:
            import modin.pandas as md

            md.DEFAULT_NPARTITIONS = self._npartitions
        return self

    def set_ray_compute(self, num_cpus=None, memory=None, **kwds):
        """
        Set the amount of compute used by ray for modin dataframes.

        Args:
            num_cpus: the number of cpus used by ray multiprocessing
            memory: the amount of memory allocated to ray workers
                If a proportion of 1 is provided (0 < memory <= 1],
                    then that proportion of available memory is used
                If a value greater than 1 is provided (1 < memory <= virtual_memory().available]
                    then that many bytes of memory are used
            kwds: key-word arguments to pass to `ray.init()`
        """
        import ray

        if memory is None:
            self._ray_memory = memory
        elif 0 < memory <= 1:
            self._ray_memory = ceil(virtual_memory().available * memory)
        elif 1 < memory <= virtual_memory().available:
            self._ray_memory = ceil(memory)
        else:
            raise MemoryError(
                f"Cannot allocate {memory} bytes of memory to ray. "
                f"Only {virtual_memory().available} bytes are currently available."
            )
        ray.shutdown()
        try:
            ray.init(num_cpus=num_cpus, _memory=self._ray_memory, **kwds)
        except TypeError:
            ray.init(num_cpus=num_cpus, memory=self._ray_memory, **kwds)
        return self
