import logging
import importlib
import numpy as np
from os import devnull
from math import ceil
from psutil import cpu_count
from contextlib import contextmanager, redirect_stderr, redirect_stdout

ERRORS_TO_HANDLE = [AttributeError, ValueError, TypeError, KeyError]
try:
    from numba.core.errors import TypingError

    ERRORS_TO_HANDLE.append(TypingError)
except ImportError:
    pass
ERRORS_TO_HANDLE = tuple(ERRORS_TO_HANDLE)

RAY_INSTALLED = importlib.util.find_spec("ray") is not None


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
        self._SAMPLE_SIZE = SAMPLE_SIZE if self._nrows > (25 * SAMPLE_SIZE) else int(ceil(self._nrows / 25))
        self._SAMPLE_INDEX = sorted(np.random.choice(range(self._nrows), size=self._SAMPLE_SIZE, replace=False))
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

        return self
