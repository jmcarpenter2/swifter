# flake8: noqa
import sys
import warnings

from logging import config as logging_config
from .swifter import SeriesAccessor, DataFrameAccessor
from .parallel_accessor import (
    register_parallel_dataframe_accessor,
    register_parallel_series_accessor,
    register_modin,
)
from .config import (
    config,
    set_dask_threshold,
    set_dask_scheduler,
    progress_bar,
    allow_dask_on_strings,
)

warnings.filterwarnings("ignore", category=FutureWarning)

if "modin.pandas" in sys.modules:
    register_modin()

__all__ = [
    "SeriesAccessor",
    "DataFrameAccessor",
    "register_parallel_dataframe_accessor",
    "register_parallel_series_accessor",
    "register_modin",
    "config",
    "set_dask_threshold",
    "set_dask_scheduler",
    "progress_bar",
    "allow_dask_on_strings",
]
__version__ = "1.0.9"
