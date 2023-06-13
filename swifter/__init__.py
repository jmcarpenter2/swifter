# flake8: noqa
import sys
import warnings
from logging import config
from .swifter import SeriesAccessor, DataFrameAccessor, set_defaults
from .parallel_accessor import (
    register_parallel_dataframe_accessor,
    register_parallel_series_accessor,
    register_modin,
)

warnings.filterwarnings("ignore", category=FutureWarning)

if "modin.pandas" in sys.modules:
    register_modin()

__all__ = [
    "set_defaults",
    "SeriesAccessor",
    "DataFrameAccessor",
    "register_parallel_dataframe_accessor",
    "register_parallel_series_accessor",
    "register_modin",
]
__version__ = "1.3.5"
