# flake8: noqa
import sys
from .swifter import SeriesAccessor, DataFrameAccessor
from .parallel_accessor import register_parallel_dataframe_accessor, register_parallel_series_accessor, register_modin

if "modin.pandas" in sys.modules:
    register_modin()

__all__ = [
    "SeriesAccessor",
    "DataFrameAccessor",
    "register_parallel_dataframe_accessor",
    "register_parallel_series_accessor",
    "register_modin",
]
__version__ = "1.0.1"
