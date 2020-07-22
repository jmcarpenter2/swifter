# flake8: noqa

from .swifter import SeriesAccessor, DataFrameAccessor
from .parallel_accessor import (
    register_parallel_dataframe_accessor, register_parallel_series_accessor
)

__all__ = ["register_parallel_dataframe_accessor", "register_parallel_series_accessor"]
__version__ = "0.305"
