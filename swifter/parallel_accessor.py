import numpy as np
import warnings
from .base import _SwifterBaseObject, ERRORS_TO_HANDLE, suppress_stdout_stderr


class ParallelSeriesAccessor(_SwifterBaseObject):
    def apply(self, func, convert_dtype=True, args=(), **kwds):
        """
        Apply the function to the Series using swifter
        """

        # if the series is empty, return early using Pandas
        if not self._nrows:
            return self._obj.apply(func, convert_dtype=convert_dtype, args=args, **kwds)

        sample = self._obj.iloc[:20]
        if "axis" in kwds.keys():
            kwds.pop("axis")
            warnings.warn("Axis keyword not necessary because applying on a Series.")

        try:  # try to vectorize
            with suppress_stdout_stderr():
                tmp_df = func(sample, *args, **kwds)
                sample_df = sample.apply(func, convert_dtype=convert_dtype, args=args, **kwds)
                self._validate_apply(
                    np.array_equal(sample_df, tmp_df) & (sample_df.shape == tmp_df.shape),
                    error_message="Vectorized function sample doesn't match parallel series apply sample.",
                )
            return func(self._obj, *args, **kwds)
        except ERRORS_TO_HANDLE:  # if can't vectorize, return regular apply
            return self._obj.apply(func, convert_dtype=convert_dtype, args=args, **kwds)


class ParallelDataFrameAccessor(_SwifterBaseObject):
    def apply(self, func, axis=0, raw=False, result_type=None, args=(), **kwds):
        """
        Apply the function to the Parallel DataFrame using swifter
        """
        # If there are no rows return early using default
        if not self._nrows:
            return self._obj.apply(func, axis=axis, raw=raw, result_type=result_type, args=args, **kwds)

        sample = self._obj.iloc[:20, :]

        try:  # try to vectorize
            with suppress_stdout_stderr():
                tmp_df = func(sample, *args, **kwds)
                sample_df = sample.apply(func, axis=axis, raw=raw, result_type=result_type, args=args, **kwds)
                self._validate_apply(
                    np.array_equal(sample_df, tmp_df) & (sample_df.shape == tmp_df.shape),
                    error_message="Vectorized function sample does not match parallel dataframe apply sample.",
                )
            return func(self._obj, *args, **kwds)
        except ERRORS_TO_HANDLE:  # if can't vectorize, return regular apply
            return self._obj.apply(func, axis=axis, raw=raw, result_type=result_type, args=args, **kwds)


def register_parallel_series_accessor(series_to_register):
    """
    Register a parallel series type with swifter attribute,
        giving access to automatic vectorization
    """
    current_init = series_to_register.__init__

    def new_init(self, *args, **kwds):
        current_init(self, *args, **kwds)
        self.swifter = ParallelSeriesAccessor(self)

    series_to_register.__init__ = new_init


def register_parallel_dataframe_accessor(dataframe_to_register):
    """
    Register a parallel dataframe type with swifter attribute,
        giving access to automatic vectorization
    """
    current_init = dataframe_to_register.__init__

    def new_init(self, *args, **kwds):
        current_init(self, *args, **kwds)
        self.swifter = ParallelDataFrameAccessor(self)

    dataframe_to_register.__init__ = new_init


def register_modin():
    """
    Register modin's series/dataframe as parallel accessors
    """
    from modin.pandas import Series, DataFrame

    register_parallel_series_accessor(Series)
    register_parallel_dataframe_accessor(DataFrame)
