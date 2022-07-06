import numpy as np
import warnings
from .base import _SwifterBaseObject, ERRORS_TO_HANDLE, suppress_stdout_stderr_logging


class _SwifterParallelBaseObject(_SwifterBaseObject):
    def set_dask_threshold(self, dask_threshold=1):
        """
        Set the threshold (seconds) for maximum allowed estimated duration
        of pandas apply before switching to dask
        """
        warnings.warn("Parallel Accessor does not use Dask.")
        return self

    def set_dask_scheduler(self, scheduler="processes"):
        """
        Set the dask scheduler
        :param scheduler: String, ["threads", "processes"]
        """
        warnings.warn("Parallel Accessor does not use Dask.")
        return self

    def progress_bar(self, enable=True, desc=None):
        """
        Turn on/off the progress bar, and optionally add a custom description
        """
        warnings.warn("Parallel Accessor does not use have a progress bar.")
        return self

    def allow_dask_on_strings(self, enable=True):
        """
        Override the string processing default, which is to not use dask if
        a string is contained in the pandas object
        """
        warnings.warn("Parallel Accessor does not use Dask.")
        return self

    def force_parallel(self, enable=True):
        """
        Force swifter to use dask parallel processing, without attempting any
        vectorized solution or estimating pandas apply duration to determine
        what will be the fastest approach
        """
        warnings.warn("Parallel Accessor does not use Dask.")
        return self

    def rolling(
        self,
        window,
        min_periods=None,
        center=False,
        win_type=None,
        on=None,
        axis=0,
        closed=None,
    ):
        """
        Create a swifter rolling object
        """
        raise NotImplementedError("Parallel Accessor cannot create Rolling objects.")

    def resample(
        self,
        rule,
        axis=0,
        closed=None,
        label=None,
        convention="start",
        kind=None,
        loffset=None,
        base=0,
        on=None,
        level=None,
    ):
        """
        Create a swifter resampler object
        """
        raise NotImplementedError("Parallel Accessor cannot create Resampler objects.")


class ParallelSeriesAccessor(_SwifterParallelBaseObject):
    def apply(self, func, convert_dtype=True, args=(), **kwds):
        """
        Apply the function to the Series using swifter
        """

        # if the series is empty, return early using Pandas
        if not self._nrows:
            return self._obj.apply(func, convert_dtype=convert_dtype, args=args, **kwds)

        sample = self._obj.iloc[self._SAMPLE_INDEX]
        if "axis" in kwds.keys():
            kwds.pop("axis")
            warnings.warn("Axis keyword not necessary because applying on a Series.")

        try:  # try to vectorize
            with suppress_stdout_stderr_logging():
                tmp_df = func(sample, *args, **kwds)
                sample_df = sample.apply(func, convert_dtype=convert_dtype, args=args, **kwds)
                self._validate_apply(
                    np.array_equal(sample_df, tmp_df) & (sample_df.shape == tmp_df.shape),
                    error_message=("Vectorized function sample doesn't " "match parallel series apply sample."),
                )
            return func(self._obj, *args, **kwds)
        except ERRORS_TO_HANDLE:  # if can't vectorize, return regular apply
            return self._obj.apply(func, convert_dtype=convert_dtype, args=args, **kwds)


class ParallelDataFrameAccessor(_SwifterParallelBaseObject):
    def apply(self, func, axis=0, raw=False, result_type=None, args=(), **kwds):
        """
        Apply the function to the Parallel DataFrame using swifter
        """
        # If there are no rows return early using default
        if not self._nrows:
            return self._obj.apply(func, axis=axis, raw=raw, result_type=result_type, args=args, **kwds)

        sample = self._obj.iloc[self._SAMPLE_INDEX]

        try:  # try to vectorize
            with suppress_stdout_stderr_logging():
                tmp_df = func(sample, *args, **kwds)
                sample_df = sample.apply(func, axis=axis, raw=raw, result_type=result_type, args=args, **kwds)
                self._validate_apply(
                    np.array_equal(sample_df, tmp_df) & (sample_df.shape == tmp_df.shape),
                    error_message=("Vectorized function sample doesn't " "match parallel dataframe apply sample."),
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
