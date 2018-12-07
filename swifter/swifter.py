import timeit
import warnings
import pandas as pd

from psutil import cpu_count
from dask import dataframe as dd

from tqdm import tqdm
from .tqdm_dask_progressbar import TQDMDaskProgressBar

from numba.errors import TypingError

SAMP_SIZE = 1000


@pd.api.extensions.register_series_accessor("swifter")
class SeriesAccessor:
    def __init__(self, pandas_series, npartitions=None, dask_threshold=1, progress_bar=True):
        self._obj = pandas_series

        if npartitions is None:
            self._npartitions = cpu_count() * 2
        else:
            self._npartitions = npartitions
        self._dask_threshold = dask_threshold
        self._progress_bar = progress_bar

    def set_npartitions(self, npartitions=None):
        if npartitions is None:
            self._npartitions = cpu_count() * 2
        else:
            self._npartitions = npartitions
        return self

    def set_dask_threshold(self, dask_threshold=1):
        self._dask_threshold = dask_threshold
        return self

    def progress_bar(self, enable=True):
        self._progress_bar = enable
        return self

    def rolling(self, window, min_periods=None, center=False, win_type=None, on=None, axis=0, closed=None):
        kwds = {
            "window": window,
            "min_periods": min_periods,
            "center": center,
            "win_type": win_type,
            "on": on,
            "axis": axis,
            "closed": closed,
        }
        return Rolling(self._obj, self._npartitions, self._dask_threshold, self._progress_bar, **kwds)

    def _wrapped_apply(self, func, convert_dtype=True, args=(), **kwds):
        def wrapped():
            self._obj.iloc[:SAMP_SIZE].apply(func, convert_dtype=convert_dtype, args=args, **kwds)

        return wrapped

    def _dask_apply(self, func, convert_dtype, *args, **kwds):
        samp = self._obj.iloc[: self._npartitions * 2]
        meta = samp.apply(func, convert_dtype=convert_dtype, args=args, **kwds)
        try:
            tmp_df = (
                dd.from_pandas(samp, npartitions=self._npartitions)
                .map_partitions(func, *args, meta=meta, **kwds)
                .compute(scheduler="processes")
            )
            assert tmp_df.shape == meta.shape
            if self._progress_bar:
                with TQDMDaskProgressBar(desc="Dask Apply"):
                    return (
                        dd.from_pandas(self._obj, npartitions=self._npartitions)
                        .map_partitions(func, *args, meta=meta, **kwds)
                        .compute(scheduler="processes")
                    )
            else:
                return (
                    dd.from_pandas(self._obj, npartitions=self._npartitions)
                    .map_partitions(func, *args, meta=meta, **kwds)
                    .compute(scheduler="processes")
                )
        except (AssertionError, AttributeError, ValueError, TypeError) as e:
            if self._progress_bar:
                with TQDMDaskProgressBar(desc="Dask Apply"):
                    return (
                        dd.from_pandas(self._obj, npartitions=self._npartitions)
                        .apply(lambda x: func(x, *args, **kwds), meta=meta)
                        .compute(scheduler="processes")
                    )
            else:
                return (
                    dd.from_pandas(self._obj, npartitions=self._npartitions)
                    .apply(lambda x: func(x, *args, **kwds), meta=meta)
                    .compute(scheduler="processes")
                )

    def apply(self, func, convert_dtype=True, args=(), **kwds):
        samp = self._obj.iloc[: self._npartitions * 2]
        str_object = samp.dtype == "object"  # check if input is string

        if "axis" in kwds.keys():
            kwds.pop("axis")
            warnings.warn("Axis keyword not necessary because applying on a Series.")

        try:  # try to vectorize
            tmp_df = func(samp, *args, **kwds)
            assert tmp_df.shape == samp.apply(func, convert_dtype=convert_dtype, args=args, **kwds).shape
            return func(self._obj, *args, **kwds)
        except (
            AssertionError,
            AttributeError,
            ValueError,
            TypeError,
            TypingError,
        ) as e:  # if can't vectorize, estimate time to pandas apply
            wrapped = self._wrapped_apply(func, convert_dtype=convert_dtype, args=args, **kwds)
            n_repeats = 3
            timed = timeit.timeit(wrapped, number=n_repeats)
            samp_proc_est = timed / n_repeats
            est_apply_duration = samp_proc_est / SAMP_SIZE * self._obj.shape[0]

            # if pandas apply takes too long and input is not str, use dask
            if (est_apply_duration > self._dask_threshold) and (not str_object):
                return self._dask_apply(func, convert_dtype, *args, **kwds)
            else:  # use pandas
                if self._progress_bar:
                    tqdm.pandas(desc="Pandas Apply")
                    return self._obj.progress_apply(func, convert_dtype=convert_dtype, args=args, **kwds)
                else:
                    return self._obj.apply(func, convert_dtype=convert_dtype, args=args, **kwds)


@pd.api.extensions.register_dataframe_accessor("swifter")
class DataFrameAccessor:
    def __init__(self, pandas_dataframe, npartitions=None, dask_threshold=1, progress_bar=True):
        self._obj = pandas_dataframe

        if npartitions is None:
            self._npartitions = cpu_count() * 2
        else:
            self._npartitions = npartitions
        self._dask_threshold = dask_threshold
        self._progress_bar = progress_bar

    def set_npartitions(self, npartitions=None):
        if npartitions is None:
            self._npartitions = cpu_count() * 2
        else:
            self._npartitions = npartitions
        return self

    def set_dask_threshold(self, dask_threshold=1):
        self._dask_threshold = dask_threshold
        return self

    def progress_bar(self, enable=True):
        self._progress_bar = enable
        return self

    def rolling(self, window, min_periods=None, center=False, win_type=None, on=None, axis=0, closed=None):
        kwds = {
            "window": window,
            "min_periods": min_periods,
            "center": center,
            "win_type": win_type,
            "on": on,
            "axis": axis,
            "closed": closed,
        }
        return Rolling(self._obj, self._npartitions, self._dask_threshold, self._progress_bar, **kwds)

    def _wrapped_apply(self, func, axis=0, broadcast=None, raw=False, reduce=None, result_type=None, args=(), **kwds):
        def wrapped():
            self._obj.iloc[:SAMP_SIZE, :].apply(
                func, axis=axis, broadcast=broadcast, raw=raw, reduce=reduce, result_type=result_type, args=args, **kwds
            )

        return wrapped

    def _dask_apply(self, func, axis=0, broadcast=None, raw=False, reduce=None, result_type=None, *args, **kwds):
        samp = self._obj.iloc[: self._npartitions * 2, :]
        meta = samp.apply(
            func, axis=axis, broadcast=broadcast, raw=raw, reduce=reduce, result_type=result_type, args=args, **kwds
        )
        try:
            tmp_df = (
                dd.from_pandas(samp, npartitions=self._npartitions)
                .apply(func, *args, axis=axis, meta=meta, **kwds)
                .compute(scheduler="processes")
            )
            assert tmp_df.shape == meta.shape
            if self._progress_bar:
                with TQDMDaskProgressBar(desc="Dask Apply"):
                    return (
                        dd.from_pandas(self._obj, npartitions=self._npartitions)
                        .apply(func, *args, axis=axis, meta=meta, **kwds)
                        .compute(scheduler="processes")
                    )
            else:
                return (
                    dd.from_pandas(self._obj, npartitions=self._npartitions)
                    .apply(func, *args, axis=axis, meta=meta, **kwds)
                    .compute(scheduler="processes")
                )
        except (AssertionError, AttributeError, ValueError, TypeError) as e:
            if self._progress_bar:
                tqdm.pandas(desc="Pandas Apply")
                return self._obj.progress_apply(
                    func,
                    axis=axis,
                    broadcast=broadcast,
                    raw=raw,
                    reduce=reduce,
                    result_type=result_type,
                    args=args,
                    **kwds
                )
            else:
                return self._obj.apply(
                    func,
                    axis=axis,
                    broadcast=broadcast,
                    raw=raw,
                    reduce=reduce,
                    result_type=result_type,
                    args=args,
                    **kwds
                )

    def apply(self, func, axis=1, broadcast=None, raw=False, reduce=None, result_type=None, args=(), **kwds):
        samp = self._obj.iloc[: self._npartitions * 2, :]
        str_object = "object" in samp.dtypes.values  # check if input is string

        try:  # try to vectorize
            if "axis" in kwds.keys():
                kwds.pop("axis")
            tmp_df = func(samp, *args, **kwds)
            assert (
                tmp_df.shape
                == samp.apply(
                    func,
                    axis=axis,
                    broadcast=broadcast,
                    raw=raw,
                    reduce=reduce,
                    result_type=result_type,
                    args=args,
                    **kwds
                ).shape
            )
            return func(self._obj, *args, **kwds)
        except (
            AssertionError,
            AttributeError,
            ValueError,
            TypeError,
            TypingError,
        ) as e:  # if can't vectorize, estimate time to pandas apply
            wrapped = self._wrapped_apply(
                func, axis=axis, broadcast=broadcast, raw=raw, reduce=reduce, result_type=result_type, args=args, **kwds
            )
            n_repeats = 3
            timed = timeit.timeit(wrapped, number=n_repeats)
            samp_proc_est = timed / n_repeats
            est_apply_duration = samp_proc_est / SAMP_SIZE * self._obj.shape[0]

            # if pandas apply takes too long and input is not str, use dask
            if (est_apply_duration > self._dask_threshold) and (not str_object):
                return self._dask_apply(func, axis, broadcast, raw, reduce, result_type, *args, **kwds)
            else:  # use pandas
                if self._progress_bar:
                    tqdm.pandas(desc="Pandas Apply")
                    return self._obj.progress_apply(
                        func,
                        axis=axis,
                        broadcast=broadcast,
                        raw=raw,
                        reduce=reduce,
                        result_type=result_type,
                        args=args,
                        **kwds
                    )
                else:
                    return self._obj.apply(
                        func,
                        axis=axis,
                        broadcast=broadcast,
                        raw=raw,
                        reduce=reduce,
                        result_type=result_type,
                        args=args,
                        **kwds
                    )


class Transformation:
    def __init__(self, obj, npartitions=None, dask_threshold=1, progress_bar=True):
        self._obj = obj
        self._samp_pd = obj.iloc[:SAMP_SIZE]
        self._obj_pd = obj
        self._obj_dd = dd.from_pandas(obj, npartitions=npartitions)
        self._nrows = obj.shape[0]
        self._npartitions = npartitions
        self._dask_threshold = dask_threshold
        self._progress_bar = progress_bar

    def set_npartitions(self, npartitions=None):
        if npartitions is None:
            self._npartitions = cpu_count() * 2
        else:
            self._npartitions = npartitions
        return self

    def set_dask_threshold(self, dask_threshold=1):
        self._dask_threshold = dask_threshold
        return self

    def progress_bar(self, enable=True):
        self._progress_bar = enable
        return self

    def _wrapped_apply(self, func, *args, **kwds):
        def wrapped():
            self._samp_pd.apply(func, *args, **kwds)

        return wrapped

    def _dask_apply(self, func, *args, **kwds):
        if self._progress_bar:
            with TQDMDaskProgressBar(desc="Dask Apply"):
                return self._obj_dd.apply(func, *args, **kwds).compute(scheduler="processes")
        else:
            return self._obj_dd.apply(func, *args, **kwds).compute(scheduler="processes")

    def apply(self, func, *args, **kwds):
        # estimate time to pandas apply
        wrapped = self._wrapped_apply(func, *args, **kwds)
        n_repeats = 3
        timed = timeit.timeit(wrapped, number=n_repeats)
        samp_proc_est = timed / n_repeats
        est_apply_duration = samp_proc_est / SAMP_SIZE * self._nrows

        # if pandas apply takes too long, use dask
        if est_apply_duration > self._dask_threshold:
            return self._dask_apply(func, *args, **kwds)
        else:  # use pandas
            if self._progress_bar:
                tqdm.pandas(desc="Pandas Apply")
                return self._obj_pd.apply(func, *args, **kwds)
            else:
                return self._obj_pd.apply(func, *args, **kwds)


class Rolling(Transformation):
    def __init__(self, obj, npartitions=None, dask_threshold=1, progress_bar=True, **kwds):
        super(self).__init__(obj, npartitions, dask_threshold, progress_bar)
        self._samp_pd = self._samp_pd.rolling(**kwds)
        self._obj_pd = self._obj_pd.rolling(**kwds)
        kwds.pop("on")
        kwds.pop("closed")
        self._obj_dd = self._obj_dd.rolling(**kwds)
