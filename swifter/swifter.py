import timeit
import warnings
import pandas as pd

from psutil import cpu_count
from dask import dataframe as dd

from tqdm.autonotebook import tqdm
from .tqdm_dask_progressbar import TQDMDaskProgressBar

from numba.errors import TypingError

SAMP_SIZE = 1000


class _SwifterObject:
    def __init__(
        self,
        pandas_obj,
        npartitions=None,
        dask_threshold=1,
        scheduler="processes",
        progress_bar=True,
        allow_dask_on_strings=False,
    ):
        self._obj = pandas_obj
        self._nrows = self._obj.shape[0]
        self._SAMP_SIZE = SAMP_SIZE if self._nrows > 25000 else int(round(self._nrows / 25))

        if npartitions is None:
            self._npartitions = cpu_count() * 2
        else:
            self._npartitions = npartitions
        self._dask_threshold = dask_threshold
        self._scheduler = scheduler
        self._progress_bar = progress_bar
        self._allow_dask_on_strings = allow_dask_on_strings

    def set_npartitions(self, npartitions=None):
        """
        Set the number of partitions to use for dask
        """
        if npartitions is None:
            self._npartitions = cpu_count() * 2
        else:
            self._npartitions = npartitions
        return self

    def set_dask_threshold(self, dask_threshold=1):
        """
        Set the threshold (seconds) for maximum allowed estimated duration of pandas apply before switching to dask
        """
        self._dask_threshold = dask_threshold
        return self

    def set_dask_scheduler(self, scheduler="processes"):
        """
        Set the dask scheduler
        :param scheduler: String, ["threads", "processes"]
        """
        self._scheduler = scheduler
        return self

    def progress_bar(self, enable=True):
        """
        Turn on/off the progress bar
        """
        self._progress_bar = enable
        return self

    def allow_dask_on_strings(self, enable=True):
        """
        Override the string processing default, which is to not use dask if a string is contained in the pandas object
        """
        self._allow_dask_on_strings = enable
        return self

    def rolling(self, window, min_periods=None, center=False, win_type=None, on=None, axis=0, closed=None):
        """
        Create a swifter rolling object
        """
        kwds = {
            "window": window,
            "min_periods": min_periods,
            "center": center,
            "win_type": win_type,
            "on": on,
            "axis": axis,
            "closed": closed,
        }
        return Rolling(self._obj, self._npartitions, self._dask_threshold, self._scheduler, self._progress_bar, **kwds)


@pd.api.extensions.register_series_accessor("swifter")
class SeriesAccessor(_SwifterObject):
    def _wrapped_apply(self, func, convert_dtype=True, args=(), **kwds):
        def wrapped():
            self._obj.iloc[: self._SAMP_SIZE].apply(func, convert_dtype=convert_dtype, args=args, **kwds)

        return wrapped

    def _dask_apply(self, func, convert_dtype, *args, **kwds):
        samp = self._obj.iloc[: self._npartitions * 2]
        meta = samp.apply(func, convert_dtype=convert_dtype, args=args, **kwds)
        try:
            tmp_df = (
                dd.from_pandas(samp, npartitions=self._npartitions)
                .map_partitions(func, *args, meta=meta, **kwds)
                .compute(scheduler=self._scheduler)
            )
            assert tmp_df.equals(meta)
            if self._progress_bar:
                with TQDMDaskProgressBar(desc="Dask Apply"):
                    return (
                        dd.from_pandas(self._obj, npartitions=self._npartitions)
                        .map_partitions(func, *args, meta=meta, **kwds)
                        .compute(scheduler=self._scheduler)
                    )
            else:
                return (
                    dd.from_pandas(self._obj, npartitions=self._npartitions)
                    .map_partitions(func, *args, meta=meta, **kwds)
                    .compute(scheduler=self._scheduler)
                )
        except (AssertionError, AttributeError, ValueError, TypeError):
            if self._progress_bar:
                with TQDMDaskProgressBar(desc="Dask Apply"):
                    return (
                        dd.from_pandas(self._obj, npartitions=self._npartitions)
                        .apply(lambda x: func(x, *args, **kwds), convert_dtype=convert_dtype, meta=meta)
                        .compute(scheduler=self._scheduler)
                    )
            else:
                return (
                    dd.from_pandas(self._obj, npartitions=self._npartitions)
                    .apply(lambda x: func(x, *args, **kwds), convert_dtype=convert_dtype, meta=meta)
                    .compute(scheduler=self._scheduler)
                )

    def apply(self, func, convert_dtype=True, args=(), **kwds):
        """
        Apply the function to the Series using swifter
        """
        samp = self._obj.iloc[: self._npartitions * 2]
        # check if input is string or if the user is overriding the string processing default
        str_processing = (samp.dtype == "object") if not self._allow_dask_on_strings else False

        if "axis" in kwds.keys():
            kwds.pop("axis")
            warnings.warn("Axis keyword not necessary because applying on a Series.")

        try:  # try to vectorize
            tmp_df = func(samp, *args, **kwds)
            assert samp.apply(func, convert_dtype=convert_dtype, args=args, **kwds).equals(tmp_df)
            return func(self._obj, *args, **kwds)
        except (
            AssertionError,
            AttributeError,
            ValueError,
            TypeError,
            TypingError,
        ):  # if can't vectorize, estimate time to pandas apply
            wrapped = self._wrapped_apply(func, convert_dtype=convert_dtype, args=args, **kwds)
            n_repeats = 3
            timed = timeit.timeit(wrapped, number=n_repeats)
            samp_proc_est = timed / n_repeats
            est_apply_duration = samp_proc_est / self._SAMP_SIZE * self._obj.shape[0]

            # if pandas apply takes too long and not performing str processing, use dask
            if (est_apply_duration > self._dask_threshold) and (not str_processing):
                return self._dask_apply(func, convert_dtype, *args, **kwds)
            else:  # use pandas
                if self._progress_bar:
                    tqdm.pandas(desc="Pandas Apply")
                    return self._obj.progress_apply(func, convert_dtype=convert_dtype, args=args, **kwds)
                else:
                    return self._obj.apply(func, convert_dtype=convert_dtype, args=args, **kwds)


@pd.api.extensions.register_dataframe_accessor("swifter")
class DataFrameAccessor(_SwifterObject):
    def _wrapped_apply(self, func, axis=0, broadcast=None, raw=False, reduce=None, result_type=None, args=(), **kwds):
        def wrapped():
            self._obj.iloc[: self._SAMP_SIZE, :].apply(
                func, axis=axis, broadcast=broadcast, raw=raw, reduce=reduce, result_type=result_type, args=args, **kwds
            )

        return wrapped

    def _dask_apply(self, func, axis=0, broadcast=None, raw=False, reduce=None, result_type=None, *args, **kwds):
        samp = self._obj.iloc[: self._npartitions * 2, :]
        meta = samp.apply(
            func, axis=axis, broadcast=broadcast, raw=raw, reduce=reduce, result_type=result_type, args=args, **kwds
        )
        try:
            if broadcast:
                result_type = "broadcast"
            elif reduce:
                result_type = "reduce"

            tmp_df = (
                dd.from_pandas(samp, npartitions=self._npartitions)
                .apply(func, *args, axis=axis, raw=raw, result_type=result_type, meta=meta, **kwds)
                .compute(scheduler=self._scheduler)
            )
            assert tmp_df.equals(meta)
            if self._progress_bar:
                with TQDMDaskProgressBar(desc="Dask Apply"):
                    return (
                        dd.from_pandas(self._obj, npartitions=self._npartitions)
                        .apply(func, *args, axis=axis, raw=raw, result_type=result_type, meta=meta, **kwds)
                        .compute(scheduler=self._scheduler)
                    )
            else:
                return (
                    dd.from_pandas(self._obj, npartitions=self._npartitions)
                    .apply(func, *args, axis=axis, raw=raw, result_type=result_type, meta=meta, **kwds)
                    .compute(scheduler=self._scheduler)
                )
        except (AssertionError, AttributeError, ValueError, TypeError):
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

    def apply(self, func, axis=0, broadcast=None, raw=False, reduce=None, result_type=None, args=(), **kwds):
        """
        Apply the function to the DataFrame using swifter
        """
        samp = self._obj.iloc[: self._npartitions * 2, :]
        # check if input is string or if the user is overriding the string processing default
        str_processing = ("object" in samp.dtypes.values) if not self._allow_dask_on_strings else False

        try:  # try to vectorize
            tmp_df = func(samp, *args, **kwds)
            assert samp.apply(
                func, axis=axis, broadcast=broadcast, raw=raw, reduce=reduce, result_type=result_type, args=args, **kwds
            ).equals(tmp_df)
            return func(self._obj, *args, **kwds)
        except (
            AssertionError,
            AttributeError,
            ValueError,
            TypeError,
            TypingError,
        ):  # if can't vectorize, estimate time to pandas apply
            wrapped = self._wrapped_apply(
                func, axis=axis, broadcast=broadcast, raw=raw, reduce=reduce, result_type=result_type, args=args, **kwds
            )
            n_repeats = 3
            timed = timeit.timeit(wrapped, number=n_repeats)
            samp_proc_est = timed / n_repeats
            est_apply_duration = samp_proc_est / self._SAMP_SIZE * self._obj.shape[0]

            # if pandas apply takes too long and not performing str processing, use dask
            if (est_apply_duration > self._dask_threshold) and (not str_processing):
                if axis == 0:
                    raise NotImplementedError(
                        "Swifter cannot perform axis=0 applies on large datasets.\n"
                        "Dask currently does not have an axis=0 apply implemented.\n"
                        "More details at https://github.com/jmcarpenter2/swifter/issues/10"
                    )
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


class Transformation(_SwifterObject):
    def __init__(
        self,
        obj,
        npartitions=None,
        dask_threshold=1,
        scheduler="processes",
        progress_bar=True,
        allow_dask_on_strings=False,
    ):
        super().__init__(obj, npartitions, dask_threshold, scheduler, progress_bar, allow_dask_on_strings)
        self._samp_pd = obj.iloc[: self._SAMP_SIZE]
        self._obj_pd = obj
        self._obj_dd = dd.from_pandas(obj, npartitions=npartitions)
        self._nrows = obj.shape[0]

    def _wrapped_apply(self, func, *args, **kwds):
        def wrapped():
            self._samp_pd.apply(func, *args, **kwds)

        return wrapped

    def _dask_apply(self, func, *args, **kwds):
        if self._progress_bar:
            with TQDMDaskProgressBar(desc="Dask Apply"):
                return self._obj_dd.apply(func, *args, **kwds).compute(scheduler=self._scheduler)
        else:
            return self._obj_dd.apply(func, *args, **kwds).compute(scheduler=self._scheduler)

    def apply(self, func, *args, **kwds):
        """
        Apply the function to the transformed swifter object
        """
        # estimate time to pandas apply
        wrapped = self._wrapped_apply(func, *args, **kwds)
        n_repeats = 3
        timed = timeit.timeit(wrapped, number=n_repeats)
        samp_proc_est = timed / n_repeats
        est_apply_duration = samp_proc_est / self._SAMP_SIZE * self._nrows

        # if pandas apply takes too long, use dask
        if est_apply_duration > self._dask_threshold:
            return self._dask_apply(func, *args, **kwds)
        else:  # use pandas
            if self._progress_bar:
                tqdm.pandas(desc="Pandas Apply")
                return self._obj_pd.progress_apply(func, *args, **kwds)
            else:
                return self._obj_pd.apply(func, *args, **kwds)


class Rolling(Transformation):
    def __init__(self, obj, npartitions=None, dask_threshold=1, scheduler="processes", progress_bar=True, **kwds):
        super(Rolling, self).__init__(obj, npartitions, dask_threshold, scheduler, progress_bar)
        self._samp_pd = self._samp_pd.rolling(**kwds)
        self._obj_pd = self._obj_pd.rolling(**kwds)
        kwds.pop("on")
        kwds.pop("closed")
        self._obj_dd = self._obj_dd.rolling(**kwds)
