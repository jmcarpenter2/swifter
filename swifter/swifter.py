import timeit
import warnings
import pandas as pd

from math import ceil
from psutil import cpu_count
from dask import dataframe as dd

from tqdm.auto import tqdm
from .tqdm_dask_progressbar import TQDMDaskProgressBar

from numba.errors import TypingError

SAMPLE_SIZE = 1000
N_REPEATS = 3


class _SwifterObject:
    def __init__(
        self,
        pandas_obj,
        npartitions=None,
        dask_threshold=1,
        scheduler="processes",
        progress_bar=True,
        progress_bar_desc=None,
        allow_dask_on_strings=False,
    ):
        self._obj = pandas_obj
        self._nrows = self._obj.shape[0]
        self._SAMPLE_SIZE = SAMPLE_SIZE if self._nrows > 25000 else int(ceil(self._nrows / 25))

        if npartitions is None:
            self._npartitions = cpu_count() * 2
        else:
            self._npartitions = npartitions
        self._dask_threshold = dask_threshold
        self._scheduler = scheduler
        self._progress_bar = progress_bar
        self._progress_bar_desc = progress_bar_desc
        self._allow_dask_on_strings = allow_dask_on_strings

    @staticmethod
    def _validate_apply(expr, error_message):
        if not expr:
            raise ValueError(error_message)

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

    def progress_bar(self, enable=True, desc=None):
        """
        Turn on/off the progress bar, and optionally add a custom description
        """
        self._progress_bar = enable
        self._progress_bar_desc = desc
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
        return Rolling(
            self._obj,
            self._npartitions,
            self._dask_threshold,
            self._scheduler,
            self._progress_bar,
            self._progress_bar_desc,
            **kwds
        )


@pd.api.extensions.register_series_accessor("swifter")
class SeriesAccessor(_SwifterObject):
    def _wrapped_apply(self, func, convert_dtype=True, args=(), **kwds):
        def wrapped():
            self._obj.iloc[: self._SAMPLE_SIZE].apply(func, convert_dtype=convert_dtype, args=args, **kwds)

        return wrapped

    def _dask_apply(self, func, convert_dtype, *args, **kwds):
        sample = self._obj.iloc[: self._npartitions * 2]
        meta = sample.apply(func, convert_dtype=convert_dtype, args=args, **kwds)
        try:
            # check that the dask map partitions matches the pandas apply
            tmp_df = (
                dd.from_pandas(sample, npartitions=self._npartitions)
                .map_partitions(func, *args, meta=meta, **kwds)
                .compute(scheduler=self._scheduler)
            )
            self._validate_apply(
                tmp_df.equals(meta), error_message="Dask map-partitions sample does not match pandas apply sample."
            )
            if self._progress_bar:
                with TQDMDaskProgressBar(desc=self._progress_bar_desc or "Dask Apply"):
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
        except (AttributeError, ValueError, TypeError, KeyError):
            # if map partitions doesn't match pandas apply, we can use dask apply, but it will be a bit slower
            if self._progress_bar:
                with TQDMDaskProgressBar(desc=self._progress_bar_desc or "Dask Apply"):
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
        sample = self._obj.iloc[: self._npartitions * 2]
        # check if input is string or if the user is overriding the string processing default
        allow_dask_processing = True if self._allow_dask_on_strings else (sample.dtype != "object")

        if "axis" in kwds.keys():
            kwds.pop("axis")
            warnings.warn("Axis keyword not necessary because applying on a Series.")

        try:  # try to vectorize
            tmp_df = func(sample, *args, **kwds)
            self._validate_apply(
                sample.apply(func, convert_dtype=convert_dtype, args=args, **kwds).equals(tmp_df),
                error_message="Vectorized function sample doesn't match pandas apply sample.",
            )
            return func(self._obj, *args, **kwds)
        except (
            AttributeError,
            ValueError,
            TypeError,
            TypingError,
            KeyError,
        ):  # if can't vectorize, estimate time to pandas apply
            wrapped = self._wrapped_apply(func, convert_dtype=convert_dtype, args=args, **kwds)
            timed = timeit.timeit(wrapped, number=N_REPEATS)
            sample_proc_est = timed / N_REPEATS
            est_apply_duration = sample_proc_est / self._SAMPLE_SIZE * self._obj.shape[0]

            # if pandas sample apply takes too long and not performing str processing, use dask
            if (est_apply_duration > self._dask_threshold) and allow_dask_processing:
                return self._dask_apply(func, convert_dtype, *args, **kwds)
            else:  # use pandas
                if self._progress_bar:
                    tqdm.pandas(desc=self._progress_bar_desc or "Pandas Apply")
                    return self._obj.progress_apply(func, convert_dtype=convert_dtype, args=args, **kwds)
                else:
                    return self._obj.apply(func, convert_dtype=convert_dtype, args=args, **kwds)


@pd.api.extensions.register_dataframe_accessor("swifter")
class DataFrameAccessor(_SwifterObject):
    def _wrapped_apply(self, func, axis=0, broadcast=None, raw=False, reduce=None, result_type=None, args=(), **kwds):
        def wrapped():
            self._obj.iloc[: self._SAMPLE_SIZE, :].apply(
                func, axis=axis, broadcast=broadcast, raw=raw, reduce=reduce, result_type=result_type, args=args, **kwds
            )

        return wrapped

    def _dask_apply(self, func, axis=0, broadcast=None, raw=False, reduce=None, result_type=None, *args, **kwds):
        sample = self._obj.iloc[: self._npartitions * 2, :]
        meta = sample.apply(
            func, axis=axis, broadcast=broadcast, raw=raw, reduce=reduce, result_type=result_type, args=args, **kwds
        )
        try:
            if broadcast:
                result_type = "broadcast"
            elif reduce:
                result_type = "reduce"

            # check that the dask apply matches the pandas apply
            tmp_df = (
                dd.from_pandas(sample, npartitions=self._npartitions)
                .apply(func, *args, axis=axis, raw=raw, result_type=result_type, meta=meta, **kwds)
                .compute(scheduler=self._scheduler)
            )
            self._validate_apply(
                tmp_df.equals(meta), error_message="Dask apply sample does not match pandas apply sample."
            )
            if self._progress_bar:
                with TQDMDaskProgressBar(desc=self._progress_bar_desc or "Dask Apply"):
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
        except (AttributeError, ValueError, TypeError, KeyError):
            # if dask apply doesn't match pandas apply, fallback to pandas
            if self._progress_bar:
                tqdm.pandas(desc=self._progress_bar_desc or "Pandas Apply")
                apply_func = self._obj.progress_apply
            else:
                apply_func = self._obj.apply

            return apply_func(
                func, axis=axis, broadcast=broadcast, raw=raw, reduce=reduce, result_type=result_type, args=args, **kwds
            )

    def apply(self, func, axis=0, broadcast=None, raw=False, reduce=None, result_type=None, args=(), **kwds):
        """
        Apply the function to the DataFrame using swifter
        """
        sample = self._obj.iloc[: self._npartitions * 2, :]
        # check if input is string or if the user is overriding the string processing default
        allow_dask_processing = True if self._allow_dask_on_strings else ("object" not in sample.dtypes.values)

        try:  # try to vectorize
            tmp_df = func(sample, *args, **kwds)
            self._validate_apply(
                sample.apply(
                    func,
                    axis=axis,
                    broadcast=broadcast,
                    raw=raw,
                    reduce=reduce,
                    result_type=result_type,
                    args=args,
                    **kwds
                ).equals(tmp_df),
                error_message="Vectorized function sample does not match pandas apply sample.",
            )
            return func(self._obj, *args, **kwds)
        except (
            AttributeError,
            ValueError,
            TypeError,
            TypingError,
            KeyError,
        ):  # if can't vectorize, estimate time to pandas apply
            wrapped = self._wrapped_apply(
                func, axis=axis, broadcast=broadcast, raw=raw, reduce=reduce, result_type=result_type, args=args, **kwds
            )
            timed = timeit.timeit(wrapped, number=N_REPEATS)
            sample_proc_est = timed / N_REPEATS
            est_apply_duration = sample_proc_est / self._SAMPLE_SIZE * self._obj.shape[0]

            # if pandas sample apply takes too long and not performing str processing, use dask
            if (est_apply_duration > self._dask_threshold) and allow_dask_processing:
                if axis == 0:
                    raise NotImplementedError(
                        "Swifter cannot perform axis=0 applies on large datasets.\n"
                        "Dask currently does not have an axis=0 apply implemented.\n"
                        "More details at https://github.com/jmcarpenter2/swifter/issues/10"
                    )
                return self._dask_apply(func, axis, broadcast, raw, reduce, result_type, *args, **kwds)
            else:  # use pandas
                if self._progress_bar:
                    tqdm.pandas(desc=self._progress_bar_desc or "Pandas Apply")
                    apply_func = self._obj.progress_apply
                else:
                    apply_func = self._obj.apply

                return apply_func(
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
        progress_bar_desc=None,
        allow_dask_on_strings=False,
    ):
        super().__init__(
            obj, npartitions, dask_threshold, scheduler, progress_bar, progress_bar_desc, allow_dask_on_strings
        )
        self._sample_pd = obj.iloc[: self._SAMPLE_SIZE]
        self._obj_pd = obj
        self._obj_dd = dd.from_pandas(obj, npartitions=npartitions)
        self._nrows = obj.shape[0]

    def _wrapped_apply(self, func, *args, **kwds):
        def wrapped():
            self._sample_pd.apply(func, *args, **kwds)

        return wrapped

    def _dask_apply(self, func, *args, **kwds):
        if self._progress_bar:
            with TQDMDaskProgressBar(desc=self._progress_bar_desc or "Dask Apply"):
                return self._obj_dd.apply(func, *args, **kwds).compute(scheduler=self._scheduler)
        else:
            return self._obj_dd.apply(func, *args, **kwds).compute(scheduler=self._scheduler)

    def apply(self, func, *args, **kwds):
        """
        Apply the function to the transformed swifter object
        """
        # estimate time to pandas apply
        wrapped = self._wrapped_apply(func, *args, **kwds)
        timed = timeit.timeit(wrapped, number=N_REPEATS)
        sample_proc_est = timed / N_REPEATS
        est_apply_duration = sample_proc_est / self._SAMPLE_SIZE * self._nrows

        # if pandas sample apply takes too long, use dask
        if est_apply_duration > self._dask_threshold:
            return self._dask_apply(func, *args, **kwds)
        else:  # use pandas
            if self._progress_bar:
                tqdm.pandas(desc=self._progress_bar_desc or "Pandas Apply")
                return self._obj_pd.progress_apply(func, *args, **kwds)
            else:
                return self._obj_pd.apply(func, *args, **kwds)


class Rolling(Transformation):
    def __init__(
        self,
        obj,
        npartitions=None,
        dask_threshold=1,
        scheduler="processes",
        progress_bar=True,
        progress_bar_desc=None,
        **kwds
    ):
        super(Rolling, self).__init__(obj, npartitions, dask_threshold, scheduler, progress_bar, progress_bar_desc)
        self._sample_pd = self._sample_pd.rolling(**kwds)
        self._obj_pd = self._obj_pd.rolling(**kwds)
        kwds.pop("on")
        kwds.pop("closed")
        self._obj_dd = self._obj_dd.rolling(**kwds)
