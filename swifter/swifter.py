import sys
import timeit
import warnings
import numpy as np
import pandas as pd

from abc import abstractmethod
from math import ceil
from dask import dataframe as dd

from tqdm.auto import tqdm
from .tqdm_dask_progressbar import TQDMDaskProgressBar

from .base import (
    _SwifterBaseObject,
    suppress_stdout_stderr_logging,
    ERRORS_TO_HANDLE,
    SAMPLE_SIZE,
    N_REPEATS,
)


class _SwifterObject(_SwifterBaseObject):
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
        super().__init__(base_obj=pandas_obj, npartitions=npartitions)
        if self._obj.index.duplicated().any():
            warnings.warn(
                "This pandas object has duplicate indices, and swifter may not be able to improve performance. Consider resetting the indices with `df.reset_index(drop=True)`."
            )
        self._SAMPLE_SIZE = SAMPLE_SIZE if self._nrows > (25 * SAMPLE_SIZE) else int(ceil(self._nrows / 25))
        self._dask_threshold = dask_threshold
        self._scheduler = scheduler
        self._progress_bar = progress_bar
        self._progress_bar_desc = progress_bar_desc
        self._allow_dask_on_strings = allow_dask_on_strings

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
            npartitions=self._npartitions,
            dask_threshold=self._dask_threshold,
            scheduler=self._scheduler,
            progress_bar=self._progress_bar,
            progress_bar_desc=self._progress_bar_desc,
            allow_dask_on_strings=self._allow_dask_on_strings,
            **kwds,
        )

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
        origin=None,
        offset=None,
    ):
        """
        Create a swifter resampler object
        """
        kwds = {
            "rule": rule,
            "axis": axis,
            "closed": closed,
            "label": label,
            "convention": convention,
            "kind": kind,
            "loffset": loffset,
            "base": base,
            "on": on,
            "level": level,
            "origin": origin,
            "offset": offset,
        }
        if not base:
            kwds.pop("base")

        return Resampler(
            self._obj,
            npartitions=self._npartitions,
            dask_threshold=self._dask_threshold,
            scheduler=self._scheduler,
            progress_bar=self._progress_bar,
            progress_bar_desc=self._progress_bar_desc,
            allow_dask_on_strings=self._allow_dask_on_strings,
            **kwds,
        )


@pd.api.extensions.register_series_accessor("swifter")
class SeriesAccessor(_SwifterObject):
    def _wrapped_apply(self, func, convert_dtype=True, args=(), **kwds):
        def wrapped():
            with suppress_stdout_stderr_logging():
                self._obj.iloc[: self._SAMPLE_SIZE].apply(func, convert_dtype=convert_dtype, args=args, **kwds)

        return wrapped

    def _dask_apply(self, func, convert_dtype, *args, **kwds):
        sample = self._obj.iloc[: self._npartitions * 2]
        with suppress_stdout_stderr_logging():
            meta = sample.apply(func, convert_dtype=convert_dtype, args=args, **kwds)
        try:
            # check that the dask map partitions matches the pandas apply
            with suppress_stdout_stderr_logging():
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
        except ERRORS_TO_HANDLE:
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

        # if the series is empty, return early using Pandas
        if not self._nrows:
            return self._obj.apply(func, convert_dtype=convert_dtype, args=args, **kwds)

        sample = self._obj.iloc[: self._npartitions * 2]
        # check if input is string or if the user is overriding the string processing default
        allow_dask_processing = True if self._allow_dask_on_strings else (sample.dtype != "object")

        if "axis" in kwds.keys():
            kwds.pop("axis")
            warnings.warn("Axis keyword not necessary because applying on a Series.")

        try:  # try to vectorize
            with suppress_stdout_stderr_logging():
                tmp_df = func(sample, *args, **kwds)
                sample_df = sample.apply(func, convert_dtype=convert_dtype, args=args, **kwds)
                self._validate_apply(
                    np.array_equal(sample_df, tmp_df) & (sample_df.shape == tmp_df.shape),
                    error_message="Vectorized function sample doesn't match pandas apply sample.",
                )
            return func(self._obj, *args, **kwds)
        except ERRORS_TO_HANDLE:  # if can't vectorize, estimate time to pandas apply
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
    def _wrapped_apply(self, func, axis=0, raw=None, result_type=None, args=(), **kwds):
        def wrapped():
            with suppress_stdout_stderr_logging():
                self._obj.iloc[: self._SAMPLE_SIZE, :].apply(
                    func, axis=axis, raw=raw, result_type=result_type, args=args, **kwds
                )

        return wrapped

    def _modin_apply(self, func, axis=0, raw=None, result_type=None, *args, **kwds):
        sample = self._obj.iloc[: self._npartitions * 2, :]
        try:
            series = False
            with suppress_stdout_stderr_logging():
                import modin.pandas as md

                sample_df = sample.apply(func, axis=axis, raw=raw, result_type=result_type, args=args, **kwds)
                # check that the modin apply matches the pandas APPLY
                tmp_df = (
                    md.DataFrame(sample)
                    .apply(func, axis=axis, raw=raw, result_type=result_type, args=args, **kwds)
                    ._to_pandas()
                )
                if isinstance(sample_df, pd.Series) and isinstance(tmp_df, pd.DataFrame):
                    tmp_df = pd.Series(tmp_df.values[:, 0])
                    series = True
                self._validate_apply(
                    tmp_df.equals(sample_df), error_message="Modin apply sample does not match pandas apply sample."
                )
            output_df = (
                md.DataFrame(self._obj)
                .apply(func, *args, axis=axis, raw=raw, result_type=result_type, **kwds)
                ._to_pandas()
            )
            return pd.Series(output_df.values[:, 0]) if series else output_df
        except ERRORS_TO_HANDLE:
            if self._progress_bar:
                tqdm.pandas(desc=self._progress_bar_desc or "Pandas Apply")
                apply_func = self._obj.progress_apply
            else:
                apply_func = self._obj.apply
            return apply_func(func, axis=axis, raw=raw, result_type=result_type, args=args, **kwds)

    def _dask_apply(self, func, axis=0, raw=None, result_type=None, *args, **kwds):
        sample = self._obj.iloc[: self._npartitions * 2, :]
        with suppress_stdout_stderr_logging():
            meta = sample.apply(func, axis=axis, raw=raw, result_type=result_type, args=args, **kwds)
        try:
            with suppress_stdout_stderr_logging():
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
        except ERRORS_TO_HANDLE:
            # if dask apply doesn't match pandas apply, fallback to pandas
            if self._progress_bar:
                tqdm.pandas(desc=self._progress_bar_desc or "Pandas Apply")
                apply_func = self._obj.progress_apply
            else:
                apply_func = self._obj.apply

            return apply_func(func, axis=axis, raw=raw, result_type=result_type, args=args, **kwds)

    def apply(self, func, axis=0, raw=False, result_type=None, args=(), **kwds):
        """
        Apply the function to the DataFrame using swifter
        """
        # If there are no rows return early using Pandas
        if not self._nrows:
            return self._obj.apply(func, axis=axis, raw=raw, result_type=result_type, args=args, **kwds)

        sample = self._obj.iloc[: self._npartitions * 2, :]
        # check if input is string or if the user is overriding the string processing default
        allow_dask_processing = True if self._allow_dask_on_strings else ("object" not in sample.dtypes.values)

        try:  # try to vectorize
            with suppress_stdout_stderr_logging():
                tmp_df = func(sample, *args, **kwds)
                sample_df = sample.apply(func, axis=axis, raw=raw, result_type=result_type, args=args, **kwds)
                self._validate_apply(
                    np.array_equal(sample_df, tmp_df) & (sample_df.shape == tmp_df.shape),
                    error_message="Vectorized function sample does not match pandas apply sample.",
                )
            return func(self._obj, *args, **kwds)
        except ERRORS_TO_HANDLE:  # if can't vectorize, estimate time to pandas apply
            wrapped = self._wrapped_apply(func, axis=axis, raw=raw, result_type=result_type, args=args, **kwds)
            timed = timeit.timeit(wrapped, number=N_REPEATS)
            sample_proc_est = timed / N_REPEATS
            est_apply_duration = sample_proc_est / self._SAMPLE_SIZE * self._obj.shape[0]

            # if pandas sample apply takes too long and not performing str processing, use dask
            if (
                (est_apply_duration > self._dask_threshold)
                and (allow_dask_processing or ("modin.pandas" not in sys.modules))
                and axis == 1
            ):
                return self._dask_apply(func, axis, raw, result_type, *args, **kwds)
            # if not dask, use modin for string processing
            elif (est_apply_duration > self._dask_threshold) and axis == 1:
                return self._modin_apply(func, axis, raw, result_type, *args, **kwds)
            else:  # use pandas
                if self._progress_bar:
                    tqdm.pandas(desc=self._progress_bar_desc or "Pandas Apply")
                    apply_func = self._obj.progress_apply
                else:
                    apply_func = self._obj.apply

                return apply_func(func, axis=axis, raw=raw, result_type=result_type, args=args, **kwds)

    def _wrapped_applymap(self, func):
        def wrapped():
            with suppress_stdout_stderr_logging():
                self._obj.iloc[: self._SAMPLE_SIZE, :].applymap(func)

        return wrapped

    def _dask_applymap(self, func):
        sample = self._obj.iloc[: self._npartitions * 2, :]
        with suppress_stdout_stderr_logging():
            meta = sample.applymap(func)
        try:
            with suppress_stdout_stderr_logging():
                # check that the dask apply matches the pandas apply
                tmp_df = (
                    dd.from_pandas(sample, npartitions=self._npartitions)
                    .applymap(func, meta=meta)
                    .compute(scheduler=self._scheduler)
                )
                self._validate_apply(
                    tmp_df.equals(meta), error_message="Dask applymap sample does not match pandas applymap sample."
                )
            if self._progress_bar:
                with TQDMDaskProgressBar(desc=self._progress_bar_desc or "Dask Applymap"):
                    return (
                        dd.from_pandas(self._obj, npartitions=self._npartitions)
                        .applymap(func, meta=meta)
                        .compute(scheduler=self._scheduler)
                    )
            else:
                return (
                    dd.from_pandas(self._obj, npartitions=self._npartitions)
                    .applymap(func, meta=meta)
                    .compute(scheduler=self._scheduler)
                )
        except ERRORS_TO_HANDLE:
            # if dask apply doesn't match pandas apply, fallback to pandas
            if self._progress_bar:
                tqdm.pandas(desc=self._progress_bar_desc or "Pandas Apply")
                applymap_func = self._obj.progress_applymap
            else:
                applymap_func = self._obj.applymap

            return applymap_func(func)

    def applymap(self, func):
        """
        Applymap the function to the DataFrame using swifter
        """

        # If there are no rows return early using Pandas
        if not self._nrows:
            return self._obj.applymap(func)

        sample = self._obj.iloc[: self._npartitions * 2, :]
        # check if input is string or if the user is overriding the string processing default
        allow_dask_processing = True if self._allow_dask_on_strings else ("object" not in sample.dtypes.values)

        try:  # try to vectorize
            with suppress_stdout_stderr_logging():
                tmp_df = func(sample)
                sample_df = sample.applymap(func)
                self._validate_apply(
                    np.array_equal(sample_df, tmp_df) & (sample_df.shape == tmp_df.shape),
                    error_message="Vectorized function sample does not match pandas apply sample.",
                )
            return func(self._obj)
        except ERRORS_TO_HANDLE:  # if can't vectorize, estimate time to pandas apply
            wrapped = self._wrapped_applymap(func)
            timed = timeit.timeit(wrapped, number=N_REPEATS)
            sample_proc_est = timed / N_REPEATS
            est_apply_duration = sample_proc_est / self._SAMPLE_SIZE * self._obj.shape[0]

            # if pandas sample apply takes too long and not performing str processing, use dask
            if (est_apply_duration > self._dask_threshold) and allow_dask_processing:
                return self._dask_applymap(func)
            else:  # use pandas
                if self._progress_bar:
                    tqdm.pandas(desc=self._progress_bar_desc or "Pandas Apply")
                    applymap_func = self._obj.progress_applymap
                else:
                    applymap_func = self._obj.applymap

                return applymap_func(func)


class Transformation(_SwifterObject):
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
        super(Transformation, self).__init__(
            pandas_obj, npartitions, dask_threshold, scheduler, progress_bar, progress_bar_desc, allow_dask_on_strings
        )
        self._sample_pd = pandas_obj.iloc[: self._SAMPLE_SIZE]
        self._obj_pd = pandas_obj
        self._obj_dd = dd.from_pandas(pandas_obj, npartitions=npartitions)
        self._nrows = pandas_obj.shape[0]

    def _wrapped_apply(self, func, *args, **kwds):
        def wrapped():
            with suppress_stdout_stderr_logging():
                self._sample_pd.apply(func, *args, **kwds)

        return wrapped

    @abstractmethod
    def _dask_apply(self, func, *args, **kwds):
        raise NotImplementedError("Transformation class does not implement _dask_apply")

    def apply(self, func, *args, **kwds):
        """
        Apply the function to the transformed swifter object
        """
        # if the transformed dataframe is empty, return early using Pandas
        if not self._nrows:
            return self._obj_pd.apply(func, args=args, **kwds)

        # estimate time to pandas apply
        wrapped = self._wrapped_apply(func, *args, **kwds)
        timed = timeit.timeit(wrapped, number=N_REPEATS)
        sample_proc_est = timed / N_REPEATS
        est_apply_duration = sample_proc_est / self._SAMPLE_SIZE * self._nrows

        # No `allow_dask_processing` variable here, because we don't know the dtypes of the transformation
        if est_apply_duration > self._dask_threshold:
            return self._dask_apply(func, *args, **kwds)
        else:  # use pandas
            if self._progress_bar and hasattr(self._obj_pd, "progress_apply"):
                tqdm.pandas(desc=self._progress_bar_desc or "Pandas Apply")
                return self._obj_pd.progress_apply(func, *args, **kwds)
            else:
                return self._obj_pd.apply(func, *args, **kwds)


class Rolling(Transformation):
    def __init__(
        self,
        pandas_obj,
        npartitions=None,
        dask_threshold=1,
        scheduler="processes",
        progress_bar=True,
        progress_bar_desc=None,
        allow_dask_on_strings=False,
        **kwds,
    ):
        super(Rolling, self).__init__(
            pandas_obj,
            npartitions=npartitions,
            dask_threshold=dask_threshold,
            scheduler=scheduler,
            progress_bar=progress_bar,
            progress_bar_desc=progress_bar_desc,
            allow_dask_on_strings=allow_dask_on_strings,
        )
        self._rolling_kwds = kwds.copy()
        self._comparison_pd = self._obj_pd.iloc[: self._npartitions * 2]
        self._sample_pd = self._sample_pd.rolling(**kwds)
        self._obj_pd = self._obj_pd.rolling(**kwds)
        self._obj_dd = self._obj_dd.rolling(**{k: v for k, v in kwds.items() if k not in ["on", "closed"]})

    def _dask_apply(self, func, *args, **kwds):
        try:
            # check that the dask rolling apply matches the pandas apply
            with suppress_stdout_stderr_logging():
                tmp_df = (
                    dd.from_pandas(self._comparison_pd, npartitions=self._npartitions)
                    .rolling(**{k: v for k, v in self._rolling_kwds.items() if k not in ["on", "closed"]})
                    .apply(func, *args, **kwds)
                    .compute(scheduler=self._scheduler)
                )
                self._validate_apply(
                    tmp_df.equals(self._comparison_pd.rolling(**self._rolling_kwds).apply(func, *args, **kwds)),
                    error_message="Dask rolling apply sample does not match pandas rolling apply sample.",
                )
            if self._progress_bar:
                with TQDMDaskProgressBar(desc=self._progress_bar_desc or "Dask Apply"):
                    return self._obj_dd.apply(func, *args, **kwds).compute(scheduler=self._scheduler)
            else:
                return self._obj_dd.apply(func, *args, **kwds).compute(scheduler=self._scheduler)
        except ERRORS_TO_HANDLE:
            if self._progress_bar:
                tqdm.pandas(desc=self._progress_bar_desc or "Pandas Apply")
                return self._obj_pd.progress_apply(func, *args, **kwds)
            else:
                return self._obj_pd.apply(func, *args, **kwds)


class Resampler(Transformation):
    def __init__(
        self,
        pandas_obj,
        npartitions=None,
        dask_threshold=1,
        scheduler="processes",
        progress_bar=True,
        progress_bar_desc=None,
        allow_dask_on_strings=False,
        **kwds,
    ):
        super(Resampler, self).__init__(
            pandas_obj,
            npartitions=npartitions,
            dask_threshold=dask_threshold,
            scheduler=scheduler,
            progress_bar=progress_bar,
            progress_bar_desc=progress_bar_desc,
            allow_dask_on_strings=allow_dask_on_strings,
        )
        self._resampler_kwds = kwds.copy()
        self._comparison_pd = self._obj_pd.iloc[: self._npartitions * 2]
        self._sample_pd = self._sample_pd.resample(**kwds)
        self._obj_pd = self._obj_pd.resample(**kwds)
        # Setting dask dataframe `self._obj_dd` to None when there are 0 `self._nrows` because
        # swifter will immediately return the pandas form during the apply function if there are 0 `self._nrows`
        self._obj_dd = (
            self._obj_dd.resample(**{k: v for k, v in kwds.items() if k in ["rule", "closed", "label"]})
            if self._nrows
            else None
        )

    def _dask_apply(self, func, *args, **kwds):
        try:
            # check that the dask resampler apply matches the pandas apply
            with suppress_stdout_stderr_logging():
                tmp_df = (
                    dd.from_pandas(self._comparison_pd, npartitions=self._npartitions)
                    .resample(**{k: v for k, v in self._resampler_kwds.items() if k in ["rule", "closed", "label"]})
                    .agg(func, *args, **kwds)
                    .compute(scheduler=self._scheduler)
                )
                self._validate_apply(
                    tmp_df.equals(self._comparison_pd.resample(**self._resampler_kwds).apply(func, *args, **kwds)),
                    error_message="Dask resampler apply sample does not match pandas resampler apply sample.",
                )

            if self._progress_bar:
                with TQDMDaskProgressBar(desc=self._progress_bar_desc or "Dask Apply"):
                    return self._obj_dd.agg(func, *args, **kwds).compute(scheduler=self._scheduler)
            else:
                return self._obj_dd.agg(func, *args, **kwds).compute(scheduler=self._scheduler)
        except ERRORS_TO_HANDLE:
            # use pandas -- no progress_apply available for resampler objects
            return self._obj_pd.apply(func, *args, **kwds)
