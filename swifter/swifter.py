import timeit
import warnings

import numpy as np
import pandas as pd

from abc import abstractmethod
from dask import dataframe as dd
from functools import partial
from tqdm.auto import tqdm
from .tqdm_dask_progressbar import TQDMDaskProgressBar

from .base import (
    _SwifterBaseObject,
    suppress_stdout_stderr_logging,
    ERRORS_TO_HANDLE,
    RAY_INSTALLED,
    N_REPEATS,
)

DEFAULT_KWARGS = {
    "npartitions": None,
    "dask_threshold": 1,
    "scheduler": "processes",
    "progress_bar": True,
    "progress_bar_desc": None,
    "allow_dask_on_strings": False,
    "force_parallel": False,
}

GROUPBY_MAX_ROWS_PANDAS_DEFAULT = 5000


def register_default_config_dataframe_accessor(dataframe_to_register, kwargs):
    """
    Register dataframe type with default swifter config
    """
    current_init = dataframe_to_register.__init__

    def new_init(self, *args, **kwds):
        current_init(self, *args, **kwds)
        self.swifter = (
            self.swifter.set_npartitions(npartitions=kwargs.get("npartitions", DEFAULT_KWARGS["npartitions"]))
            .set_dask_threshold(dask_threshold=kwargs.get("dask_threshold", DEFAULT_KWARGS["dask_threshold"]))
            .set_dask_scheduler(scheduler=kwargs.get("scheduler", DEFAULT_KWARGS["scheduler"]))
            .progress_bar(
                enable=kwargs.get("progress_bar", DEFAULT_KWARGS["progress_bar"]),
                desc=kwargs.get("progress_bar_desc", DEFAULT_KWARGS["progress_bar_desc"]),
            )
            .allow_dask_on_strings(enable=kwargs.get("allow_dask_on_strings", DEFAULT_KWARGS["allow_dask_on_strings"]))
            .force_parallel(enable=kwargs.get("force_parallel", DEFAULT_KWARGS["force_parallel"]))
        )

    dataframe_to_register.__init__ = new_init


def set_defaults(**kwargs):
    """
    Register swifter's default kwargs
        npartitions=None,
        dask_threshold=1,
        scheduler="processes",
        progress_bar=True,
        progress_bar_desc=None,
        allow_dask_on_strings=False,
    """
    from pandas import Series, DataFrame

    register_default_config_dataframe_accessor(Series, kwargs)
    register_default_config_dataframe_accessor(DataFrame, kwargs)


class _SwifterObject(_SwifterBaseObject):
    def __init__(
        self,
        pandas_obj,
        npartitions=DEFAULT_KWARGS["npartitions"],
        dask_threshold=DEFAULT_KWARGS["dask_threshold"],
        scheduler=DEFAULT_KWARGS["scheduler"],
        progress_bar=DEFAULT_KWARGS["progress_bar"],
        progress_bar_desc=DEFAULT_KWARGS["progress_bar_desc"],
        allow_dask_on_strings=DEFAULT_KWARGS["allow_dask_on_strings"],
        force_parallel=DEFAULT_KWARGS["force_parallel"],
    ):
        super().__init__(base_obj=pandas_obj, npartitions=npartitions)
        if self._obj.index.duplicated().any():
            warnings.warn(
                "This pandas object has duplicate indices, "
                "and swifter may not be able to improve performance. "
                "Consider resetting the indices with `df.reset_index(drop=True)`."
            )
        self._dask_threshold = dask_threshold
        self._scheduler = scheduler
        self._progress_bar = progress_bar
        self._progress_bar_desc = progress_bar_desc
        self._allow_dask_on_strings = allow_dask_on_strings
        self._force_parallel = force_parallel

    def set_dask_threshold(self, dask_threshold=1):
        """
        Set the threshold (seconds) for maximum allowed estimated duration
        of pandas apply before switching to dask
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
        Override the string processing default, which is to not use dask
        if a string is contained in the pandas object
        """
        self._allow_dask_on_strings = enable
        return self

    def force_parallel(self, enable=True):
        """
        Force swifter to use dask parallel processing, without attempting any
        vectorized solution or estimating pandas apply duration to determine
        what will be the fastest approach
        """
        self._force_parallel = enable
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
            force_parallel=self._force_parallel,
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
            force_parallel=self._force_parallel,
            **kwds,
        )


@pd.api.extensions.register_series_accessor("swifter")
class SeriesAccessor(_SwifterObject):
    def _wrapped_apply(self, func, convert_dtype=True, args=(), **kwds):
        def wrapped():
            with suppress_stdout_stderr_logging():
                self._obj.iloc[self._SAMPLE_INDEX].apply(func, convert_dtype=convert_dtype, args=args, **kwds)

        return wrapped

    def _pandas_apply(self, df, func, convert_dtype, *args, **kwds):
        if self._progress_bar:
            tqdm.pandas(desc=self._progress_bar_desc or "Pandas Apply")
            return df.progress_apply(func, convert_dtype=convert_dtype, args=args, **kwds)
        else:
            return df.apply(func, convert_dtype=convert_dtype, args=args, **kwds)

    def _dask_map_partitions(self, df, func, meta, *args, **kwds):
        return (
            dd.from_pandas(df, npartitions=self._npartitions)
            .map_partitions(func, *args, meta=meta, **kwds)
            .compute(scheduler=self._scheduler)
        )

    def _dask_apply(self, df, func, convert_dtype, meta, *args, **kwds):
        return (
            dd.from_pandas(df, npartitions=self._npartitions)
            .apply(
                lambda x: func(x, *args, **kwds),
                convert_dtype=convert_dtype,
                meta=meta,
            )
            .compute(scheduler=self._scheduler)
        )

    def _parallel_apply(self, func, convert_dtype, *args, **kwds):
        sample = self._obj.iloc[self._SAMPLE_INDEX]
        with suppress_stdout_stderr_logging():
            meta = sample.apply(func, convert_dtype=convert_dtype, args=args, **kwds)
        try:
            # check that the dask map partitions matches the pandas apply
            with suppress_stdout_stderr_logging():
                tmp_df = self._dask_map_partitions(sample, func, meta, *args, **kwds)
                self._validate_apply(
                    tmp_df.equals(meta),
                    error_message=("Dask map-partitions sample does not match pandas apply sample."),
                )
            if self._progress_bar:
                with TQDMDaskProgressBar(desc=self._progress_bar_desc or "Dask Apply"):
                    return self._dask_map_partitions(self._obj, func, meta, *args, **kwds)
            else:
                return self._dask_map_partitions(self._obj, func, meta, *args, **kwds)
        except ERRORS_TO_HANDLE:
            # if map partitions doesn't match pandas apply,
            # we can use dask apply, but it will be a bit slower
            try:
                if self._progress_bar:
                    with TQDMDaskProgressBar(desc=self._progress_bar_desc or "Dask Apply"):
                        return self._dask_apply(self._obj, func, convert_dtype, meta, *args, **kwds)
                else:
                    return self._dask_apply(self._obj, func, convert_dtype, meta, *args, **kwds)
            except ERRORS_TO_HANDLE:
                # Second fallback to pandas if dask apply fails
                return self._pandas_apply(self._obj, func, convert_dtype, *args, **kwds)

    def apply(self, func, convert_dtype=True, args=(), **kwds):
        """
        Apply the function to the Series using swifter
        """

        # if the series is empty, return early using Pandas
        if not self._nrows:
            return self._obj.apply(func, convert_dtype=convert_dtype, args=args, **kwds)

        # If parallel processing is forced by the user, then skip the logic and apply dask
        if self._force_parallel:
            return self._parallel_apply(func, convert_dtype, *args, **kwds)

        sample = self._obj.iloc[self._SAMPLE_INDEX]
        # check if input is string or
        # if the user is overriding the string processing default
        allow_dask_processing = True if self._allow_dask_on_strings else (sample.dtype != "object")

        if "axis" in kwds.keys():
            kwds.pop("axis")
            warnings.warn("Axis keyword not necessary because applying on a Series.")

        try:  # try to vectorize
            with suppress_stdout_stderr_logging():
                tmp_df = func(sample, *args, **kwds)
                sample_df = sample.apply(func, convert_dtype=convert_dtype, args=args, **kwds)
                self._validate_apply(
                    np.array_equal(sample_df, tmp_df) & (hasattr(tmp_df, "shape")) & (sample_df.shape == tmp_df.shape),
                    error_message=("Vectorized function sample doesn't match pandas apply sample."),
                )
            return func(self._obj, *args, **kwds)
        except ERRORS_TO_HANDLE:  # if can't vectorize, estimate time to pandas apply
            wrapped = self._wrapped_apply(func, convert_dtype=convert_dtype, args=args, **kwds)
            timed = timeit.timeit(wrapped, number=N_REPEATS)
            sample_proc_est = timed / N_REPEATS
            est_apply_duration = sample_proc_est / self._SAMPLE_SIZE * self._nrows

            # if pandas sample apply takes too long and not performing str processing
            # then use dask
            if (est_apply_duration > self._dask_threshold) and allow_dask_processing:
                return self._parallel_apply(func, convert_dtype, *args, **kwds)
            else:  # use pandas
                return self._pandas_apply(self._obj, func, convert_dtype, *args, **kwds)


@pd.api.extensions.register_dataframe_accessor("swifter")
class DataFrameAccessor(_SwifterObject):
    def _wrapped_apply(self, func, axis=0, raw=None, result_type=None, args=(), **kwds):
        def wrapped():
            with suppress_stdout_stderr_logging():
                self._obj.iloc[self._SAMPLE_INDEX].apply(
                    func, axis=axis, raw=raw, result_type=result_type, args=args, **kwds
                )

        return wrapped

    def _pandas_apply(self, df, func, axis, raw, result_type, *args, **kwds):
        if self._progress_bar:
            tqdm.pandas(desc=self._progress_bar_desc or "Pandas Apply")
            apply_func = df.progress_apply
        else:
            apply_func = df.apply

        return apply_func(func, axis=axis, raw=raw, result_type=result_type, args=args, **kwds)

    def _dask_apply(self, df, func, axis, raw, result_type, meta, *args, **kwds):
        return (
            dd.from_pandas(df, npartitions=self._npartitions)
            .apply(
                func,
                *args,
                axis=axis,
                raw=raw,
                result_type=result_type,
                meta=meta,
                **kwds,
            )
            .compute(scheduler=self._scheduler)
        )

    def _parallel_apply(self, func, axis=0, raw=None, result_type=None, *args, **kwds):
        sample = self._obj.iloc[self._SAMPLE_INDEX]
        with suppress_stdout_stderr_logging():
            meta = sample.apply(func, axis=axis, raw=raw, result_type=result_type, args=args, **kwds)
        try:
            with suppress_stdout_stderr_logging():
                # check that the dask apply matches the pandas apply
                tmp_df = (
                    dd.from_pandas(sample, npartitions=self._npartitions)
                    .apply(
                        func,
                        *args,
                        axis=axis,
                        raw=raw,
                        result_type=result_type,
                        meta=meta,
                        **kwds,
                    )
                    .compute(scheduler=self._scheduler)
                )
                self._validate_apply(
                    tmp_df.equals(meta),
                    error_message="Dask apply sample does not match pandas apply sample.",
                )
            if self._progress_bar:
                with TQDMDaskProgressBar(desc=self._progress_bar_desc or "Dask Apply"):
                    return self._dask_apply(self._obj, func, axis, raw, result_type, meta, *args, **kwds)
            else:
                return self._dask_apply(self._obj, func, axis, raw, result_type, meta, *args, **kwds)
        except ERRORS_TO_HANDLE:
            # if dask apply doesn't match pandas apply, fallback to pandas
            return self._pandas_apply(self._obj, func, axis, raw, result_type, *args, **kwds)

    def apply(self, func, axis=0, raw=False, result_type=None, args=(), **kwds):
        """
        Apply the function to the DataFrame using swifter
        """
        # If there are no rows return early using Pandas
        if not self._nrows:
            return self._obj.apply(func, axis=axis, raw=raw, result_type=result_type, args=args, **kwds)

        # If parallel processing is forced by the user, then skip the logic and apply dask
        if self._force_parallel:
            return self._parallel_apply(func, axis, raw, result_type, *args, **kwds)

        sample = self._obj.iloc[self._SAMPLE_INDEX]
        # check if input is string
        # or if the user is overriding the string processing default
        allow_dask_processing = True if self._allow_dask_on_strings else ("object" not in sample.dtypes.values)

        try:  # try to vectorize
            with suppress_stdout_stderr_logging():
                tmp_df = func(sample, *args, **kwds)
                sample_df = sample.apply(func, axis=axis, raw=raw, result_type=result_type, args=args, **kwds)
                self._validate_apply(
                    np.array_equal(sample_df, tmp_df) & (hasattr(tmp_df, "shape")) & (sample_df.shape == tmp_df.shape),
                    error_message=("Vectorized function sample does not match pandas apply sample."),
                )
            return func(self._obj, *args, **kwds)
        except ERRORS_TO_HANDLE:  # if can't vectorize, estimate time to pandas apply
            wrapped = self._wrapped_apply(func, axis=axis, raw=raw, result_type=result_type, args=args, **kwds)
            timed = timeit.timeit(wrapped, number=N_REPEATS)
            sample_proc_est = timed / N_REPEATS
            est_apply_duration = sample_proc_est / self._SAMPLE_SIZE * self._nrows

            # if pandas sample apply takes too long
            # and not performing str processing, use dask
            if (est_apply_duration > self._dask_threshold) and allow_dask_processing and axis == 1:
                return self._parallel_apply(func, axis, raw, result_type, *args, **kwds)
            else:  # use pandas
                return self._pandas_apply(self._obj, func, axis, raw, result_type, *args, **kwds)

    def _wrapped_applymap(self, func):
        def wrapped():
            with suppress_stdout_stderr_logging():
                self._obj.iloc[self._SAMPLE_INDEX].applymap(func)

        return wrapped

    def _pandas_applymap(self, df, func):
        if self._progress_bar:
            tqdm.pandas(desc=self._progress_bar_desc or "Pandas Apply")
            applymap_func = df.progress_applymap
        else:
            applymap_func = df.applymap

        return applymap_func(func)

    def _dask_applymap(self, df, func, meta):
        return (
            dd.from_pandas(df, npartitions=self._npartitions)
            .applymap(func, meta=meta)
            .compute(scheduler=self._scheduler)
        )

    def _parallel_applymap(self, func):
        sample = self._obj.iloc[self._SAMPLE_INDEX]
        with suppress_stdout_stderr_logging():
            meta = sample.applymap(func)
        try:
            with suppress_stdout_stderr_logging():
                # check that the dask apply matches the pandas apply
                tmp_df = self._dask_applymap(sample, func, meta)
                self._validate_apply(
                    tmp_df.equals(meta),
                    error_message=("Dask applymap sample does not match pandas applymap sample."),
                )
            if self._progress_bar:
                with TQDMDaskProgressBar(desc=self._progress_bar_desc or "Dask Applymap"):
                    return self._dask_applymap(self._obj, func, meta)
            else:
                return self._dask_applymap(self._obj, func, meta)
        except ERRORS_TO_HANDLE:
            # if dask apply doesn't match pandas apply, fallback to pandas
            return self._pandas_applymap(self._obj, func)

    def applymap(self, func):
        """
        Applymap the function to the DataFrame using swifter
        """

        # If there are no rows return early using Pandas
        if not self._nrows:
            return self._obj.applymap(func)

        # If parallel processing is forced by the user, then skip the logic and apply dask
        if self._force_parallel:
            return self._parallel_applymap(func)

        sample = self._obj.iloc[self._SAMPLE_INDEX]
        # check if input is string
        # or if the user is overriding the string processing default
        allow_dask_processing = True if self._allow_dask_on_strings else ("object" not in sample.dtypes.values)

        try:  # try to vectorize
            with suppress_stdout_stderr_logging():
                tmp_df = func(sample)
                sample_df = sample.applymap(func)
                self._validate_apply(
                    np.array_equal(sample_df, tmp_df) & (sample_df.shape == tmp_df.shape),
                    error_message=("Vectorized function sample does not match pandas apply sample."),
                )
            return func(self._obj)
        except ERRORS_TO_HANDLE:  # if can't vectorize, estimate time to pandas apply
            wrapped = self._wrapped_applymap(func)
            timed = timeit.timeit(wrapped, number=N_REPEATS)
            sample_proc_est = timed / N_REPEATS
            est_apply_duration = sample_proc_est / self._SAMPLE_SIZE * self._nrows

            # if pandas sample apply takes too long
            # and not performing str processing, use dask
            if (est_apply_duration > self._dask_threshold) and allow_dask_processing:
                return self._parallel_applymap(func)
            else:  # use pandas
                return self._pandas_applymap(self._obj, func)

    def groupby(
        self, by=None, axis=0, level=None, as_index=True, sort=True, group_keys=True, observed=False, dropna=True
    ):
        """
        Create a swifter groupby object
        """
        grpby_kwargs = {
            "level": level,
            "as_index": as_index,
            "sort": sort,
            "group_keys": group_keys,
            "observed": observed,
            "dropna": dropna,
        }
        if RAY_INSTALLED:
            return GroupBy(
                self._obj,
                by=[by] if isinstance(by, str) else by,
                axis=axis,
                progress_bar=self._progress_bar,
                progress_bar_desc=self._progress_bar_desc,
                **grpby_kwargs,
            )
        else:
            raise NotImplementedError(
                "Ray is required for groupby apply functionality."
                "Please install `ray` before continuing and then restart your script or kernel."
            )


if RAY_INSTALLED:  # noqa: C901

    class GroupBy(DataFrameAccessor):
        import ray

        def __init__(
            self,
            pandas_obj,
            by,
            axis=0,
            npartitions=DEFAULT_KWARGS["npartitions"],
            dask_threshold=DEFAULT_KWARGS["dask_threshold"],
            progress_bar=DEFAULT_KWARGS["progress_bar"],
            progress_bar_desc="Ray GroupBy Apply",
            **grpby_kwargs,
        ):
            super(GroupBy, self).__init__(
                pandas_obj,
                npartitions=npartitions,
                dask_threshold=dask_threshold,
                progress_bar=progress_bar,
                progress_bar_desc=progress_bar_desc,
            )
            self._obj_pd = pandas_obj
            self._nrows = pandas_obj.shape[0]
            self._by = by
            self._grpby_index = self._obj_pd.index.equals(self._by)
            self._axis = axis
            self._grpby_kwargs = grpby_kwargs
            self._subset_columns = None

        def __getitem__(self, key):
            self._subset_columns = key
            return self

        # NOTE: All credit for the Ray Groupby Apply logic goes to github user @diditforlulz273
        # NOTE: He provided a gist which I adapted to work in swifter's codebase
        # NOTE: https://gist.github.com/diditforlulz273/06ffa5f5b1c00830671ce0330851352f
        def _get_chunks(self):
            subset_df = self._obj_pd.index if self._grpby_index else self._obj_pd[self._by[0]]
            unique_groups = subset_df.unique()
            n_splits = min(len(unique_groups), self._npartitions)
            splits = np.array_split(unique_groups, n_splits)
            return [self._obj_pd.loc[subset_df.isin(splits[x])] for x in range(n_splits)]

        @ray.remote
        def _ray_groupby_apply_chunk(self, chunk, func, *args, **kwds):
            by = chunk.index if self._grpby_index else self._by
            grpby = chunk.groupby(by, axis=self._axis, **self._grpby_kwargs)
            grpby = grpby if self._subset_columns is None else grpby[self._subset_columns]
            return grpby.apply(func, *args, **kwds)

        def _ray_submit_apply(self, chunks, func, *args, **kwds):
            import ray

            return [self._ray_groupby_apply_chunk.remote(self, ray.put(chunk), func, *args, **kwds) for chunk in chunks]

        def _ray_progress_apply(self, ray_submit_apply, total_chunks):
            import ray

            with tqdm(desc=self._progress_bar_desc, total=total_chunks) as pbar:
                apply_chunks = ray_submit_apply()
                for complete_chunk in range(total_chunks):
                    ray.wait(apply_chunks, num_returns=complete_chunk + 1)
                    pbar.update(1)
            return apply_chunks

        def _ray_apply(self, func, *args, **kwds):
            import ray

            chunks = self._get_chunks()
            ray_submit_apply = partial(self._ray_submit_apply, chunks=chunks, func=func, *args, **kwds)
            apply_chunks = (
                self._ray_progress_apply(ray_submit_apply, len(chunks)) if self._progress_bar else ray_submit_apply()
            )
            return pd.concat(ray.get(apply_chunks), axis=self._axis).sort_index()

        def apply(self, func, *args, **kwds):
            """
            Apply the function to the groupby swifter object
            """
            # if the transformed dataframe is empty or very small, return early using Pandas
            if not self._nrows or self._nrows <= GROUPBY_MAX_ROWS_PANDAS_DEFAULT:
                return self._obj_pd.groupby(self._by, axis=self._axis, **self._grpby_kwargs).apply(func, *args, **kwds)

            # Swifter logic can't accurately estimate groupby applies, so always parallelize
            return self._ray_apply(func, *args, **kwds)


class Transformation(_SwifterObject):
    def __init__(
        self,
        pandas_obj,
        npartitions=DEFAULT_KWARGS["npartitions"],
        dask_threshold=DEFAULT_KWARGS["dask_threshold"],
        scheduler=DEFAULT_KWARGS["scheduler"],
        progress_bar=DEFAULT_KWARGS["progress_bar"],
        progress_bar_desc=DEFAULT_KWARGS["progress_bar_desc"],
        allow_dask_on_strings=DEFAULT_KWARGS["allow_dask_on_strings"],
        force_parallel=DEFAULT_KWARGS["force_parallel"],
    ):
        super(Transformation, self).__init__(
            pandas_obj,
            npartitions,
            dask_threshold,
            scheduler,
            progress_bar,
            progress_bar_desc,
            allow_dask_on_strings,
            force_parallel,
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
    def _parallel_apply(self, func, *args, **kwds):
        raise NotImplementedError("Transformation class does not implement _parallel_apply")

    def apply(self, func, *args, **kwds):
        """
        Apply the function to the transformed swifter object
        """
        # if the transformed dataframe is empty, return early using Pandas
        if not self._nrows:
            return self._obj_pd.apply(func, *args, **kwds)

        # If parallel processing is forced by the user, then skip the logic and apply dask
        if self._force_parallel:
            return self._parallel_apply(func, *args, **kwds)

        # estimate time to pandas apply
        wrapped = self._wrapped_apply(func, *args, **kwds)
        timed = timeit.timeit(wrapped, number=N_REPEATS)
        sample_proc_est = timed / N_REPEATS
        est_apply_duration = sample_proc_est / self._SAMPLE_SIZE * self._nrows

        # No `allow_dask_processing` variable here,
        # because we don't know the dtypes of the transformation
        if est_apply_duration > self._dask_threshold:
            return self._parallel_apply(func, *args, **kwds)
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
        force_parallel=False,
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
            force_parallel=force_parallel,
        )
        self._rolling_kwds = kwds.copy()
        self._comparison_pd = self._obj_pd.iloc[: self._SAMPLE_SIZE]
        self._sample_pd = self._sample_pd.rolling(**kwds)
        self._obj_pd = self._obj_pd.rolling(**kwds)
        self._obj_dd = self._obj_dd.rolling(**{k: v for k, v in kwds.items() if k not in ["on", "closed"]})

    def _parallel_apply(self, func, *args, **kwds):
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
                    error_message=("Dask rolling apply sample does not match " "pandas rolling apply sample."),
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
        force_parallel=False,
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
            force_parallel=force_parallel,
        )
        self._resampler_kwds = kwds.copy()
        self._comparison_pd = self._obj_pd.iloc[: self._SAMPLE_SIZE]
        self._sample_pd = self._sample_pd.resample(**kwds)
        self._obj_pd = self._obj_pd.resample(**kwds)
        # Setting dask dataframe `self._obj_dd` to None when there are 0 `self._nrows`
        # because swifter will immediately return the pandas form during the apply
        # function if there are 0 `self._nrows`
        self._obj_dd = (
            self._obj_dd.resample(**{k: v for k, v in kwds.items() if k in ["rule", "closed", "label"]})
            if self._nrows
            else None
        )

    def _parallel_apply(self, func, *args, **kwds):
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
                    error_message=("Dask resampler apply sample does not match " "pandas resampler apply sample."),
                )

            if self._progress_bar:
                with TQDMDaskProgressBar(desc=self._progress_bar_desc or "Dask Apply"):
                    return self._obj_dd.agg(func, *args, **kwds).compute(scheduler=self._scheduler)
            else:
                return self._obj_dd.agg(func, *args, **kwds).compute(scheduler=self._scheduler)
        except ERRORS_TO_HANDLE:
            # use pandas -- no progress_apply available for resampler objects
            return self._obj_pd.apply(func, *args, **kwds)
