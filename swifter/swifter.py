import timeit
import warnings
import numpy as np
import pandas as pd

from abc import abstractmethod
from dask import dataframe as dd

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

    def _dask_apply(self, func, convert_dtype, *args, **kwds):
        sample = self._obj.iloc[self._SAMPLE_INDEX]
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
                    tmp_df.equals(meta),
                    error_message=("Dask map-partitions sample does not match pandas apply sample."),
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
            # if map partitions doesn't match pandas apply,
            # we can use dask apply, but it will be a bit slower
            if self._progress_bar:
                with TQDMDaskProgressBar(desc=self._progress_bar_desc or "Dask Apply"):
                    return (
                        dd.from_pandas(self._obj, npartitions=self._npartitions)
                        .apply(
                            lambda x: func(x, *args, **kwds),
                            convert_dtype=convert_dtype,
                            meta=meta,
                        )
                        .compute(scheduler=self._scheduler)
                    )
            else:
                return (
                    dd.from_pandas(self._obj, npartitions=self._npartitions)
                    .apply(
                        lambda x: func(x, *args, **kwds),
                        convert_dtype=convert_dtype,
                        meta=meta,
                    )
                    .compute(scheduler=self._scheduler)
                )

    def apply(self, func, convert_dtype=True, args=(), **kwds):
        """
        Apply the function to the Series using swifter
        """

        # if the series is empty, return early using Pandas
        if not self._nrows:
            return self._obj.apply(func, convert_dtype=convert_dtype, args=args, **kwds)

        # If parallel processing is forced by the user, then skip the logic and apply dask
        if self._force_parallel:
            return self._dask_apply(func, convert_dtype, *args, **kwds)

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
                self._obj.iloc[self._SAMPLE_INDEX].apply(
                    func, axis=axis, raw=raw, result_type=result_type, args=args, **kwds
                )

        return wrapped

    def _dask_apply(self, func, axis=0, raw=None, result_type=None, *args, **kwds):
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
                    return (
                        dd.from_pandas(self._obj, npartitions=self._npartitions)
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
            else:
                return (
                    dd.from_pandas(self._obj, npartitions=self._npartitions)
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

        # If parallel processing is forced by the user, then skip the logic and apply dask
        if self._force_parallel:
            return self._dask_apply(func, axis, raw, result_type, *args, **kwds)

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
                return self._dask_apply(func, axis, raw, result_type, *args, **kwds)
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
                self._obj.iloc[self._SAMPLE_INDEX].applymap(func)

        return wrapped

    def _dask_applymap(self, func):
        sample = self._obj.iloc[self._SAMPLE_INDEX]
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
                    tmp_df.equals(meta),
                    error_message=("Dask applymap sample does not match pandas applymap sample."),
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

        # If parallel processing is forced by the user, then skip the logic and apply dask
        if self._force_parallel:
            return self._dask_applymap(func)

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
                return self._dask_applymap(func)
            else:  # use pandas
                if self._progress_bar:
                    tqdm.pandas(desc=self._progress_bar_desc or "Pandas Apply")
                    applymap_func = self._obj.progress_applymap
                else:
                    applymap_func = self._obj.applymap

                return applymap_func(func)

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
            self._sample_pd = pandas_obj.iloc[self._SAMPLE_INDEX]
            self._obj_pd = pandas_obj
            self._nrows = pandas_obj.shape[0]
            self._by = by
            self._axis = axis
            self._grpby_kwargs = grpby_kwargs

        def _wrapped_apply(self, func, *args, **kwds):
            def wrapped():
                with suppress_stdout_stderr_logging():
                    self._sample_pd.groupby(by=self._by, axis=self._axis, **self._grpby_kwargs).apply(
                        func, *args, **kwds
                    )

            return wrapped

        # NOTE: All credit for the _ray_apply/_ray_apply_chunk logic goes to github user @diditforlulz273
        # NOTE: He provided a gist which I adapted to work in swifter's codebase
        # NOTE: https://gist.github.com/diditforlulz273/06ffa5f5b1c00830671ce0330851352f
        @ray.remote
        def _ray_apply_chunk(self, chunk, func, *args, **kwds):
            # Ray makes data immutable when stored in its memory.
            # This approach prevents state sharing among processes, but we have a separate chunk for each process
            # to get rid of copying data, we make it mutable in-place again by this hack
            for d in range(len(chunk._data.blocks)):
                try:
                    chunk._data.blocks[d].values.flags.writeable = True
                except Exception:
                    pass

            return chunk.groupby(self._by, axis=self._axis, **self._grpby_kwargs).apply(func, *args, **kwds)

        def _ray_apply(self, func, *args, **kwds):
            import ray

            unique_groups = self._obj_pd[self._by[0]].unique()
            n_splits = min(len(unique_groups), self._npartitions)
            splits = np.array_split(unique_groups, n_splits)
            chunks = [self._obj_pd.loc[self._obj_pd[self._by[0]].isin(splits[x])] for x in range(n_splits)]

            # Fire and forget
            chunk_id = [ray.put(ch) for ch in chunks]
            ray_obj_refs = [
                self._ray_apply_chunk.remote(self, chunk_id[i], func, *args, **kwds) for i in range(n_splits)
            ]

            # TQDM progress bar
            if self._progress_bar:
                for chunk in tqdm(range(n_splits), desc=self._progress_bar_desc):
                    ray.wait(ray_obj_refs, num_returns=chunk + 1)

            # Collect, sort, and return
            return pd.concat(ray.get(ray_obj_refs), axis=self._axis).sort_index()

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
    def _dask_apply(self, func, *args, **kwds):
        raise NotImplementedError("Transformation class does not implement _dask_apply")

    def apply(self, func, *args, **kwds):
        """
        Apply the function to the transformed swifter object
        """
        # if the transformed dataframe is empty, return early using Pandas
        if not self._nrows:
            return self._obj_pd.apply(func, *args, **kwds)

        # If parallel processing is forced by the user, then skip the logic and apply dask
        if self._force_parallel:
            return self._dask_apply(func, *args, **kwds)

        # estimate time to pandas apply
        wrapped = self._wrapped_apply(func, *args, **kwds)
        timed = timeit.timeit(wrapped, number=N_REPEATS)
        sample_proc_est = timed / N_REPEATS
        est_apply_duration = sample_proc_est / self._SAMPLE_SIZE * self._nrows

        # No `allow_dask_processing` variable here,
        # because we don't know the dtypes of the transformation
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
