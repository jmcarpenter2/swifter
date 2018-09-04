import pandas as pd
from psutil import cpu_count
from dask import dataframe as dd
import timeit
import warnings


@pd.api.extensions.register_series_accessor("swifter")
class SeriesAccessor:
    def __init__(self, pandas_series, npartitions=None, dask_threshold=1):
        self._obj = pandas_series

        if npartitions is None:
            self._obj.npartitions = cpu_count() * 2
        else:
            self._obj.npartitions = npartitions
        self._obj.dask_threshold = dask_threshold

    def _wrapped_apply(self, func, convert_dtype=True, args=(), **kwds):
        def wrapped():
                self._obj.iloc[:1000].apply(func, convert_dtype=convert_dtype, args=args, **kwds)
        return wrapped

    def _dask_apply(self, func, axis=0, *args, **kwds):
        samp = self._obj.iloc[:self._obj.npartitions*2]
        meta = kwds.pop('meta')
        try:
            tmp_df = dd.from_pandas(samp, npartitions=self._obj.npartitions). \
                map_partitions(func, *args, meta=meta, **kwds).compute(scheduler='processes')
            assert tmp_df.shape == meta.shape
            return dd.from_pandas(self._obj, npartitions=self._obj.npartitions). \
                map_partitions(func, *args, meta=meta, **kwds).compute(scheduler='processes')
        except (AssertionError, AttributeError, ValueError, TypeError) as e:
            return dd.from_pandas(self._obj, npartitions=self._obj.npartitions). \
                map(lambda x: func(x, *args, **kwds), meta=meta).compute(scheduler='processes')

    def apply(self, func, convert_dtype=True, args=(), **kwds):
        samp = self._obj.iloc[:self._obj.npartitions*2]
        str_object = samp.dtype == 'object' # check if input is string

        if 'axis' in kwds.keys():
            kwds.pop('axis')
            warnings.warn('Axis keyword not necessary because applying on a Series.')

        try:  # try to vectorize
            tmp_df = func(samp, *args, **kwds)
            assert tmp_df.shape == samp.shape
            return func(self._obj, *args, **kwds)
        except (AssertionError, AttributeError, ValueError, TypeError) as e:  # if can't vectorize, estimate time to pandas apply
            wrapped = self._wrapped_apply(func, convert_dtype=convert_dtype, args=args, **kwds)
            n_repeats = 3
            timed = timeit.timeit(wrapped, number=n_repeats)
            samp_proc_est = timed / n_repeats
            est_apply_duration = samp_proc_est / 1000 * self._obj.shape[0]

            # Get meta information for dask
            kwds['meta'] = self._obj.iloc[:2].apply(func, convert_dtype=convert_dtype, args=args, **kwds)

            # if pandas apply takes too long and input is not str, use dask
            if (est_apply_duration > self._obj.dask_threshold) and (not str_object):
                return self._dask_apply(func, *args, **kwds)
            else:  # use pandas
                kwds.pop('meta')
                return self._obj.apply(func, convert_dtype=convert_dtype, args=args, **kwds)


@pd.api.extensions.register_dataframe_accessor("swifter")
class DataFrameAccessor:
    def __init__(self, pandas_dataframe, npartitions=None, dask_threshold=1):
        self._obj = pandas_dataframe

        if npartitions is None:
            self._obj.npartitions = cpu_count() * 2
        else:
            self._obj.npartitions = npartitions
        self._obj.dask_threshold = dask_threshold

    def _wrapped_apply(self, func, axis=0, broadcast=None, raw=False, reduce=None, result_type=None, args=(), **kwds):
        def wrapped():
            self._obj.iloc[:1000, :].apply(func, axis=axis, broadcast=broadcast, raw=raw, reduce=reduce,
                                       result_type=result_type, args=args, **kwds)
        return wrapped

    def _dask_apply(self, func, axis=0, *args, **kwds):
        samp = self._obj.iloc[:self._obj.npartitions*2, :]
        tmp = kwds.pop('meta')
        if type(tmp) == pd.Series:
            meta = tmp
        else:
            meta = {c: tmp[c].dtype for c in tmp.columns}
        try:
            tmp_df = dd.from_pandas(samp, npartitions=self._obj.npartitions).\
                apply(func, *args, axis=axis, meta=meta, **kwds).compute(scheduler='processes')
            assert tmp_df.shape == meta.shape
            return dd.from_pandas(self._obj, npartitions=self._obj.npartitions).\
                apply(func, *args, axis=axis, meta=meta, **kwds).compute(scheduler='processes')
        except (AssertionError, AttributeError, ValueError, TypeError) as e:
            warnings.warn('Dask applymap not working correctly. Concatenating swiftapplies instead.')
            return pd.concat([self._obj[c].swifter.apply(func, *args, **kwds)
                              for c in self._obj.columns], axis=1)

    def apply(self, func, axis=1, broadcast=None, raw=False, reduce=None, result_type=None, args=(), **kwds):
        samp = self._obj.iloc[:self._obj.npartitions*2, :]
        str_object = 'object' in samp.dtypes.values # check if input is string

        try:  # try to vectorize
            if 'axis' in kwds.keys():
                kwds.pop('axis')
            tmp_df = func(samp, *args, **kwds)
            assert tmp_df.shape == samp.shape
            return func(self._obj, *args, **kwds)
        except (AssertionError, AttributeError, ValueError, TypeError) as e:  # if can't vectorize, estimate time to pandas apply
            wrapped = self._wrapped_apply(func, axis=axis, broadcast=broadcast, raw=raw, reduce=reduce,
                                       result_type=result_type, args=args, **kwds)
            n_repeats = 3
            timed = timeit.timeit(wrapped, number=n_repeats)
            samp_proc_est = timed / n_repeats
            est_apply_duration = samp_proc_est / 1000 * self._obj.shape[0]

            # Get meta information for dask
            kwds['meta'] = self._obj.iloc[:self._obj.npartitions*2, :].apply(func, axis=axis, broadcast=broadcast, raw=raw, reduce=reduce,
                                       result_type=result_type, args=args, **kwds)

            # if pandas apply takes too long and input is not str, use dask
            if (est_apply_duration > self._obj.dask_threshold) and (not str_object):
                return self._dask_apply(func, axis=axis, *args, **kwds)
            else:  # use pandas
                kwds.pop('meta')
                return self._obj.apply(func, axis=axis, broadcast=broadcast, raw=raw, reduce=reduce,
                                       result_type=result_type, args=args, **kwds)

    def _wrapped_groupby_apply(self, groupby_col, func, *args, **kwds):
        def wrapped():
            self._obj.iloc[:1000, :].groupby(groupby_col).apply(func, *args, **kwds)
        return wrapped

    def _dask_groupby_apply(self, groupby_col, func, *args, **kwds):
        tmp = kwds.pop('meta')
        meta = {c: tmp[c].dtype for c in tmp.columns if c is not groupby_col}
        try:
            return dd.from_pandas(self._obj, npartitions=self._obj.npartitions).set_index(groupby_col).\
                groupby(groupby_col).apply(func, *args, meta=meta, **kwds).compute(scheduler='processes')
        except (AssertionError, AttributeError, ValueError, TypeError) as e:
            print(e)

    def groupby_apply(self, groupby_col, func, *args, **kwds):
        wrapped = self._wrapped_groupby_apply(groupby_col, func, *args, **kwds)
        n_repeats = 3
        timed = timeit.timeit(wrapped, number=n_repeats)
        samp_proc_est = timed / n_repeats
        est_apply_duration = samp_proc_est / 1000 * self._obj.shape[0]

        # Get meta information for dask
        kwds['meta'] = self._obj.iloc[:self._obj.npartitions*2, :].groupby(groupby_col).apply(func, *args, **kwds)

        # if pandas apply takes too long, use dask
        if (est_apply_duration) > self._obj.dask_threshold:
            return self._dask_groupby_apply(groupby_col, func, *args, **kwds)
        else: # use pandas
            kwds.pop('meta')
            return self._obj.groupby(groupby_col).apply(func, *args, **kwds)