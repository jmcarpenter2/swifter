import pandas as pd
from psutil import cpu_count
from dask import dataframe as dd
import timeit
import warnings


@pd.api.extensions.register_series_accessor("swifter")
class SeriesAccessor():
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
                map_partitions(func, *args, meta=meta, **kwds).compute(scheduler='multiprocessing')
            assert tmp_df.shape == samp.shape
            return dd.from_pandas(self._obj, npartitions=self._obj.npartitions). \
                map_partitions(func, *args, meta=meta, **kwds).compute(scheduler='multiprocessing')
        except (AssertionError, AttributeError, ValueError) as e:
            return dd.from_pandas(self._obj, npartitions=self._obj.npartitions). \
                map(lambda x: func(x, *args, **kwds), meta=meta).compute(scheduler='multiprocessing')

    def apply(self, func, convert_dtype=True, args=(), **kwds):
        samp = self._obj.iloc[:1000]

        try:  # try to vectorize
            tmp_df = func(samp, *args, **kwds)
            assert tmp_df.shape == samp.shape
            return func(self._obj, *args, **kwds)
        except (AssertionError, AttributeError, ValueError) as e:  # if can't vectorize, estimate time to pandas apply
            wrapped = self._wrapped_apply(func, convert_dtype=convert_dtype, args=args, **kwds)
            n_repeats = 3
            timed = timeit.timeit(wrapped, number=n_repeats)
            samp_proc_est = timed / n_repeats
            est_apply_duration = samp_proc_est / 1000 * self._obj.shape[0]

            # Get meta information for dask, and check if output is str
            kwds['meta'] = self._obj.iloc[:2].apply(func, convert_dtype=convert_dtype, args=args, **kwds)
            str_object = object == kwds['meta'].dtypes

            # if pandas apply takes too long and output is not str, use dask
            if (est_apply_duration > self._obj.dask_threshold) and (not str_object):
                return self._dask_apply(func, *args, **kwds)
            else:  # use pandas
                kwds.pop('meta')
                return self._obj.apply(func, convert_dtype=convert_dtype, args=args, **kwds)


@pd.api.extensions.register_dataframe_accessor("swifter")
class DataFrameAccessor():
    def __init__(self, pandas_dataframe, npartitions=None, dask_threshold=1):
        self._obj = pandas_dataframe

        if npartitions is None:
            self._obj.npartitions = cpu_count() * 2
        else:
            self._obj.npartitions = npartitions
        self._obj.dask_threshold = dask_threshold

    def _wrapped_apply(self, func, axis=1, broadcast=None, raw=False, reduce=None, result_type=None, args=(), **kwds):
        def wrapped():
            pd.concat([self._obj.loc[:1000, c].apply(func, convert_dtype=True, args=args, **kwds)
                       for c in self._obj.columns], axis=axis)
        return wrapped

    def _dask_apply(self, func, axis=1, *args, **kwds):
        samp = self._obj.iloc[:self._obj.npartitions*2]
        tmp = kwds.pop('meta')
        meta = {c: tmp[c].dtype for c in tmp.columns}
        try:
            tmp_df = dd.from_pandas(samp, npartitions=self._obj.npartitions).\
                apply(func, *args, axis=axis, meta=meta, **kwds).compute(scheduler='multiprocessing')
            assert tmp_df.shape == samp.shape
            return dd.from_pandas(self._obj, npartitions=self._obj.npartitions).\
                apply(func, *args, axis=axis, meta=meta, **kwds).compute(scheduler='multiprocessing')
        except (AssertionError, AttributeError, ValueError) as e:
            warnings.warn('Dask applymap not working correctly. Concatenating swiftapplies instead.')
            return pd.concat([self._obj[c].swifter.apply(func, *args, **kwds)
                              for c in self._obj.columns], axis=1)

    def apply(self, func, axis=1, broadcast=None, raw=False, reduce=None, result_type=None, args=(), **kwds):
        samp = self._obj.iloc[:1000]

        try:  # try to vectorize
            tmp_df = pd.concat([pd.Series(func(samp[c], *args, **kwds), name=c)
                                for c in samp.columns], axis=axis)
            assert tmp_df.shape == samp.shape
            return pd.concat([pd.Series(func(self._obj[c], *args, **kwds), name=c)
                              for c in self._obj.columns], axis=1)
        except (AssertionError, AttributeError, ValueError) as e:  # if can't vectorize, estimate time to pandas apply
            wrapped = self._wrapped_apply(func, axis=axis, broadcast=broadcast, raw=raw, reduce=reduce,
                                          result_type=result_type, args=args, **kwds)
            n_repeats = 3
            timed = timeit.timeit(wrapped, number=n_repeats)
            samp_proc_est = timed / n_repeats
            est_apply_duration = samp_proc_est / 1000 * self._obj.shape[0]

            # Get meta information for dask, and check if output is str
            kwds['meta'] = pd.concat([self._obj.loc[:1000, c].apply(func, convert_dtype=True, args=args, **kwds)
                                      for c in self._obj.columns], axis=1)
            str_object = object in kwds['meta'].dtypes.values

            # if pandas apply takes too long and output is not str, use dask
            if (est_apply_duration > self._obj.dask_threshold) and (not str_object):
                return self._dask_apply(func, axis=axis, *args, **kwds)
            else:  # use pandas
                kwds.pop('meta')
                return pd.concat([self._obj.loc[:1000, c].apply(func, convert_dtype=True, args=args, **kwds)
                                  for c in self._obj.columns], axis=1)
