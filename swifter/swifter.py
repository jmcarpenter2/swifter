import pandas as pd
from psutil import cpu_count
from dask import dataframe as dd
from dask.multiprocessing import get
import timeit
import warnings


def pd_apply(df, myfunc, *args, **kwargs):
    def wrapped():
        if type(df) == pd.DataFrame:
            pd.concat([df[c].apply(myfunc, args=args, **kwargs)
                       for c in df.columns], axis=1)
        else:
            df.apply(myfunc, args=args, **kwargs)
    return wrapped


def dask_apply(df, npartitions, myfunc, *args, **kwargs):
    samp = df.iloc[:npartitions*2]

    if type(df) == pd.DataFrame:
        tmp = kwargs.pop('meta')
        meta = {c: tmp[c].dtype for c in tmp.columns}
        try:
            tmp_df = dd.from_pandas(samp, npartitions=npartitions).\
                apply(myfunc, *args, axis=1, meta=meta, **kwargs).compute(get=get)
            assert tmp_df.shape == samp.shape
            return dd.from_pandas(df, npartitions=npartitions).\
                apply(myfunc, *args, axis=1, meta=meta, **kwargs).compute(get=get)
        except (AssertionError, AttributeError, ValueError) as e:
            warnings.warn('Dask applymap not working correctly. Concatenating swiftapplies instead.')
            return pd.concat([swiftapply(df[c], myfunc, *args, **kwargs)
                              for c in df.columns], axis=1)
    else:
        meta = kwargs.pop('meta')
        try:
            tmp_df = dd.from_pandas(samp, npartitions=npartitions).\
                map_partitions(myfunc, *args, meta=meta, **kwargs).compute(get=get)
            assert tmp_df.shape == samp.shape
            return dd.from_pandas(df, npartitions=npartitions).\
                map_partitions(myfunc, *args, meta=meta, **kwargs).compute(get=get)
        except (AssertionError, AttributeError, ValueError) as e:
            return dd.from_pandas(df, npartitions=npartitions).\
                map(lambda x: myfunc(x, *args, **kwargs), meta=meta).compute(get=get)


def swiftapply(df, myfunc, *args, **kwargs):
    """
    Efficiently apply any function to a pandas dataframe or series
    in the fastest available manner
    :param df: The dataframe or series to apply the function to
    :param myfunc: The function you wish to apply
    :param args: The positional arguments of the function
    :param kwargs: The key word arguments of the function
        You can also specify npartitions and dask_threshold
        npartitions will affect the speed of dask multiprocessing
        dask_threshold is the maximum allowed time (in seconds) for a normal pandas apply
            before switching to a dask operation
    :return: The new dataframe/series with the function applied as quickly as possible
    """
    if 'npartitions' in kwargs.keys():
        npartitions = kwargs.pop('npartitions')
    else:
        npartitions = cpu_count() * 2
    if 'dask_threshold' in kwargs.keys():
        dask_threshold = kwargs.pop('dask_threshold')
    else:
        dask_threshold = 1

    if myfunc is not str:
        samp = df.iloc[:1000]

        try:  # try to vectorize
            if type(df) == pd.DataFrame:
                tmp_df = pd.concat([pd.Series(myfunc(samp[c], *args, **kwargs), name=c)
                                    for c in samp.columns], axis=1)
                assert tmp_df.shape == samp.shape
                return pd.concat([pd.Series(myfunc(df[c], *args, **kwargs), name=c)
                                  for c in df.columns], axis=1)
            else:
                tmp_df = myfunc(samp, *args, **kwargs)
                assert tmp_df.shape == samp.shape
                return myfunc(df, *args, **kwargs)

        except (AssertionError, AttributeError, ValueError) as e:  # if can't vectorize, estimate time to pandas apply
            wrapped = pd_apply(samp, myfunc, *args, **kwargs)
            n_repeats = 3
            timed = timeit.timeit(wrapped, number=n_repeats)
            samp_proc_est = timed/n_repeats
            est_apply_duration = samp_proc_est / len(samp) * df.shape[0]

            # Get meta information for dask, and check if output is str
            if type(df) == pd.DataFrame:
                kwargs['meta'] = pd.concat([df.loc[:2, c].apply(myfunc, args=args, **kwargs)
                                            for c in df.columns], axis=1)
                str_object = object in kwargs['meta'].dtypes.values
            else:
                kwargs['meta'] = df.iloc[:2].apply(myfunc, args=args, **kwargs)
                str_object = object == kwargs['meta'].dtypes

            # if pandas apply takes too long and output is not str, use dask
            if (est_apply_duration > dask_threshold) and (not str_object):
                return dask_apply(df, npartitions, myfunc, *args, **kwargs)
            else:  # use pandas
                kwargs.pop('meta')
                if type(df) == pd.DataFrame:
                    return pd.concat([df[c].apply(myfunc, args=args, **kwargs)
                                      for c in df.columns], axis=1)
                else:
                    return df.apply(myfunc, args=args, **kwargs)
    else:
        return df.astype(str)
