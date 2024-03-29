# Documentation

## Important Notes

1. Please upgrade your version of pandas, as the pandas extension api used in this module is a recent addition to pandas.

2. Do not use swifter to apply a function that modifies external variables. Under the hood, swifter does sample applies to optimize performance. These sample applies will modify the external variable in addition to the final apply. Thus, you will end up with an erroneously modified external variable.

3. It is advised to disable the progress bar if calling swifter from a forked process as the progress bar may get confused between various multiprocessing modules. 

## 1. `swifter.set_defaults`

Allows for upfront configuration of swifter settings. Set once, re-use across all dataframe swifter invocations.
**NOTE: You must set the defaults before creating the dataframe because this entrypoint is part of the `__init__`.**

```python
from swifter import set_defaults
set_defaults(
    npartitions=None,
    dask_threshold=1,
    scheduler="processes",
    progress_bar=True,
    progress_bar_desc=None,
    allow_dask_on_strings=False,
    force_parallel=False,
)
```

**Parameters:**

`npartitions` : Integer. The number of partitions to distribute the data into for dask processing.
    Default: `2*cpu_count()`

`dask_threshold` : Float. The amount of seconds to use for estimating whether to use dask or pandas apply.
    Default: `1` second

`scheduler` : String. Whether to use `threads` or `processes` for the dask scheduler
    Default: `processes`

`progress_bar` : Boolean. Whether to turn the progress bar on or off.
    Default: `True`

`progress_bar_desc` : String. Progress Bar Description
    Default: `None`

`allow_dask_on_strings` : Boolean. Allows user to enable dask parallel processing on string data
    Default: `False`

`force_parallel` : Boolean. Allows user to override swifter algorithm and jump straight to using dask processing
    Default: `False`



## 2. `pandas.Series.swifter.apply` OR `modin.pandas.Series.swifter.apply`

Efficiently apply any function to a pandas series in the fastest available manner

```python
def pandas.Series.swifter.apply(func, convert_dtype=True, args=(), **kwds)
```

**Parameters:**

`func` : function. Function to apply to each element of the series.

`convert_dtype` : boolean, default True. Try to find better dtype for elementwise function results. If False, leave as dtype=object

`args` : tuple. Positional arguments to pass to function in addition to the value

`kwds` : Additional keyword arguments will be passed as keywords to the function

NOTE: docstring taken from pandas documentation.


## 3. `pandas.DataFrame.swifter.apply` OR `modin.pandas.DataFrame.swifter.apply`

Efficiently apply any function to a pandas dataframe in the fastest available manner.

```python
def pandas.DataFrame.swifter.apply(
        func, 
        axis=0, 
        raw=False, 
        result_type=None,
        args=(), 
        **kwds
    )
```

**Parameters:**

`func` : function. Function to apply to each column or row.

`axis` : {0 or 'index', 1 or 'columns'}, default 0. **For now, Dask only supports axis=1, and thus swifter is limited to axis=1 on large datasets when the function cannot be vectorized.** Axis along which the function is applied:

* 0 or 'index': apply function to each column.
* 1 or 'columns': apply function to each row.

`raw` : bool, default False
False : passes each row or column as a Series to the function.
True : the passed function will receive ndarray objects instead. If you are just applying a NumPy reduction function this will achieve much better performance.

`result_type` : {'expand', 'reduce', 'broadcast', None}, default None. These only act when axis=1 (columns):

'expand' : list-like results will be turned into columns.
'reduce' : returns a Series if possible rather than expanding list-like results. This is the opposite of 'expand'.
'broadcast' : results will be broadcast to the original shape of the DataFrame, the original index and columns will be retained.
The default behaviour (None) depends on the return value of the applied function: list-like results will be returned as a Series of those. However if the apply function returns a Series these are expanded to columns.

`args` : tuple. Positional arguments to pass to func in addition to the array/series.

`kwds` : Additional keyword arguments to pass as keywords arguments to func.

NOTE: docstring taken from pandas documentation.

**returns:**

The new dataframe/series with the function applied as quickly as possible

## 4. `pandas.DataFrame.swifter.applymap`

Efficiently applymap any function to a pandas dataframe in the fastest available manner. Applymap is elementwise.

```python
def pandas.DataFrame.swifter.applymap(func)
```

## 5. `pandas.DataFrame.swifter.groupby.apply`

Applies over a resampler object on the original series/dataframe in the fastest available manner.

```python
def pandas.DataFrame.swifter.groupby(
        by,
        axis=0,
        level=None,
        as_index=True,
        sort=True,
        group_keys=True,
        squeeze=False,
        observed=False,
        dropna=True
    ).apply(func, *args, **kwds)
```

## 6. `pandas.DataFrame.swifter.rolling.apply`

Applies over a rolling object on the original series/dataframe in the fastest available manner.

```python
def pandas.DataFrame.swifter.rolling(
        window, 
        min_periods=None, 
        center=False, 
        win_type=None, 
        on=None, 
        axis=0, 
        closed=None
    ).apply(func, *args, **kwds)
```

## 7. `pandas.DataFrame.swifter.resample.apply`

Applies over a resampler object on the original series/dataframe in the fastest available manner.

```python
def pandas.DataFrame.swifter.resample(
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
    ).apply(func, *args, **kwds)
```

## 8. `pandas.DataFrame.swifter.progress_bar(False).apply`

Enable or disable the TQDM progress bar by setting the enable parameter to True/False, respectively. You can also specify a custom description.

Note: It is advised to disable the progress bar if calling swifter from a forked process as the progress bar may get confused between various multiprocessing modules. 

```python
def pandas.DataFrame.swifter.progress_bar(enable=True, desc=None)
```

For example, let's say we have a pandas dataframe df. The following will perform a swifter apply, without the TQDM progress bar.

```python
df.swifter.progress_bar(False).apply(lambda x: x+1)
```

## 9. `pandas.DataFrame.swifter.set_npartitions(npartitions=None).apply`

Specify the number of partitions to allocate to swifter, if parallel processing is chosen to be the quickest apply.
If npartitions=None, it defaults to cpu_count()*2

```python
def pandas.DataFrame.swifter.set_npartitions(npartitions=None)
```

For example, let's say we have a pandas dataframe df. The following will perform a swifter apply, using 2 partitions
```python
df.swifter.set_npartitions(2).apply(lambda x: x+1)
```

## 10. `pandas.DataFrame.swifter.set_dask_threshold(dask_threshold=1).apply`

Specify the dask threshold (in seconds) for the max allowable time estimate for a pandas apply on the full dataframe
```python
def pandas.DataFrame.swifter.set_dask_threshold(dask_threshold=1)
```

For example, let's say we have a pandas dataframe df. The following will perform a swifter apply, with the threshold set to 3 seconds
```python
df.swifter.set_dask_threshold(dask_threshold=3).apply(lambda x: x+1)
```

## 11. `pandas.DataFrame.swifter.set_dask_scheduler(scheduler="processes").apply`

Set the dask scheduler

:param scheduler: String, ["threads", "processes"]
```python
def pandas.DataFrame.swifter.set_dask_scheduler(scheduler="processes")
```

For example, let's say we have a pandas dataframe df. The following will perform a swifter apply, with the scheduler set to multithreading.
```python
df.swifter.set_dask_scheduler(scheduler="threads").apply(lambda x: x+1)
```

## 12. `pandas.DataFrame.swifter.allow_dask_on_strings(enable=True).apply`

This flag allows the user to specify whether to allow dask to handle dataframes containing string types. Dask can be particularly slow if you are actually manipulating strings, but if you just have a string column in your data frame this will allow dask to handle the execution.
```python
def pandas.DataFrame.swifter.allow_dask_on_strings(enable=True)
```

For example, let's say we have a pandas dataframe df. The following will allow Dask to process a dataframe with string columns.
```python
df.swifter.allow_dask_on_strings().apply(lambda x: x+1)
```

## 13. `pandas.DataFrame.swifter.force_parallel(enable=True).apply`

This flag allows the user to specify to override swifter's default functionality to run try vectorization, sample applies, and determine the fastest apply possible. Instead it forces swifter to use dask.
```python
def pandas.DataFrame.swifter.force_parallel(enable=True)
```

For example, let's say we have a pandas dataframe df. The following will force Dask to process the dataframe.
```python
df.swifter.force_parallel().apply(lambda x: x+1)
```

## 14. `swifter.register_modin()`

This gives access to `modin.DataFrame.swifter.apply(...)` and `modin.Series.swifter.apply(...)`. This registers modin dataframes and series with swifter as accessors.
* NOTE: This is only necessary if you import swifter BEFORE modin. If you import modin before swifter you do not need to execute this method.
