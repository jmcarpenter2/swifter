# Documentation

## 1. `swift_apply`

```python
def swift_apply(df, myfunc, *args, **kwargs, npartitions=cpu_count()*2, dask_threshold=1)
```

Efficiently apply any function to a pandas dataframe or series in the fastest available manner

**Parameters:**

`df`: The dataframe or series to apply the function to

`myfunc`: The function you wish to apply

`args`: The positional arguments of the function

`kwargs`: The key word arguments of the function


You can also specify npartitions and dask_threshold


`npartitions`: The number of partitions to use for multiprocessing (affects speed of dask)

`dask_threshold`: The maximum allowed time (in seconds) for a normal pandas apply before switching to a dask operation


**returns:**

The new dataframe/series with the function applied as quickly as possible


