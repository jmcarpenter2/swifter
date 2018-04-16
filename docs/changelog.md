# Changelog

## Version 0.140
Added support for vectorized and pandas applies to dataframes.
Converted all string manipulations to pandas apply (unless vectorizable) because dask processes string manipulations slowly.

## Version 0.13
Removed numba jit function, because this was adding to the total runtime. Will do some experiments and consider readding later.

## Version 0.1
Currently works very well with pandas series, needs some work to optimize dask multiprocessing for pandas dataframes. For now, it is probably best to apply to each series independently, rather than multiple columns of a dataframe at once.
