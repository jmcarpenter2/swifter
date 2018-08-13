# Changelog

## Version 0.200
Completely refactored the package as an extension to pandas, rather than an independent function call. This will allow for increased flexibility of the user and simplicity of using swiftapply.

## Version  0.150
Fixed bug that would allow certain functions to be applied to the entire series/dataframe, rather than to each element. For example, len(x) returned the length of the series, rather than the length of each string within the series. A special thanks to @bharatvem for pointing this out.

## Version 0.140
Added support for vectorized and pandas applies to dataframes.
Converted all string manipulations to pandas apply (unless vectorizable) because dask processes string manipulations slowly.ls

## Version 0.13
Removed numba jit function, because this was adding to the total runtime. Will do some experiments and consider readding later.

## Version 0.1
Currently works very well with pandas series, needs some work to optimize dask multiprocessing for pandas dataframes. For now, it is probably best to apply to each series independently, rather than multiple columns of a dataframe at once.
