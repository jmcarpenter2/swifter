# swifter
A package which efficiently applies any function to a pandas dataframe or series in the fastest available manner

*Current version == 0.224*

Installation:
```
$ pip install -U pandas # upgrade pandas
$ pip install swifter # first time installation

$ pip install -U swifter # upgrade to latest version
``` 

and then import into your code along with pandas using:
```
import pandas as pd
import swifter
```

## Easy to use
```
df = pd.DataFrame({'x': [1, 2, 3, 4], 'y': ['a', 'b', 'a', 'b']})

df['x2'] = df['x'].swifter.apply(lambda x: x**2)
df['outCol'] = df['inCol'].swifter.apply(my_func)
df['outCol'] = df['inCol'].swifter.apply(my_func, positional_arg, keyword_arg=keyword_argval)

df_grouped = df.swifter.groupby_apply('y', lambda x: x.mean() - x.min())
df_grouped = df.swifter.groupby_apply(groupby_col, my_func)
df-grouped = df.swifter.groupby_apply(groupby_col, my_func, positional_arg, keyword_arg=keyword_argvaL)
```

Check out the [examples notebook](examples/swiftapply_examples.ipynb), along with the [speed benchmark notebook](examples/swiftapply_speedcomparison.ipynb)

## Vectorizes your function, when possible
![Alt text](/assets/vectorizes_when_possible_real.png?raw=true)
![Alt text](/assets/vectorizes_when_possible_log10.png?raw=true)

## When vectorization is not possible, automatically decides which is faster: to use dask parallel processing or a simple pandas apply
![Alt text](/assets/multiprocessing_v_single_real.png?raw=true)
![Alt text](/assets/multiprocessing_v_single_log10.png?raw=true)

## Notes
1. The function is documented in the .py file. In Jupyter Notebooks, you can see the docs by pressing Shift+Tab(x3). Also, check out the complete documentation [here](docs/documentation.md) along with the [changelog](docs/changelog.md).

2. Please upgrade your version of pandas, as the pandas extension api used in this module is a recent addition to pandas.

3. When using swiftapply on a dataframe with a non-vectorized function call, swiftapply will resort to concatenating swiftapplies on each column because dask applymap does not work properly as of now. Submitting a bug fix shortly.
