# swifter [![Tweet](https://img.shields.io/twitter/url/http/shields.io.svg?style=social)](https://twitter.com/intent/tweet?text=Enjoying%20swifter%20optimized%20processing&url=https://www.github.com/jmcarpenter2/swifter&via=jcarpenter542&hashtags=python,pandas,parallel-processing,dask)
A package which efficiently applies any function to a pandas dataframe or series in the fastest available manner.

[![PyPI version](https://badge.fury.io/py/swifter.svg)](https://badge.fury.io/py/swifter)
[![CircleCI](https://circleci.com/gh/jmcarpenter2/swifter.svg?style=shield)](https://circleci.com/gh/jmcarpenter2/swifter)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)
![GitHub stars](https://img.shields.io/github/stars/jmcarpenter2/swifter.svg?style=popout)
![PyPI - Downloads](https://img.shields.io/pypi/dm/swifter.svg)

To know about latest improvements, please check [changelog](docs/changelog.md).

## Installation:
```
$ pip install -U pandas # upgrade pandas
$ pip install swifter # first time installation

$ pip install -U swifter # upgrade to latest version if already installed
```

It is also possible to install using `conda`:

```
conda install -c conda-forge swifter 
```

and then import into your code along with pandas using:
```
import pandas as pd
import swifter
```

## Easy to use
```
df = pd.DataFrame({'x': [1, 2, 3, 4], 'y': [5, 6, 7, 8]})

# runs on single core
df['x2'] = df['x'].apply(lambda x: x**2)
# runs on multiple cores
df['x2'] = df['x'].swifter.apply(lambda x: x**2)

# use swifter apply on whole dataframe
df['agg'] = df.swifter.apply(lambda x: x.sum() - x.min())

# use swifter apply on specific columns
df['outCol'] = df[['inCol1', 'inCol2']].swifter.apply(my_func)
df['outCol'] = df[['inCol1', 'inCol2', 'inCol3']].swifter.apply(my_func,
             positional_arg, keyword_arg=keyword_argval)
```

Further documentations on swifter is available [here](docs/documentation.md).

Check out the [examples notebook](examples/swifter_apply_examples.ipynb), along with the [speed benchmark notebook](examples/swiftapply_speedcomparison.ipynb)

## Vectorizes your function, when possible
![Alt text](/assets/vectorizes_when_possible_real.png?raw=true)
![Alt text](/assets/vectorizes_when_possible_log10.png?raw=true)

## When vectorization is not possible, automatically decides which is faster: to use dask parallel processing or a simple pandas apply
![Alt text](/assets/multiprocessing_v_single_real.png?raw=true)
![Alt text](/assets/multiprocessing_v_single_log10.png?raw=true)


## Notes
1. The function is documented in the .py file. In Jupyter Notebooks, you can see the docs by pressing Shift+Tab(x3). Also, check out the complete documentation [here](docs/documentation.md) along with the [changelog](docs/changelog.md).

2. Please upgrade your version of pandas, as the pandas extension api used in this module is a recent addition to pandas.

3. It is advised to disable the progress bar if calling swifter from a forked process as the progress bar may get confused between various multiprocessing modules. 
