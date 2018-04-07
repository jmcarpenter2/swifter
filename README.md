# swifter
A package which efficiently applies any function to a pandas dataframe or series in the fastest available manner

Installation:
```
$pip install swifter # first time installation
$pip install -U swifter # upgrade to latest version
``` 

and then import into your code using:
```
from swifter import swiftapply
```

## Easy to use
```
myDF['outCol'] = swiftapply(myDF['inCol'], my_func, my_func_arg=my_func_argval)
```

## Notes
1. The function is documented in the .py file. In Jupyter Notebooks, you can see the docs by pressing Shift+Tab(x3). Also, check out the complete documentation [here](docs/documentation.md) along with the [changelog](docs/changelog.md).
