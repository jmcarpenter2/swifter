# Configuration

You can configure `swifter` on a global level, which impacts all instances, or
on a local (inline) level, which impacts only that instance of
`.swifter.apply`.

## Global configuration

You can set global configuration options by calling the corresponding method on
the main import `swifter`.

```python
import swifter

swifter.set_dask_threshold(dask_threshold=1)
swifter.set_dask_scheduler(scheduler="processes")
swifter.progress_bar(enable=True, desc=None)
swifter.allow_dask_on_strings(enable=True)

# Global configuration is stored in the following dictionary
print(swifter.config)
# => {
#        "npartitions": None,
#        "dask_threshold": 1,
#        "scheduler": "processes",
#        "progress_bar": True,
#        "progress_bar_desc": None,
#        "allow_dask_on_strings": False,
#    }
```

## Inline configuration

You can set local configuration options for a specific `.swifter.apply` by
calling the corresponding method on the accessor.

```python
import pandas as pd
import swifter

series = pd.Series(np.arange(1000))

def square(x): return x ** 2

series.swifter.set_dask_threshold(dask_threshold=1).apply(square)
series.swifter.set_dask_scheduler(scheduler="processes").apply(square)
series.swifter.progress_bar(enable=True, desc=None).apply(square)
series.swifter.allow_dask_on_strings(enable=True).apply(square)
```
