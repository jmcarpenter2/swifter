config = {
    "npartitions": None,
    "dask_threshold": 1,
    "scheduler": "processes",
    "progress_bar": True,
    "progress_bar_desc": None,
    "allow_dask_on_strings": False,
}


def set_dask_threshold(dask_threshold=1):
    """
    Set the threshold (seconds) for maximum allowed estimated duration of
    pandas apply before switching to dask.
    """
    config["dask_threshold"] = dask_threshold


def set_dask_scheduler(scheduler="processes"):
    """
    Set the dask scheduler.

    :param scheduler: String, ["threads", "processes"]
    """
    config["scheduler"] = scheduler


def progress_bar(enable=True, desc=None):
    """
    Turn on/off the progress bar, and optionally add a custom description
    """
    config["progress_bar"] = enable
    config["progress_bar_desc"] = desc


def allow_dask_on_strings(enable=True):
    """
    Override the string processing default, which is to not use dask if a
    string is contained in the pandas object.
    """
    config["allow_dask_on_strings"] = enable
