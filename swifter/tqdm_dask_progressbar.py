from dask.callbacks import Callback
from tqdm import tqdm_notebook


class TQDMDaskProgressBar(Callback, object):
    """
    A tqdm progress bar for dask.

    Usage:
        ```
        with TQDMDaskProgressBar():
            da.compute()
        ```
    Author : wassname
    Source : https://gist.github.com/wassname/1837d0365247430e02abda41f0e7f184
    See: http://dask.pydata.org/en/latest/diagnostics-local.html?highlight=progress
    """

    def __init__(self, start=None, start_state=None, pretask=None, posttask=None, finish=None, **kwargs):
        super(TQDMDaskProgressBar, self).__init__(start=start, start_state=start_state, pretask=pretask, posttask=posttask, finish=finish)
        self.tqdm_args = kwargs
        self.states = ["ready", "waiting", "running", "finished"]

    def _start_state(self, dsk, state):
        self._tqdm = tqdm_notebook(total=sum(len(state[k]) for k in self.states), **self.tqdm_args)

    def _posttask(self, key, result, dsk, state, worker_id):
        self._tqdm.update(1)

    def _finish(self, dsk, state, errored):
        self._tqdm.close()
