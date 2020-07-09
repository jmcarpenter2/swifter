import sys
import unittest
import subprocess
import time
import logging

import numpy as np
import pandas as pd
import swifter

from math import ceil
from tqdm.auto import tqdm

from psutil import cpu_count, virtual_memory

LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)-8s.%(msecs)03d %(levelname)-8s %(name)s:%(lineno)-3s %(message)s")
ch.setFormatter(formatter)
LOG.addHandler(ch)


def math_vec_square(x):
    return x ** 2


def math_foo(x, compare_to=1):
    return x ** 2 if x < compare_to else x ** (1 / 2)


def math_vec_multiply(row):
    return row["x"] * row["y"]


def math_agg_foo(row):
    return row.sum() - row.min()


def text_foo(row):
    if row["letter"] == "A":
        return row["value"] * 3
    elif row["letter"] == "B":
        return row["value"] ** 3
    elif row["letter"] == "C":
        return row["value"] / 3
    elif row["letter"] == "D":
        return row["value"] ** (1 / 3)
    elif row["letter"] == "E":
        return row["value"]


def clean_text_foo(row):
    text = " ".join(row)
    text = text.strip()
    text = text.replace(" ", "_")
    return text


class TestSetup(unittest.TestCase):
    def assertSeriesEqual(self, a, b, msg):
        try:
            pd.testing.assert_series_equal(a, b)
        except AssertionError as e:
            raise self.failureException(msg) from e

    def assertDataFrameEqual(self, a, b, msg):
        try:
            pd.testing.assert_frame_equal(a, b)
        except AssertionError as e:
            raise self.failureException(msg) from e

    def setUp(self):
        LOG.info(f"Version {swifter.__version__}")
        self.addTypeEqualityFunc(pd.Series, self.assertSeriesEqual)
        self.addTypeEqualityFunc(pd.DataFrame, self.assertDataFrameEqual)
        self.ncores = cpu_count()

    def test_set_npartitions(self):
        LOG.info("test_set_npartitions")
        for swifter_df, set_npartitions, expected in zip(
            [
                pd.DataFrame().swifter,
                pd.Series().swifter,
                pd.DataFrame(
                    {"x": np.arange(0, 10)}, index=pd.date_range("2019-01-1", "2020-01-1", periods=10)
                ).swifter.rolling("1d"),
                pd.DataFrame(
                    {"x": np.arange(0, 10)}, index=pd.date_range("2019-01-1", "2020-01-1", periods=10)
                ).swifter.resample("3T"),
            ],
            [None, 1000, 1001, 1002],
            [cpu_count() * 2, 1000, 1001, 1002],
        ):
            before = swifter_df._npartitions
            swifter_df.set_npartitions(set_npartitions)
            actual = swifter_df._npartitions
            self.assertEqual(actual, expected)
            if set_npartitions is not None:
                self.assertNotEqual(before, actual)

    def test_set_ray_memory(self):
        LOG.info("test_set_ray_memory")
        for swifter_df, set_ray_memory, expected in zip(
            [
                pd.DataFrame().swifter,
                pd.Series().swifter,
                pd.DataFrame(
                    {"x": np.arange(0, 10)}, index=pd.date_range("2019-01-1", "2020-01-1", periods=10)
                ).swifter.rolling("1d"),
                pd.DataFrame(
                    {"x": np.arange(0, 10)}, index=pd.date_range("2019-01-1", "2020-01-1", periods=10)
                ).swifter.resample("3T"),
            ],
            [None, 0.5, 0.99, 100000],
            [
                ceil(virtual_memory().available * 3 / 4),
                ceil(virtual_memory().available * 0.5),
                ceil(virtual_memory().available * 0.99),
                100000,
            ],
        ):
            before = swifter_df._ray_memory
            swifter_df.set_ray_memory(set_ray_memory)
            actual = swifter_df._ray_memory
            self.assertEqual(actual, expected)
            self.assertNotEqual(before, actual)

    def test_cant_set_ray_memory_OOM(self):
        LOG.info("test_cant_set_ray_memory_OOM")
        for swifter_df, set_ray_memory in zip(
            [
                pd.DataFrame().swifter,
                pd.Series().swifter,
                pd.DataFrame(
                    {"x": np.arange(0, 10)}, index=pd.date_range("2019-01-1", "2020-01-1", periods=10)
                ).swifter.rolling("1d"),
                pd.DataFrame(
                    {"x": np.arange(0, 10)}, index=pd.date_range("2019-01-1", "2020-01-1", periods=10)
                ).swifter.resample("3T"),
            ],
            [1e100, 1e100, 1e100, 1e100],
        ):
            with self.assertRaises(MemoryError):
                swifter_df.set_ray_memory(set_ray_memory)

    def test_set_dask_threshold(self):
        LOG.info("test_set_dask_threshold")
        expected = 1000
        for swifter_df in [
            pd.DataFrame().swifter,
            pd.Series().swifter,
            pd.DataFrame(
                {"x": np.arange(0, 10)}, index=pd.date_range("2019-01-1", "2020-01-1", periods=10)
            ).swifter.rolling("1d"),
            pd.DataFrame(
                {"x": np.arange(0, 10)}, index=pd.date_range("2019-01-1", "2020-01-1", periods=10)
            ).swifter.resample("3T"),
        ]:
            before = swifter_df._dask_threshold
            swifter_df.set_dask_threshold(expected)
            actual = swifter_df._dask_threshold
            self.assertEqual(actual, expected)
            self.assertNotEqual(before, actual)

    def test_set_dask_scheduler(self):
        LOG.info("test_set_dask_scheduler")
        expected = "my-scheduler"
        for swifter_df in [
            pd.DataFrame().swifter,
            pd.Series().swifter,
            pd.DataFrame(
                {"x": np.arange(0, 10)}, index=pd.date_range("2019-01-1", "2020-01-1", periods=10)
            ).swifter.rolling("1d"),
            pd.DataFrame(
                {"x": np.arange(0, 10)}, index=pd.date_range("2019-01-1", "2020-01-1", periods=10)
            ).swifter.resample("3T"),
        ]:
            before = swifter_df._scheduler
            swifter_df.set_dask_scheduler(expected)
            actual = swifter_df._scheduler
            self.assertEqual(actual, expected)
            self.assertNotEqual(before, actual)

    def test_disable_progress_bar(self):
        LOG.info("test_disable_progress_bar")
        expected = False
        for swifter_df in [
            pd.DataFrame().swifter,
            pd.Series().swifter,
            pd.DataFrame(
                {"x": np.arange(0, 10)}, index=pd.date_range("2019-01-1", "2020-01-1", periods=10)
            ).swifter.rolling("1d"),
            pd.DataFrame(
                {"x": np.arange(0, 10)}, index=pd.date_range("2019-01-1", "2020-01-1", periods=10)
            ).swifter.resample("3T"),
        ]:
            before = swifter_df._progress_bar
            swifter_df.progress_bar(expected)
            actual = swifter_df._progress_bar
            self.assertEqual(actual, expected)
            self.assertNotEqual(before, actual)

    def test_allow_dask_on_strings(self):
        LOG.info("test_allow_dask_on_strings")
        expected = True
        swifter_df = pd.DataFrame().swifter
        before = swifter_df._allow_dask_on_strings
        swifter_df.allow_dask_on_strings(expected)
        actual = swifter_df._allow_dask_on_strings
        self.assertEqual(actual, expected)
        self.assertNotEqual(before, actual)

    def test_stdout_redirected(self):
        LOG.info("test_stdout_redirected")
        print_messages = subprocess.check_output(
            [
                sys.executable,
                "-c",
                "import pandas as pd; import numpy as np; import swifter; "
                + "df = pd.DataFrame({'x': np.random.normal(size=4)}, dtype='float32'); "
                + "df.swifter.progress_bar(enable=False).apply(lambda x: print(x.values))",
            ],
            stderr=subprocess.STDOUT,
        )
        self.assertEqual(len(print_messages.decode("utf-8").rstrip("\n").split("\n")), 1)
