import sys
import importlib
import unittest
import subprocess
import time
import logging
import warnings
from math import ceil
from psutil import cpu_count, virtual_memory

import numpy as np
import numpy.testing as npt
import pandas as pd
import swifter

from math import ceil, isclose
from tqdm.auto import tqdm


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


def run_if_modin_installed(cls):
    # if modin is installed, run the test/test suite
    if importlib.util.find_spec("modin") is not None:
        return cls
    else:  # if modin isnt installed just skip the test(s)
        return True


class TestSwifter(unittest.TestCase):
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

    def assertModinSeriesEqual(self, a, b, msg):
        try:
            npt.assert_array_almost_equal(a, b)
        except AssertionError as e:
            raise self.failureException(msg) from e

    def assertModinDataFrameEqual(self, a, b, msg):
        try:
            npt.assert_array_almost_equal(a, b)
        except AssertionError as e:
            raise self.failureException(msg) from e

    def modinSetUp(self):
        """
        Imports modin before swifter so that we have access to modin functionality
        """
        import modin.pandas as md
        import swifter

        swifter.register_modin()
        self.addTypeEqualityFunc(md.Series, self.assertModinSeriesEqual)
        self.addTypeEqualityFunc(md.DataFrame, self.assertModinDataFrameEqual)
        return md

    def setUp(self):
        LOG.info(f"Version {swifter.__version__}")
        self.addTypeEqualityFunc(pd.Series, self.assertSeriesEqual)
        self.addTypeEqualityFunc(pd.DataFrame, self.assertDataFrameEqual)
        self.ncores = cpu_count()


class TestSetup(TestSwifter):
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

    def test_set_ray_compute(self):
        LOG.info("test_set_ray_compute")
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
            [0.5, 0.99, 100000000],
            [
                ceil(virtual_memory().available * 0.5),
                ceil(virtual_memory().available * 0.99),
                100000000,
            ],
        ):
            before = swifter_df._ray_memory
            swifter_df.set_ray_compute(num_cpus=1, memory=set_ray_memory)
            actual = swifter_df._ray_memory
            self.assertTrue(isclose(actual, expected, rel_tol=0.2))
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
                swifter_df.set_ray_compute(memory=set_ray_memory)

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


class TestPandasSeries(TestSwifter):
    def test_apply_on_empty_series(self):
        LOG.info("test_apply_on_empty_series")
        series = pd.Series()
        pd_val = series.apply(math_foo, compare_to=1)
        swifter_val = series.swifter.apply(math_foo, compare_to=1)
        self.assertEqual(pd_val, swifter_val)  # equality test

    def test_nonvectorized_math_apply_on_small_series(self):
        LOG.info("test_nonvectorized_math_apply_on_small_series")
        df = pd.DataFrame({"x": np.random.normal(size=1000)})
        series = df["x"]
        tqdm.pandas(desc="Pandas Vec math apply ~ Series")
        pd_val = series.progress_apply(math_foo, compare_to=1)
        swifter_val = series.swifter.progress_bar(desc="Vec math apply ~ Series").apply(math_foo, compare_to=1)
        self.assertEqual(pd_val, swifter_val)  # equality test

    def test_nonvectorized_math_apply_on_small_series_no_progress_bar(self):
        LOG.info("test_nonvectorized_math_apply_on_small_series_no_progress_bar")
        df = pd.DataFrame({"x": np.random.normal(size=1000)})
        series = df["x"]
        pd_val = series.apply(math_foo, compare_to=1)
        swifter_val = series.swifter.progress_bar(enable=False).apply(math_foo, compare_to=1)
        self.assertEqual(pd_val, swifter_val)  # equality test

    def test_vectorized_math_apply_on_large_series(self):
        LOG.info("test_vectorized_math_apply_on_large_series")
        df = pd.DataFrame({"x": np.random.normal(size=1_000_000)})
        series = df["x"]

        tqdm.pandas(desc="Pandas Vec math apply ~ Series")
        start_pd = time.time()
        pd_val = series.progress_apply(math_vec_square)
        end_pd = time.time()
        pd_time = end_pd - start_pd

        start_swifter = time.time()
        swifter_val = (
            series.swifter.set_npartitions(4)
            .progress_bar(desc="Vec math apply ~ Series")
            .apply(math_vec_square, axis=0)
        )
        end_swifter = time.time()
        swifter_time = end_swifter - start_swifter

        self.assertEqual(pd_val, swifter_val)  # equality test
        if self.ncores > 1:  # speed test
            self.assertLess(swifter_time, pd_time)

    def test_nonvectorized_math_apply_on_large_series(self):
        LOG.info("test_nonvectorized_math_apply_on_large_series")
        df = pd.DataFrame({"x": np.random.normal(size=10_000_000)})
        series = df["x"]

        tqdm.pandas(desc="Pandas Nonvec math apply ~ Series")
        start_pd = time.time()
        pd_val = series.progress_apply(math_foo, compare_to=1)
        end_pd = time.time()
        pd_time = end_pd - start_pd

        start_swifter = time.time()
        swifter_val = (
            series.swifter.set_npartitions(4)
            .progress_bar(desc="Nonvec math apply ~ Series")
            .apply(math_foo, compare_to=1)
        )
        end_swifter = time.time()
        swifter_time = end_swifter - start_swifter

        self.assertEqual(pd_val, swifter_val)  # equality test
        if self.ncores > 1:  # speed test
            self.assertLess(swifter_time, pd_time)


class TestPandasDataFrame(TestSwifter):
    def test_apply_on_empty_dataframe(self):
        LOG.info("test_apply_on_empty_dataframe")
        df = pd.DataFrame(columns=["x", "y"])
        pd_val = df.apply(math_vec_multiply, axis=1)
        swifter_val = df.swifter.apply(math_vec_multiply, axis=1)
        self.assertEqual(pd_val, swifter_val)  # equality test

    def test_applymap_on_empty_dataframe(self):
        LOG.info("test_applymap_on_empty_dataframe")
        df = pd.DataFrame(columns=["x", "y"])
        pd_val = df.applymap(math_vec_square)
        swifter_val = df.swifter.applymap(math_vec_square)
        self.assertEqual(pd_val, swifter_val)  # equality test

    def test_nonvectorized_math_apply_on_small_dataframe(self):
        LOG.info("test_nonvectorized_math_apply_on_small_dataframe")
        df = pd.DataFrame({"x": np.random.normal(size=1000), "y": np.random.uniform(size=1000)})
        tqdm.pandas(desc="Pandas Nonvec math apply ~ DF")
        pd_val = df.progress_apply(math_agg_foo)
        swifter_val = df.swifter.progress_bar(desc="Vec math apply ~ DF").apply(math_agg_foo)
        self.assertEqual(pd_val, swifter_val)  # equality test

    def test_nonvectorized_math_apply_on_small_dataframe_no_progress_bar(self):
        LOG.info("test_nonvectorized_math_apply_on_small_dataframe_no_progress_bar")
        df = pd.DataFrame({"x": np.random.normal(size=1000), "y": np.random.uniform(size=1000)})
        pd_val = df.apply(math_agg_foo)
        swifter_val = df.swifter.progress_bar(enable=False).apply(math_agg_foo)
        self.assertEqual(pd_val, swifter_val)  # equality test

    def test_vectorized_math_apply_on_large_dataframe(self):
        LOG.info("test_vectorized_math_apply_on_large_dataframe")
        df = pd.DataFrame({"x": np.random.normal(size=1_000_000), "y": np.random.uniform(size=1_000_000)})

        tqdm.pandas(desc="Pandas Vec math apply ~ DF")
        start_pd = time.time()
        pd_val = df.progress_apply(math_vec_multiply, axis=1)
        end_pd = time.time()
        pd_time = end_pd - start_pd

        start_swifter = time.time()
        swifter_val = (
            df.swifter.set_npartitions(4).progress_bar(desc="Vec math apply ~ DF").apply(math_vec_multiply, axis=1)
        )
        end_swifter = time.time()
        swifter_time = end_swifter - start_swifter

        self.assertEqual(pd_val, swifter_val)  # equality test
        if self.ncores > 1:  # speed test
            self.assertLess(swifter_time, pd_time)

    def test_nonvectorized_math_apply_on_large_dataframe_broadcast(self):
        LOG.info("test_nonvectorized_math_apply_on_large_dataframe_broadcast")
        df = pd.DataFrame({"x": np.random.normal(size=250_000), "y": np.random.uniform(size=250_000)})

        tqdm.pandas(desc="Pandas Nonvec math apply + broadcast ~ DF")
        start_pd = time.time()
        pd_val = df.progress_apply(math_agg_foo, axis=1, result_type="broadcast")
        end_pd = time.time()
        pd_time = end_pd - start_pd

        start_swifter = time.time()
        swifter_val = (
            df.swifter.set_npartitions(4)
            .progress_bar(desc="Nonvec math apply + broadcast ~ DF")
            .apply(math_agg_foo, axis=1, result_type="broadcast")
        )
        end_swifter = time.time()
        swifter_time = end_swifter - start_swifter

        self.assertEqual(pd_val, swifter_val)  # equality test
        if self.ncores > 1:  # speed test
            self.assertLess(swifter_time, pd_time)

    def test_nonvectorized_math_apply_on_large_dataframe_reduce(self):
        LOG.info("test_nonvectorized_math_apply_on_large_dataframe_reduce")
        df = pd.DataFrame({"x": np.random.normal(size=250_000), "y": np.random.uniform(size=250_000)})

        tqdm.pandas(desc="Pandas Nonvec math apply + reduce ~ DF")
        start_pd = time.time()
        pd_val = df.progress_apply(math_agg_foo, axis=1, result_type="reduce")
        end_pd = time.time()
        pd_time = end_pd - start_pd

        start_swifter = time.time()
        swifter_val = (
            df.swifter.set_npartitions(4)
            .progress_bar(desc="Nonvec math apply + reduce ~ DF")
            .apply(math_agg_foo, axis=1, result_type="reduce")
        )
        end_swifter = time.time()
        swifter_time = end_swifter - start_swifter

        self.assertEqual(pd_val, swifter_val)  # equality test
        if self.ncores > 1:  # speed test
            self.assertLess(swifter_time, pd_time)

    def test_nonvectorized_text_dask_apply_on_large_dataframe(self):
        LOG.info("test_nonvectorized_text_dask_apply_on_large_dataframe")
        df = pd.DataFrame({"letter": ["A", "B", "C", "D", "E"] * 200_000, "value": np.random.normal(size=1_000_000)})

        tqdm.pandas(desc="Pandas Nonvec text apply ~ DF")
        start_pd = time.time()
        pd_val = df.progress_apply(text_foo, axis=1)
        end_pd = time.time()
        pd_time = end_pd - start_pd

        start_swifter = time.time()
        swifter_val = (
            df.swifter.allow_dask_on_strings(True)
            .set_npartitions(4)
            .progress_bar(desc="Nonvec Dask text apply ~ DF")
            .apply(text_foo, axis=1)
        )
        end_swifter = time.time()
        swifter_time = end_swifter - start_swifter

        self.assertEqual(pd_val, swifter_val)  # equality test
        if self.ncores > 1:  # speed test
            self.assertLess(swifter_time, pd_time)

    @run_if_modin_installed
    def test_nonvectorized_text_modin_apply_on_large_dataframe(self):
        LOG.info("test_nonvectorized_text_modin_apply_on_large_dataframe")
        self.modinSetUp()
        df = pd.DataFrame({"letter": ["I", "You", "We"] * 1_000_000, "value": ["want to break free"] * 3_000_000})

        tqdm.pandas(desc="Pandas Nonvec text apply ~ DF")
        start_pd = time.time()
        pd_val = df.progress_apply(clean_text_foo, axis=1)
        end_pd = time.time()
        pd_time = end_pd - start_pd

        start_swifter = time.time()
        swifter_val = (
            df.swifter.allow_dask_on_strings(False)
            .set_npartitions(4)
            .set_ray_compute(num_cpus=2 if self.ncores >= 2 else 1, memory=0.25)
            .progress_bar(desc="Nonvec Modin text apply ~ DF")
            .apply(clean_text_foo, axis=1)
        )
        end_swifter = time.time()
        swifter_time = end_swifter - start_swifter

        self.assertEqual(pd_val, swifter_val)  # equality test
        if self.ncores > 1:  # speed test
            self.assertLess(swifter_time, pd_time)

    @run_if_modin_installed
    def test_nonvectorized_text_modin_apply_on_large_dataframe_returns_series(self):
        LOG.info("test_nonvectorized_text_modin_apply_on_large_dataframe_returns_series")
        self.modinSetUp()
        df = pd.DataFrame({"str_date": ["2000/01/01 00:00:00"] * 1_000_000})

        tqdm.pandas(desc="Pandas Nonvec text apply ~ DF -> Srs")
        start_pd = time.time()
        pd_val = df.progress_apply(lambda row: row["str_date"].split()[0], axis=1)
        end_pd = time.time()
        pd_time = end_pd - start_pd

        start_swifter = time.time()
        swifter_val = (
            df.swifter.allow_dask_on_strings(False)
            .set_npartitions(4)
            .set_ray_compute(num_cpus=2 if self.ncores >= 2 else 1, memory=0.25)
            .progress_bar(desc="Nonvec Modin text apply ~ DF -> Srs")
            .apply(lambda row: row["str_date"].split()[0], axis=1)
        )
        end_swifter = time.time()
        swifter_time = end_swifter - start_swifter

        self.assertEqual(pd_val, swifter_val)  # equality test
        if self.ncores > 1:  # speed test
            self.assertLess(swifter_time, pd_time)

    def test_vectorized_math_applymap_on_large_dataframe(self):
        LOG.info("test_vectorized_math_applymap_on_large_dataframe")
        df = pd.DataFrame({"x": np.random.normal(size=1_000_000), "y": np.random.uniform(size=1_000_000)})

        tqdm.pandas(desc="Pandas Vec math applymap ~ DF")
        start_pd = time.time()
        pd_val = df.progress_applymap(math_vec_square)
        end_pd = time.time()
        pd_time = end_pd - start_pd

        start_swifter = time.time()
        swifter_val = (
            df.swifter.set_npartitions(4).progress_bar(desc="Vec math applymap ~ DF").applymap(math_vec_square)
        )
        end_swifter = time.time()
        swifter_time = end_swifter - start_swifter

        self.assertEqual(pd_val, swifter_val)  # equality test
        if self.ncores > 1:  # speed test
            self.assertLess(swifter_time, pd_time)

    def test_nonvectorized_math_applymap_on_large_dataframe(self):
        LOG.info("test_nonvectorized_math_applymap_on_large_dataframe")
        df = pd.DataFrame({"x": np.random.normal(size=5_000_000), "y": np.random.uniform(size=5_000_000)})

        tqdm.pandas(desc="Pandas Nonvec math applymap ~ DF")
        start_pd = time.time()
        pd_val = df.progress_applymap(math_foo)
        end_pd = time.time()
        pd_time = end_pd - start_pd

        start_swifter = time.time()
        swifter_val = df.swifter.set_npartitions(4).progress_bar(desc="Nonvec math applymap ~ DF").applymap(math_foo)
        end_swifter = time.time()
        swifter_time = end_swifter - start_swifter

        self.assertEqual(pd_val, swifter_val)  # equality test
        if self.ncores > 1:  # speed test
            self.assertLess(swifter_time, pd_time)

    def test_nonvectorized_math_applymap_on_small_dataframe(self):
        LOG.info("test_nonvectorized_math_applymap_on_small_dataframe")
        df = pd.DataFrame({"x": np.random.normal(size=1000), "y": np.random.uniform(size=1000)})
        pd_val = df.applymap(math_foo)
        swifter_val = df.swifter.set_npartitions(4).applymap(math_foo)
        self.assertEqual(pd_val, swifter_val)  # equality test

    def test_nonvectorized_math_applymap_on_small_dataframe_no_progress_bar(self):
        LOG.info("test_nonvectorized_math_applymap_on_small_dataframe_no_progress_bar")
        df = pd.DataFrame({"x": np.random.normal(size=1000), "y": np.random.uniform(size=1000)})
        pd_val = df.applymap(math_foo)
        swifter_val = df.swifter.progress_bar(enable=False).applymap(math_foo)
        self.assertEqual(pd_val, swifter_val)  # equality test


class TestPandasTransformation(TestSwifter):
    def test_rolling_apply_on_empty_dataframe(self):
        LOG.info("test_rolling_apply_on_empty_dataframe")
        df = pd.DataFrame(columns=["x", "y"])
        pd_val = df.rolling(1).apply(math_agg_foo, raw=True)
        swifter_val = df.swifter.set_npartitions(4).rolling(1).apply(math_agg_foo, raw=True)
        self.assertEqual(pd_val, swifter_val)  # equality test

    def test_resample_apply_on_empty_dataframe(self):
        LOG.info("test_resample_apply_on_empty_dataframe")
        df = pd.DataFrame(columns=["x", "y"], index=pd.date_range(start="2020/01/01", periods=0))
        pd_val = df.resample("1d").apply(math_agg_foo)
        swifter_val = df.swifter.set_npartitions(4).resample("1d").apply(math_agg_foo)
        self.assertEqual(pd_val, swifter_val)  # equality test

    def test_nonvectorized_math_apply_on_small_rolling_dataframe(self):
        LOG.info("test_nonvectorized_math_apply_on_small_rolling_dataframe")
        df = pd.DataFrame({"x": np.arange(0, 1000)}, index=pd.date_range("2019-01-1", "2020-01-1", periods=1000))
        pd_val = df.rolling("1d").apply(math_agg_foo, raw=True)
        swifter_val = (
            df.swifter.set_npartitions(4)
            .rolling("1d")
            .progress_bar(desc="Nonvec math apply ~ Rolling DF")
            .apply(math_agg_foo, raw=True)
        )
        self.assertEqual(pd_val, swifter_val)  # equality test

    def test_nonvectorized_math_apply_on_small_rolling_dataframe_no_progress_bar(self):
        LOG.info("test_nonvectorized_math_apply_on_small_rolling_dataframe_no_progress_bar")
        df = pd.DataFrame({"x": np.arange(0, 1000)}, index=pd.date_range("2019-01-1", "2020-01-1", periods=1000))
        pd_val = df.rolling("1d").apply(math_agg_foo, raw=True)
        swifter_val = (
            df.swifter.set_npartitions(4).rolling("1d").progress_bar(enable=False).apply(math_agg_foo, raw=True)
        )
        self.assertEqual(pd_val, swifter_val)  # equality test

    def test_vectorized_math_apply_on_large_rolling_dataframe(self):
        LOG.info("test_vectorized_math_apply_on_large_rolling_dataframe")
        df = pd.DataFrame(
            {"x": np.arange(0, 1_000_000)}, index=pd.date_range("2019-01-1", "2020-01-1", periods=1_000_000)
        )
        pd_val = df.rolling("1d").apply(max, raw=True)
        swifter_val = (
            df.swifter.set_npartitions(4)
            .rolling("1d")
            .progress_bar(desc="Vec math apply ~ Rolling DF")
            .apply(max, raw=True)
        )
        self.assertEqual(pd_val, swifter_val)  # equality test

    def test_nonvectorized_math_apply_on_large_rolling_dataframe(self):
        LOG.info("test_nonvectorized_math_apply_on_large_rolling_dataframe")
        df = pd.DataFrame(
            {"x": np.arange(0, 7_000_000)}, index=pd.date_range("2019-01-1", "2020-01-1", periods=7_000_000)
        )

        start_pd = time.time()
        pd_val = df.rolling("3T").apply(math_agg_foo, raw=True)
        end_pd = time.time()
        pd_time = end_pd - start_pd

        start_swifter = time.time()
        swifter_val = (
            df.swifter.set_npartitions(7)
            .rolling("3T")
            .progress_bar(desc="Nonvec math apply ~ Rolling DF")
            .apply(math_agg_foo, raw=True)
        )
        end_swifter = time.time()
        swifter_time = end_swifter - start_swifter

        self.assertEqual(pd_val, swifter_val)  # equality test
        if self.ncores > 1:  # speed test
            self.assertLess(swifter_time, pd_time)

    def test_nonvectorized_math_apply_on_small_resampler_dataframe(self):
        LOG.info("test_nonvectorized_math_apply_on_small_resampler_dataframe")
        df = pd.DataFrame({"x": np.arange(0, 1000)}, index=pd.date_range("2019-01-1", "2020-01-1", periods=1000))
        pd_val = df.resample("1M").apply(math_agg_foo)
        swifter_val = (
            df.swifter.set_npartitions(4)
            .resample("1M")
            .progress_bar(desc="Nonvec math apply ~ Resample DF")
            .apply(math_agg_foo)
        )
        self.assertEqual(pd_val, swifter_val)  # equality test

    def test_nonvectorized_math_apply_on_large_resampler_dataframe(self):
        LOG.info("test_nonvectorized_math_apply_on_large_resampler_dataframe")
        df = pd.DataFrame(
            {"x": np.arange(0, 1_000_000)}, index=pd.date_range("2019-01-1", "2020-01-1", periods=1_000_000)
        )

        start_pd = time.time()
        pd_val = df.resample("3T").apply(math_agg_foo)
        end_pd = time.time()
        pd_time = end_pd - start_pd

        start_swifter = time.time()
        swifter_val = (
            df.swifter.set_npartitions(4)
            .resample("3T")
            .progress_bar(desc="Nonvec math apply ~ Resample DF")
            .apply(math_agg_foo)
        )
        end_swifter = time.time()
        swifter_time = end_swifter - start_swifter

        self.assertEqual(pd_val, swifter_val)  # equality test
        if self.ncores > 1:  # speed test
            self.assertLess(swifter_time, pd_time)


@run_if_modin_installed
class TestModinSeries(TestSwifter):
    def test_modin_series_warns_on_missing_attributes(self):
        LOG.info("test_modin_series_warns_on_missing_attributes")
        md = self.modinSetUp()
        series = md.Series()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            series.swifter.set_dask_threshold(1)
            self.assertEqual(len(w), 1)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            series.swifter.set_dask_scheduler("threads")
            self.assertEqual(len(w), 1)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            series.swifter.allow_dask_on_strings(True)
            self.assertEqual(len(w), 1)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            series.swifter.progress_bar(False)
            self.assertEqual(len(w), 1)

    def test_modin_series_errors_on_missing_transformations(self):
        LOG.info("test_modin_series_errors_on_missing_transformations")
        md = self.modinSetUp()
        series = md.Series()
        with self.assertRaises(NotImplementedError):
            series.swifter.rolling(1)

        with self.assertRaises(NotImplementedError):
            series.swifter.resample(1)

    def test_apply_on_empty_modin_series(self):
        LOG.info("test_apply_on_empty_series")
        md = self.modinSetUp()
        series = md.Series()
        md_val = series.apply(math_foo, compare_to=1)
        swifter_val = series.swifter.apply(math_foo, compare_to=1)
        self.assertEqual(md_val, swifter_val)  # equality test

    def test_nonvectorized_modin_apply_on_small_series(self):
        LOG.info("test_nonvectorized_modin_apply_on_small_series")
        md = self.modinSetUp()
        df = md.Series(np.random.normal(size=200_000), name="x")
        md_val = df.apply(math_foo)
        swifter_val = df.swifter.set_npartitions(4).apply(math_foo)
        self.assertEqual(md_val, swifter_val)  # equality test

    def test_vectorized_modin_apply_on_large_series(self):
        LOG.info("test_vectorized_modin_apply_on_large_series")
        md = self.modinSetUp()
        df = md.Series(np.random.uniform(size=20_000_000), name="x")
        start_md = time.time()
        md_val = df.apply(math_vec_square, axis=0)
        md_pd_val = md_val._to_pandas()  # We have to bring it into pandas to confirm swifter apply speed is quicker
        end_md = time.time()
        md_time = end_md - start_md

        start_swifter = time.time()
        swifter_val = df.swifter.set_npartitions(4).apply(math_vec_square)
        swifter_pd_val = (
            swifter_val._to_pandas()
        )  # We have to bring it into pandas to confirm swifter apply speed is quicker
        end_swifter = time.time()
        swifter_time = end_swifter - start_swifter

        self.assertEqual(md_val, swifter_val)  # equality test
        self.assertEqual(md_pd_val, swifter_pd_val)  # equality test after converting to pandas
        self.assertLess(swifter_time, md_time)  # speed test


@run_if_modin_installed
class TestModinDataFrame(TestSwifter):
    def test_modin_dataframe_warns_on_missing_attributes(self):
        LOG.info("test_modin_dataframe_warns_on_missing_attributes")
        md = self.modinSetUp()
        df = md.DataFrame()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            df.swifter.set_dask_threshold(1)
            self.assertEqual(len(w), 1)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            df.swifter.set_dask_scheduler("threads")
            self.assertEqual(len(w), 1)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            df.swifter.allow_dask_on_strings(True)
            self.assertEqual(len(w), 1)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            df.swifter.progress_bar(False)
            self.assertEqual(len(w), 1)

    def test_modin_dataframe_errors_on_missing_transformations(self):
        LOG.info("test_modin_dataframe_errors_on_missing_transformations")
        md = self.modinSetUp()
        df = md.DataFrame()
        with self.assertRaises(NotImplementedError):
            df.swifter.rolling(1)

        with self.assertRaises(NotImplementedError):
            df.swifter.resample(1)

    def test_apply_on_empty_modin_dataframe(self):
        LOG.info("test_apply_on_empty_series")
        md = self.modinSetUp()
        df = md.DataFrame()
        md_val = df.apply(math_foo, compare_to=1)
        swifter_val = df.swifter.apply(math_foo, compare_to=1)
        self.assertEqual(md_val, swifter_val)  # equality test

    def test_nonvectorized_modin_apply_on_small_dataframe(self):
        LOG.info("test_nonvectorized_modin_apply_on_small_dataframe")
        md = self.modinSetUp()
        df = md.DataFrame({"letter": ["A", "B", "C", "D", "E"] * 200_000, "value": np.random.normal(size=1_000_000)})
        md_val = df.apply(text_foo, axis=1)
        swifter_val = df.swifter.set_npartitions(4).apply(text_foo, axis=1)
        self.assertEqual(md_val, swifter_val)  # equality test

    def test_vectorized_modin_apply_on_large_dataframe(self):
        LOG.info("test_vectorized_modin_apply_on_large_dataframe")
        md = self.modinSetUp()
        df = md.DataFrame({"x": np.random.normal(size=1_000_000), "y": np.random.uniform(size=1_000_000)})
        start_md = time.time()
        md_val = df.apply(math_vec_square, axis=1)
        md_pd_val = md_val._to_pandas()  # We have to bring it into pandas to confirm swifter apply speed is quicker
        end_md = time.time()
        md_time = end_md - start_md

        start_swifter = time.time()
        swifter_val = df.swifter.set_npartitions(4).apply(math_vec_square, axis=1)
        swifter_pd_val = (
            swifter_val._to_pandas()
        )  # We have to bring it into pandas to confirm swifter apply speed is quicker
        end_swifter = time.time()
        swifter_time = end_swifter - start_swifter

        self.assertEqual(md_val, swifter_val)  # equality test
        self.assertEqual(md_pd_val, swifter_pd_val)  # equality test after converting to pandas
        self.assertLess(swifter_time, md_time)  # speed test
