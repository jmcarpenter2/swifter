import os
import sys
import importlib
import unittest
import subprocess
import time
import logging
import warnings
from psutil import cpu_count

import numpy as np
import numpy.testing as npt
import pandas as pd
import swifter

from .swifter import RAY_INSTALLED, GROUPBY_MAX_ROWS_PANDAS_DEFAULT

from tqdm.auto import tqdm

WINDOWS_CI = "windows" in os.environ.get("CIRCLE_JOB", "")


LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)-8s.%(msecs)03d %(levelname)-8s %(name)s:%(lineno)-3s %(message)s")
ch.setFormatter(formatter)
LOG.addHandler(ch)


def math_vec_square(x):
    return x**2


def math_foo(x, compare_to=1):
    return x**2 if x < compare_to else x ** (1 / 2)


def math_vec_multiply(row):
    return row["x"] * row["y"]


def math_agg_foo(row):
    return row.sum() - row.min()


def numeric_func(x):
    return x["x"].mean() / x["y"].var()


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


def run_if_ray_installed(func):
    # if ray is installed, run the test/test suite
    if RAY_INSTALLED:
        return func
    else:  # if ray isnt installed just skip the test(s)
        return True


class TestSwifter(unittest.TestCase):
    def assertLessLinux(self, a, b, msg=None):
        if WINDOWS_CI:
            pass
        else:
            super().assertLess(a, b, msg=msg)

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
        import os

        os.environ["MODIN_ENGINE"] = "dask"
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
    def test_set_defaults(self):
        LOG.info("test_set_defaults")
        from swifter import set_defaults

        expected_npartitions = 2
        expected_dask_threshold = 1.5
        expected_scheduler = "threads"
        expected_progress_bar = False
        expected_progress_bar_desc = "TEST"
        expected_allow_dask_on_strings = True
        expected_force_parallel = True
        set_defaults(
            npartitions=expected_npartitions,
            dask_threshold=expected_dask_threshold,
            scheduler=expected_scheduler,
            progress_bar=expected_progress_bar,
            progress_bar_desc=expected_progress_bar_desc,
            allow_dask_on_strings=expected_allow_dask_on_strings,
            force_parallel=expected_force_parallel,
        )
        for swifter_df in [
            pd.DataFrame().swifter,
            pd.Series().swifter,
            pd.DataFrame(
                {"x": np.arange(0, 10)},
                index=pd.date_range("2019-01-1", "2020-01-1", periods=10),
            ).swifter.rolling("1d"),
            pd.DataFrame(
                {"x": np.arange(0, 10)},
                index=pd.date_range("2019-01-1", "2020-01-1", periods=10),
            ).swifter.resample("3T"),
        ]:
            swifter_df._npartitions == expected_npartitions
            swifter_df._dask_threshold == expected_dask_threshold
            swifter_df._scheduler == expected_scheduler
            swifter_df._progress_bar == expected_progress_bar
            swifter_df._progress_bar_desc == expected_progress_bar_desc
            swifter_df._allow_dask_on_strings == expected_allow_dask_on_strings
            swifter_df._force_parallel == expected_force_parallel

    def test_override_defaults(self):
        LOG.info("test_set_defaults")
        from swifter import set_defaults

        set_npartitions = 2
        set_dask_threshold = 1.5
        set_scheduler = "threads"
        set_progress_bar = False
        set_progress_bar_desc = "TEST"
        set_allow_dask_on_strings = True
        set_force_parallel = True

        expected_npartitions = 3
        expected_dask_threshold = 4.5
        expected_scheduler = "processes"
        expected_progress_bar = True
        expected_progress_bar_desc = "TEST-AGAIN"
        expected_allow_dask_on_strings = False
        expected_force_parallel = False
        set_defaults(
            npartitions=set_npartitions,
            dask_threshold=set_dask_threshold,
            scheduler=set_scheduler,
            progress_bar=set_progress_bar,
            progress_bar_desc=set_progress_bar_desc,
            allow_dask_on_strings=set_allow_dask_on_strings,
            force_parallel=set_force_parallel,
        )
        for swifter_df_1, swifter_df_2 in [
            [pd.DataFrame().swifter, pd.Series().swifter],
            [
                pd.Series().swifter,
                pd.DataFrame(
                    {"x": np.arange(0, 10)},
                    index=pd.date_range("2019-01-1", "2020-01-1", periods=10),
                ).swifter.rolling("1d"),
            ],
            [
                pd.DataFrame(
                    {"x": np.arange(0, 10)},
                    index=pd.date_range("2019-01-1", "2020-01-1", periods=10),
                ).swifter.rolling("1d"),
                pd.DataFrame(
                    {"x": np.arange(0, 10)},
                    index=pd.date_range("2019-01-1", "2020-01-1", periods=10),
                ).swifter.resample("3T"),
            ],
            [
                pd.DataFrame(
                    {"x": np.arange(0, 10)},
                    index=pd.date_range("2019-01-1", "2020-01-1", periods=10),
                ).swifter.resample("3T"),
                pd.DataFrame().swifter,
            ],
        ]:
            swifter_df_1 = (
                swifter_df_1.set_npartitions(npartitions=expected_npartitions)
                .set_dask_threshold(dask_threshold=expected_dask_threshold)
                .set_dask_scheduler(scheduler=expected_scheduler)
                .progress_bar(enable=expected_progress_bar, desc=expected_progress_bar_desc)
                .allow_dask_on_strings(enable=expected_allow_dask_on_strings)
                .force_parallel(enable=expected_force_parallel)
            )

            swifter_df_1._npartitions == expected_npartitions
            swifter_df_1._dask_threshold == expected_dask_threshold
            swifter_df_1._scheduler == expected_scheduler
            swifter_df_1._progress_bar == expected_progress_bar
            swifter_df_1._progress_bar_desc == expected_progress_bar_desc
            swifter_df_1._allow_dask_on_strings == expected_allow_dask_on_strings
            swifter_df_1._force_parallel == expected_force_parallel

            swifter_df_2._npartitions == set_npartitions
            swifter_df_2._dask_threshold == set_dask_threshold
            swifter_df_2._scheduler == set_scheduler
            swifter_df_2._progress_bar == set_progress_bar
            swifter_df_2._progress_bar_desc == set_progress_bar_desc
            swifter_df_2._allow_dask_on_strings == set_allow_dask_on_strings
            swifter_df_2._force_parallel = set_force_parallel

    def test_set_npartitions(self):
        LOG.info("test_set_npartitions")
        for swifter_df, set_npartitions, expected in zip(
            [
                pd.DataFrame().swifter,
                pd.Series().swifter,
                pd.DataFrame(
                    {"x": np.arange(0, 10)},
                    index=pd.date_range("2019-01-1", "2020-01-1", periods=10),
                ).swifter.rolling("1d"),
                pd.DataFrame(
                    {"x": np.arange(0, 10)},
                    index=pd.date_range("2019-01-1", "2020-01-1", periods=10),
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

    def test_set_dask_threshold(self):
        LOG.info("test_set_dask_threshold")
        expected = 1000
        for swifter_df in [
            pd.DataFrame().swifter,
            pd.Series().swifter,
            pd.DataFrame(
                {"x": np.arange(0, 10)},
                index=pd.date_range("2019-01-1", "2020-01-1", periods=10),
            ).swifter.rolling("1d"),
            pd.DataFrame(
                {"x": np.arange(0, 10)},
                index=pd.date_range("2019-01-1", "2020-01-1", periods=10),
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
                {"x": np.arange(0, 10)},
                index=pd.date_range("2019-01-1", "2020-01-1", periods=10),
            ).swifter.rolling("1d"),
            pd.DataFrame(
                {"x": np.arange(0, 10)},
                index=pd.date_range("2019-01-1", "2020-01-1", periods=10),
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
                {"x": np.arange(0, 10)},
                index=pd.date_range("2019-01-1", "2020-01-1", periods=10),
            ).swifter.rolling("1d"),
            pd.DataFrame(
                {"x": np.arange(0, 10)},
                index=pd.date_range("2019-01-1", "2020-01-1", periods=10),
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

    def test_force_parallel(self):
        LOG.info("test_force_parallel")
        expected = True
        swifter_df = pd.DataFrame().swifter
        before = swifter_df._force_parallel
        swifter_df.force_parallel(expected)
        actual = swifter_df._force_parallel
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
                + "df.swifter.progress_bar(enable=False)"
                + ".apply(lambda x: print(x.values))",
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
        df = pd.DataFrame({"x": np.random.normal(size=10_000_000)})
        series = df["x"]

        tqdm.pandas(desc="Pandas Vec math apply ~ Series")
        start_pd = time.time()
        pd_val = series.progress_apply(math_vec_square)
        end_pd = time.time()
        pd_time = end_pd - start_pd

        start_swifter = time.time()
        swifter_val = (
            series.swifter.set_npartitions(4).progress_bar(desc="Vec math apply ~ Series").apply(math_vec_square)
        )
        end_swifter = time.time()
        swifter_time = end_swifter - start_swifter

        self.assertEqual(pd_val, swifter_val)  # equality test
        if self.ncores > 1:  # speed test
            self.assertLessLinux(swifter_time, pd_time)

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
            self.assertLessLinux(swifter_time, pd_time)

    def test_vectorized_force_parallel_math_apply_on_large_series(self):
        LOG.info("test_vectorized_force_parallel_math_apply_on_large_series")
        df = pd.DataFrame({"x": np.random.normal(size=2_000_000)})
        series = df["x"]

        tqdm.pandas(desc="Pandas Vec math apply ~ Series")
        start_pd = time.time()
        pd_val = series.progress_apply(math_vec_square)
        end_pd = time.time()
        pd_time = end_pd - start_pd

        start_swifter = time.time()
        swifter_val = (
            series.swifter.set_npartitions(4)
            .force_parallel(True)
            .progress_bar(desc="Force Parallel - Vec math apply ~ Series")
            .apply(math_vec_square)
        )
        end_swifter = time.time()
        swifter_time = end_swifter - start_swifter

        self.assertEqual(pd_val, swifter_val)  # equality test
        if self.ncores > 1:  # speed test
            self.assertLessLinux(swifter_time, pd_time)


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

    @run_if_ray_installed
    def test_groupby_apply_on_empty_dataframe(self):
        LOG.info("test_groupby_apply_on_empty_dataframe")
        df = pd.DataFrame(columns=["x", "y"])
        pd_val = df.groupby("x").apply(math_vec_square)
        swifter_val = df.swifter.groupby("x").apply(math_vec_square)
        self.assertEqual(pd_val, swifter_val)  # equality test

    @run_if_ray_installed
    def test_groupby_index_apply(self):
        LOG.info("test_groupby_index_apply")
        SIZE = GROUPBY_MAX_ROWS_PANDAS_DEFAULT * 2
        df = pd.DataFrame(
            {
                "x": np.random.normal(size=SIZE),
                "y": np.random.uniform(size=SIZE),
                "g": np.random.choice(np.arange(100), size=SIZE),
            }
        )
        pd_val = df.groupby("g")["x"].apply(lambda x: x.std())
        swifter_val = df.swifter.groupby("g")["x"].apply(lambda x: x.std())
        self.assertEqual(pd_val, swifter_val)

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
        df = pd.DataFrame(
            {
                "x": np.random.normal(size=1_000_000),
                "y": np.random.uniform(size=1_000_000),
            }
        )

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
            self.assertLessLinux(swifter_time, pd_time)

    def test_nonvectorized_math_apply_on_large_dataframe_broadcast(self):
        LOG.info("test_nonvectorized_math_apply_on_large_dataframe_broadcast")
        df = pd.DataFrame({"x": np.random.normal(size=500_000), "y": np.random.uniform(size=500_000)})

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
            self.assertLessLinux(swifter_time, pd_time)

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
            self.assertLessLinux(swifter_time, pd_time)

    def test_nonvectorized_text_dask_apply_on_large_dataframe(self):
        LOG.info("test_nonvectorized_text_dask_apply_on_large_dataframe")
        df = pd.DataFrame(
            {
                "letter": ["A", "B", "C", "D", "E"] * 200_000,
                "value": np.random.normal(size=1_000_000),
            }
        )

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
            self.assertLessLinux(swifter_time, pd_time)

    def test_vectorized_force_parallel_math_apply_on_large_dataframe(self):
        LOG.info("test_vectorized_force_parallel_math_apply_on_large_dataframe")
        df = pd.DataFrame(
            {
                "x": np.random.normal(size=1_000_000),
                "y": np.random.uniform(size=1_000_000),
            }
        )

        tqdm.pandas(desc="Pandas Nonvec math apply ~ DF")
        start_pd = time.time()
        pd_val = df.progress_apply(math_vec_multiply, axis=1)
        end_pd = time.time()
        pd_time = end_pd - start_pd

        start_swifter = time.time()
        swifter_val = (
            df.swifter.set_npartitions(4)
            .force_parallel(True)
            .progress_bar(desc="Forced Parallel - Vec math apply ~ DF")
            .apply(math_vec_multiply, axis=1)
        )
        end_swifter = time.time()
        swifter_time = end_swifter - start_swifter

        self.assertEqual(pd_val, swifter_val)  # equality test
        if self.ncores > 1:  # speed test
            self.assertLessLinux(swifter_time, pd_time)

    def test_vectorized_math_applymap_on_large_dataframe(self):
        LOG.info("test_vectorized_math_applymap_on_large_dataframe")
        df = pd.DataFrame(
            {
                "x": np.random.normal(size=2_000_000),
                "y": np.random.uniform(size=2_000_000),
            }
        )

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
            self.assertLessLinux(swifter_time, pd_time)

    def test_vectorized_force_parallel_math_applymap_on_large_dataframe(self):
        LOG.info("test_vectorized_force_parallel_math_applymap_on_large_dataframe")
        df = pd.DataFrame(
            {
                "x": np.random.normal(size=2_000_000),
                "y": np.random.uniform(size=2_000_000),
            }
        )

        tqdm.pandas(desc="Pandas Vec math applymap ~ DF")
        start_pd = time.time()
        pd_val = df.progress_applymap(math_vec_square)
        end_pd = time.time()
        pd_time = end_pd - start_pd

        start_swifter = time.time()
        swifter_val = (
            df.swifter.set_npartitions(4)
            .force_parallel(True)
            .progress_bar(desc="Force Parallel ~ Vec math applymap ~ DF")
            .applymap(math_vec_square)
        )
        end_swifter = time.time()
        swifter_time = end_swifter - start_swifter

        self.assertEqual(pd_val, swifter_val)  # equality test
        if self.ncores > 1:  # speed test
            self.assertLessLinux(swifter_time, pd_time)

    def test_nonvectorized_math_applymap_on_large_dataframe(self):
        LOG.info("test_nonvectorized_math_applymap_on_large_dataframe")
        df = pd.DataFrame(
            {
                "x": np.random.normal(size=5_000_000),
                "y": np.random.uniform(size=5_000_000),
            }
        )

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
            self.assertLessLinux(swifter_time, pd_time)

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

    @run_if_ray_installed
    def test_vectorized_math_groupby_apply_on_small_dataframe(self):
        LOG.info("test_vectorized_math_groupby_apply_on_small_dataframe")
        df = pd.DataFrame(
            {
                "g": np.random.choice([0, 1, 2], size=500),
                "x": np.random.normal(size=500),
                "y": np.random.uniform(size=500),
            }
        )
        pd_val = df.groupby("g").apply(numeric_func)
        swifter_val = df.swifter.groupby("g").apply(numeric_func)
        self.assertSeriesEqual(pd_val, swifter_val, "Swifter output does not equal Pandas output")  # equality test

    @run_if_ray_installed
    def test_vectorized_force_parallel_math_groupby_apply_on_small_dataframe(self):
        LOG.info("test_vectorized_force_parallel_math_groupby_apply_on_small_dataframe")
        df = pd.DataFrame(
            {
                "g": np.random.choice([0, 1, 2], size=500),
                "x": np.random.normal(size=500),
                "y": np.random.uniform(size=500),
            }
        )
        pd_val = df.groupby("g").apply(numeric_func)
        swifter_val = df.swifter.force_parallel(True).groupby("g").apply(numeric_func)
        self.assertSeriesEqual(pd_val, swifter_val, "Swifter output does not equal Pandas output")  # equality test

    @run_if_ray_installed
    def test_vectorized_math_groupby_apply_on_large_dataframe(self):
        LOG.info("test_vectorized_math_groupby_apply_on_large_dataframe")
        df = pd.DataFrame(
            {
                "g": np.random.choice(np.arange(50000), size=500000),
                "x": np.random.normal(size=500000),
                "y": np.random.uniform(size=500000),
            }
        )
        pd_val = df.groupby("g").apply(numeric_func)
        swifter_val = df.swifter.groupby("g").apply(numeric_func)
        self.assertSeriesEqual(pd_val, swifter_val, "Swifter output does not equal Pandas output")  # equality test

    @run_if_ray_installed
    def test_vectorized_math_groupby_apply_on_large_dataframe_index(self):
        LOG.info("test_vectorized_math_groupby_apply_on_large_dataframe_index")
        df = pd.DataFrame(
            {
                "x": np.random.normal(size=500000),
                "y": np.random.uniform(size=500000),
            },
            index=np.random.choice(np.arange(50000), size=500000),
        )
        pd_val = df.groupby(df.index).apply(numeric_func)
        swifter_val = df.swifter.groupby(df.index).apply(numeric_func)
        self.assertSeriesEqual(pd_val, swifter_val, "Swifter output does not equal Pandas output")  # equality test

    @run_if_ray_installed
    def test_vectorized_force_parallel_math_groupby_apply_on_large_dataframe(self):
        LOG.info("test_vectorized_force_parallel_math_groupby_apply_on_large_dataframe")
        df = pd.DataFrame(
            {
                "g": np.random.choice(np.arange(50000), size=500000),
                "x": np.random.normal(size=500000),
                "y": np.random.uniform(size=500000),
            }
        )
        pd_val = df.groupby("g").apply(numeric_func)
        swifter_val = df.swifter.force_parallel(True).groupby("g").apply(numeric_func)
        self.assertSeriesEqual(pd_val, swifter_val, "Swifter output does not equal Pandas output")  # equality test

    @run_if_ray_installed
    def test_vectorized_text_groupby_apply_on_small_dataframe(self):
        LOG.info("test_vectorized_text_groupby_apply_on_small_dataframe")
        df = pd.DataFrame(
            {"g": np.random.choice([0, 1, 2], size=500), "text": np.random.choice(["A", "B", "C"], size=500)}
        )
        pd_val = df.groupby("g").apply(clean_text_foo)
        swifter_val = df.swifter.groupby("g").apply(clean_text_foo)
        self.assertSeriesEqual(pd_val, swifter_val, "Swifter output does not equal Pandas output")  # equality test

    @run_if_ray_installed
    def test_vectorized_force_parallel_text_groupby_apply_on_small_dataframe(self):
        LOG.info("test_vectorized_force_parallel_text_groupby_apply_on_small_dataframe")
        df = pd.DataFrame(
            {"g": np.random.choice([0, 1, 2], size=500), "text": np.random.choice(["A", "B", "C"], size=500)}
        )
        pd_val = df.groupby("g").apply(clean_text_foo)
        swifter_val = df.swifter.force_parallel(True).groupby("g").apply(clean_text_foo)
        self.assertSeriesEqual(pd_val, swifter_val, "Swifter output does not equal Pandas output")  # equality test

    @run_if_ray_installed
    def test_vectorized_text_groupby_apply_on_large_dataframe(self):
        LOG.info("test_vectorized_text_groupby_apply_on_large_dataframe")
        df = pd.DataFrame(
            {
                "g": np.random.choice(np.arange(50000), size=500000),
                "text": np.random.choice(["A", "B", "C"], size=500000),
            }
        )
        pd_val = df.groupby("g").apply(clean_text_foo)
        swifter_val = df.swifter.groupby("g").apply(clean_text_foo)
        self.assertSeriesEqual(pd_val, swifter_val, "Swifter output does not equal Pandas output")  # equality test

    @run_if_ray_installed
    def test_vectorized_force_parallel_text_groupby_apply_on_large_dataframe(self):
        LOG.info("test_vectorized_force_parallel_text_groupby_apply_on_large_dataframe")
        df = pd.DataFrame(
            {
                "g": np.random.choice(np.arange(50000), size=500000),
                "text": np.random.choice(["A", "B", "C"], size=500000),
            }
        )
        pd_val = df.groupby("g").apply(clean_text_foo)
        swifter_val = df.swifter.force_parallel(True).groupby("g").apply(clean_text_foo)
        self.assertSeriesEqual(pd_val, swifter_val, "Swifter output does not equal Pandas output")  # equality test


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
        df = pd.DataFrame(
            {"x": np.arange(0, 1000)},
            index=pd.date_range("2019-01-1", "2020-01-1", periods=1000),
        )
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
        df = pd.DataFrame(
            {"x": np.arange(0, 1000)},
            index=pd.date_range("2019-01-1", "2020-01-1", periods=1000),
        )
        pd_val = df.rolling("1d").apply(math_agg_foo, raw=True)
        swifter_val = (
            df.swifter.set_npartitions(4).rolling("1d").progress_bar(enable=False).apply(math_agg_foo, raw=True)
        )
        self.assertEqual(pd_val, swifter_val)  # equality test

    def test_vectorized_math_apply_on_large_rolling_dataframe(self):
        LOG.info("test_vectorized_math_apply_on_large_rolling_dataframe")
        df = pd.DataFrame(
            {"x": np.arange(0, 1_000_000)},
            index=pd.date_range("2019-01-1", "2020-01-1", periods=1_000_000),
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
            {"x": np.arange(0, 10_000_000)},
            index=pd.date_range("2019-01-1", "2020-01-1", periods=10_000_000),
        )

        start_pd = time.time()
        pd_val = df.rolling("3T").apply(math_agg_foo, raw=True)
        end_pd = time.time()
        pd_time = end_pd - start_pd

        start_swifter = time.time()
        swifter_val = (
            df.swifter.set_npartitions(10)
            .rolling("3T")
            .progress_bar(desc="Nonvec math apply ~ Rolling DF")
            .apply(math_agg_foo, raw=True)
        )
        end_swifter = time.time()
        swifter_time = end_swifter - start_swifter

        self.assertEqual(pd_val, swifter_val)  # equality test
        if self.ncores > 1:  # speed test
            self.assertLessLinux(swifter_time, pd_time)

    def test_vectorized_force_parallel_math_apply_on_large_rolling_dataframe(self):
        LOG.info("test_vectorized_force_parallel_math_apply_on_large_rolling_dataframe")
        df = pd.DataFrame(
            {"x": np.arange(0, 1_000_000)},
            index=pd.date_range("2019-01-1", "2020-01-1", periods=1_000_000),
        )
        pd_val = df.rolling("1d").apply(max, raw=True)
        swifter_val = (
            df.swifter.set_npartitions(4)
            .force_parallel(True)
            .rolling("1d")
            .progress_bar(desc="Force Parallel ~ Vec math apply ~ Rolling DF")
            .apply(max, raw=True)
        )
        self.assertEqual(pd_val, swifter_val)  # equality test

    def test_nonvectorized_math_apply_on_small_resampler_dataframe(self):
        LOG.info("test_nonvectorized_math_apply_on_small_resampler_dataframe")
        df = pd.DataFrame(
            {"x": np.arange(0, 1000)},
            index=pd.date_range("2019-01-1", "2020-01-1", periods=1000),
        )
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
            {"x": np.arange(0, 1_000_000)},
            index=pd.date_range("2019-01-1", "2020-01-1", periods=1_000_000),
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
            self.assertLessLinux(swifter_time, pd_time)

    def test_nonvectorized_force_parallel_math_apply_on_large_resampler_dataframe(self):
        LOG.info("test_nonvectorized_force_parallel_math_apply_on_large_resampler_dataframe")
        df = pd.DataFrame(
            {"x": np.arange(0, 1_000_000)},
            index=pd.date_range("2019-01-1", "2020-01-1", periods=1_000_000),
        )

        start_pd = time.time()
        pd_val = df.resample("3T").apply(math_agg_foo)
        end_pd = time.time()
        pd_time = end_pd - start_pd

        start_swifter = time.time()
        swifter_val = (
            df.swifter.set_npartitions(4)
            .force_parallel(True)
            .resample("3T")
            .progress_bar(desc="Force Parallel ~ Nonvec math apply ~ Resample DF")
            .apply(math_agg_foo)
        )
        end_swifter = time.time()
        swifter_time = end_swifter - start_swifter

        self.assertEqual(pd_val, swifter_val)  # equality test
        if self.ncores > 1:  # speed test
            self.assertLessLinux(swifter_time, pd_time)


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

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            series.swifter.force_parallel(False)
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
        df = md.Series(np.random.uniform(size=10_000_000), name="x")

        md_val = df.apply(math_vec_square, axis=0)
        md_pd_val = md_val._to_pandas()  # We have to bring it into pandas to confirm swifter apply speed is quicker

        swifter_val = df.swifter.set_npartitions(4).apply(math_vec_square)
        swifter_pd_val = (
            swifter_val._to_pandas()
        )  # We have to bring it into pandas to confirm swifter apply speed is quicker

        self.assertEqual(md_val, swifter_val)  # equality test
        self.assertEqual(md_pd_val, swifter_pd_val)  # equality test after converting to pandas


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

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            df.swifter.force_parallel(False)
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
        df = md.DataFrame(
            {
                "letter": ["A", "B", "C", "D", "E"] * 200_000,
                "value": np.random.normal(size=1_000_000),
            }
        )
        md_val = df.apply(text_foo, axis=1)
        swifter_val = df.swifter.set_npartitions(4).apply(text_foo, axis=1)
        self.assertEqual(md_val, swifter_val)  # equality test

    def test_vectorized_modin_apply_on_large_dataframe(self):
        LOG.info("test_vectorized_modin_apply_on_large_dataframe")
        md = self.modinSetUp()
        df = md.DataFrame(
            {
                "x": np.random.normal(size=1_000_000),
                "y": np.random.uniform(size=1_000_000),
            }
        )
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
        self.assertLessLinux(swifter_time, md_time)  # speed test
