import sys
import unittest
import subprocess
import time
import logging

import numpy as np
import pandas as pd
import swifter

from psutil import cpu_count

logging.getLogger(__name__)
logging.info(f"Version {swifter.__version__}")


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

    def setUp(self):
        self.addTypeEqualityFunc(pd.Series, self.assertSeriesEqual)
        self.addTypeEqualityFunc(pd.DataFrame, self.assertDataFrameEqual)

    def test_set_npartitions(self):
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

    def test_set_dask_threshold(self):
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
        expected = True
        swifter_df = pd.DataFrame().swifter
        before = swifter_df._allow_dask_on_strings
        swifter_df.allow_dask_on_strings(expected)
        actual = swifter_df._allow_dask_on_strings
        self.assertEqual(actual, expected)
        self.assertNotEqual(before, actual)

    def test_stdout_redirected(self):
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

    def test_apply_on_empty_series(self):
        series = pd.Series()
        pd_val = series.apply(math_foo, compare_to=1)
        swifter_val = series.swifter.apply(math_foo, compare_to=1)
        self.assertEqual(pd_val, swifter_val)

    def test_apply_on_empty_dataframe(self):
        df = pd.DataFrame(columns=["x", "y"])
        pd_val = df.apply(math_vec_multiply, axis=1)
        swifter_val = df.swifter.apply(math_vec_multiply, axis=1)
        self.assertEqual(pd_val, swifter_val)

    def test_applymap_on_empty_dataframe(self):
        df = pd.DataFrame(columns=["x", "y"])
        pd_val = df.applymap(math_vec_square)
        swifter_val = df.swifter.applymap(math_vec_square)
        self.assertEqual(pd_val, swifter_val)

    def test_rolling_apply_on_empty_dataframe(self):
        df = pd.DataFrame(columns=["x", "y"])
        pd_val = df.rolling(1).apply(math_agg_foo, raw=True)
        swifter_val = df.swifter.rolling(1).apply(math_agg_foo, raw=True)
        self.assertEqual(pd_val, swifter_val)

    def test_resample_apply_on_empty_dataframe(self):
        df = pd.DataFrame(columns=["x", "y"], index=pd.date_range(start="2020/01/01", periods=0))
        pd_val = df.resample("1d").apply(math_agg_foo)
        swifter_val = df.swifter.resample("1d").apply(math_agg_foo)
        self.assertEqual(pd_val, swifter_val)

    def test_nonvectorized_math_apply_on_small_series(self):
        df = pd.DataFrame({"x": np.random.normal(size=1000)})
        series = df["x"]
        pd_val = series.apply(math_foo, compare_to=1)
        swifter_val = series.swifter.progress_bar(desc="Vec math apply ~ Series").apply(math_foo, compare_to=1)
        self.assertEqual(pd_val, swifter_val)

    def test_nonvectorized_math_apply_on_small_series_no_progress_bar(self):
        df = pd.DataFrame({"x": np.random.normal(size=1000)})
        series = df["x"]
        pd_val = series.apply(math_foo, compare_to=1)
        swifter_val = series.swifter.progress_bar(enable=False).apply(math_foo, compare_to=1)
        self.assertEqual(pd_val, swifter_val)

    def test_vectorized_math_apply_on_large_series(self):
        df = pd.DataFrame({"x": np.random.normal(size=1_000_000)})
        series = df["x"]

        start_pd = time.time()
        pd_val = series.apply(math_vec_square)
        end_pd = time.time()
        pd_time = end_pd - start_pd

        start_swifter = time.time()
        swifter_val = series.swifter.progress_bar(desc="Vec math apply ~ Series").apply(math_vec_square, axis=0)
        end_swifter = time.time()
        swifter_time = end_swifter - start_swifter

        self.assertEqual(pd_val, swifter_val)
        self.assertLess(swifter_time, pd_time)

    def test_nonvectorized_math_apply_on_large_series(self):
        df = pd.DataFrame({"x": np.random.normal(size=5_000_000)})
        series = df["x"]

        start_pd = time.time()
        pd_val = series.apply(math_foo, compare_to=1)
        end_pd = time.time()
        pd_time = end_pd - start_pd

        start_swifter = time.time()
        swifter_val = series.swifter.progress_bar(desc="Nonvec math apply ~ Series").apply(math_foo, compare_to=1)
        end_swifter = time.time()
        swifter_time = end_swifter - start_swifter

        self.assertEqual(pd_val, swifter_val)
        self.assertLess(swifter_time, pd_time)

    def test_nonvectorized_math_apply_on_small_dataframe(self):
        df = pd.DataFrame({"x": np.random.normal(size=1000), "y": np.random.uniform(size=1000)})
        pd_val = df.apply(math_agg_foo)
        swifter_val = df.swifter.progress_bar(desc="Vec math apply ~ DF").apply(math_agg_foo)
        self.assertEqual(pd_val, swifter_val)

    def test_nonvectorized_math_apply_on_small_dataframe_no_progress_bar(self):
        df = pd.DataFrame({"x": np.random.normal(size=1000), "y": np.random.uniform(size=1000)})
        pd_val = df.apply(math_agg_foo)
        swifter_val = df.swifter.progress_bar(enable=False).apply(math_agg_foo)
        self.assertEqual(pd_val, swifter_val)

    def test_vectorized_math_apply_on_large_dataframe(self):
        df = pd.DataFrame({"x": np.random.normal(size=1_000_000), "y": np.random.uniform(size=1_000_000)})

        start_pd = time.time()
        pd_val = df.apply(math_vec_multiply, axis=1)
        end_pd = time.time()
        pd_time = end_pd - start_pd

        start_swifter = time.time()
        swifter_val = df.swifter.progress_bar(desc="Vec math apply ~ DF").apply(math_vec_multiply, axis=1)
        end_swifter = time.time()
        swifter_time = end_swifter - start_swifter

        self.assertEqual(pd_val, swifter_val)
        self.assertLess(swifter_time, pd_time)

    def test_nonvectorized_math_apply_on_large_dataframe_broadcast(self):
        df = pd.DataFrame({"x": np.random.normal(size=1_000_000), "y": np.random.uniform(size=1_000_000)})

        start_pd = time.time()
        pd_val = df.apply(math_agg_foo, axis=1, result_type="broadcast")
        end_pd = time.time()
        pd_time = end_pd - start_pd

        start_swifter = time.time()
        swifter_val = df.swifter.progress_bar(desc="Nonvec math apply + broadcast ~ DF").apply(
            math_agg_foo, axis=1, result_type="broadcast"
        )
        end_swifter = time.time()
        swifter_time = end_swifter - start_swifter

        self.assertEqual(pd_val, swifter_val)
        self.assertLess(swifter_time, pd_time)

    def test_nonvectorized_math_apply_on_large_dataframe_reduce(self):
        df = pd.DataFrame({"x": np.random.normal(size=1_000_000), "y": np.random.uniform(size=1_000_000)})

        start_pd = time.time()
        pd_val = df.apply(math_agg_foo, axis=1, result_type="reduce")
        end_pd = time.time()
        pd_time = end_pd - start_pd

        start_swifter = time.time()
        swifter_val = df.swifter.progress_bar(desc="Nonvec math apply + reduce ~ DF").apply(
            math_agg_foo, axis=1, result_type="reduce"
        )
        end_swifter = time.time()
        swifter_time = end_swifter - start_swifter

        self.assertEqual(pd_val, swifter_val)
        self.assertLess(swifter_time, pd_time)

    def test_nonvectorized_text_apply_on_large_dataframe(self):
        df = pd.DataFrame({"letter": ["A", "B", "C", "D", "E"] * 200_000, "value": np.random.normal(size=1_000_000)})

        start_pd = time.time()
        pd_val = df.apply(text_foo, axis=1)
        end_pd = time.time()
        pd_time = end_pd - start_pd

        start_swifter = time.time()
        swifter_val = (
            df.swifter.allow_dask_on_strings(True).progress_bar(desc="Nonvec text apply ~ DF").apply(text_foo, axis=1)
        )
        end_swifter = time.time()
        swifter_time = end_swifter - start_swifter

        self.assertEqual(pd_val, swifter_val)
        self.assertLess(swifter_time, pd_time)

    def test_nonvectorized_math_apply_on_small_rolling_dataframe(self):
        df = pd.DataFrame({"x": np.arange(0, 1000)}, index=pd.date_range("2019-01-1", "2020-01-1", periods=1000))
        pd_val = df.rolling("3T").apply(math_agg_foo, raw=True)
        swifter_val = (
            df.swifter.rolling("3T").progress_bar(desc="Nonvec math apply ~ Rolling DF").apply(math_agg_foo, raw=True)
        )
        self.assertEqual(pd_val, swifter_val)

    def test_nonvectorized_math_apply_on_small_rolling_dataframe_no_progress_bar(self):
        df = pd.DataFrame({"x": np.arange(0, 1000)}, index=pd.date_range("2019-01-1", "2020-01-1", periods=1000))
        pd_val = df.rolling("3T").apply(math_agg_foo, raw=True)
        swifter_val = df.swifter.rolling("3T").progress_bar(enable=False).apply(math_agg_foo, raw=True)
        self.assertEqual(pd_val, swifter_val)

    def test_vectorized_math_apply_on_large_rolling_dataframe(self):
        df = pd.DataFrame(
            {"x": np.arange(0, 1_500_000)}, index=pd.date_range("2019-01-1", "2020-01-1", periods=1_500_000)
        )

        start_pd = time.time()
        pd_val = df.rolling("3T").apply(max, raw=True)
        end_pd = time.time()
        pd_time = end_pd - start_pd

        start_swifter = time.time()
        swifter_val = df.swifter.rolling("3T").progress_bar(desc="Vec math apply ~ Rolling DF").apply(max, raw=True)
        end_swifter = time.time()
        swifter_time = end_swifter - start_swifter

        self.assertEqual(pd_val, swifter_val)
        self.assertLess(swifter_time, pd_time)

    def test_nonvectorized_math_apply_on_large_rolling_dataframe(self):
        df = pd.DataFrame(
            {"x": np.arange(0, 1_000_000)}, index=pd.date_range("2019-01-1", "2020-01-1", periods=1_000_000)
        )

        start_pd = time.time()
        pd_val = df.rolling("3T").apply(math_agg_foo, raw=True)
        end_pd = time.time()
        pd_time = end_pd - start_pd

        start_swifter = time.time()
        swifter_val = (
            df.swifter.rolling("3T").progress_bar(desc="Nonvec math apply ~ Rolling DF").apply(math_agg_foo, raw=True)
        )
        end_swifter = time.time()
        swifter_time = end_swifter - start_swifter

        self.assertEqual(pd_val, swifter_val)
        self.assertLess(swifter_time, pd_time)

    def test_nonvectorized_math_apply_on_small_resampler_dataframe(self):
        df = pd.DataFrame({"x": np.arange(0, 1000)}, index=pd.date_range("2019-01-1", "2020-01-1", periods=1000))
        pd_val = df.resample("3T").apply(math_agg_foo)
        swifter_val = df.swifter.resample("3T").progress_bar(desc="Nonvec math apply ~ Resample DF").apply(math_agg_foo)
        self.assertEqual(pd_val, swifter_val)

    def test_nonvectorized_math_apply_on_large_resampler_dataframe(self):
        df = pd.DataFrame(
            {"x": np.arange(0, 1_000_000)}, index=pd.date_range("2019-01-1", "2020-01-1", periods=1_000_000)
        )

        start_pd = time.time()
        pd_val = df.resample("3T").apply(math_agg_foo)
        end_pd = time.time()
        pd_time = end_pd - start_pd

        start_swifter = time.time()
        swifter_val = df.swifter.resample("3T").progress_bar(desc="Nonvec math apply ~ Resample DF").apply(math_agg_foo)
        end_swifter = time.time()
        swifter_time = end_swifter - start_swifter

        self.assertEqual(pd_val, swifter_val)
        self.assertLess(swifter_time, pd_time)

    def test_vectorized_math_applymap_on_large_dataframe(self):
        df = pd.DataFrame({"x": np.random.normal(size=1_000_000), "y": np.random.uniform(size=1_000_000)})

        start_pd = time.time()
        pd_val = df.applymap(math_vec_square)
        end_pd = time.time()
        pd_time = end_pd - start_pd

        start_swifter = time.time()
        swifter_val = df.swifter.progress_bar(desc="Vec math applymap ~ DF").applymap(math_vec_square)
        end_swifter = time.time()
        swifter_time = end_swifter - start_swifter

        self.assertEqual(pd_val, swifter_val)
        self.assertLess(swifter_time, pd_time)

    def test_nonvectorized_math_applymap_on_large_dataframe(self):
        df = pd.DataFrame({"x": np.random.normal(size=1_000_000), "y": np.random.uniform(size=1_000_000)})

        start_pd = time.time()
        pd_val = df.applymap(math_foo)
        end_pd = time.time()
        pd_time = end_pd - start_pd

        start_swifter = time.time()
        swifter_val = df.swifter.progress_bar(desc="Nonvec math applymap ~ DF").applymap(math_foo)
        end_swifter = time.time()
        swifter_time = end_swifter - start_swifter

        self.assertEqual(pd_val, swifter_val)
        self.assertLess(swifter_time, pd_time)

    def test_nonvectorized_math_applymap_on_small_dataframe(self):
        df = pd.DataFrame({"x": np.random.normal(size=1000), "y": np.random.uniform(size=1000)})
        pd_val = df.applymap(math_foo)
        swifter_val = df.swifter.applymap(math_foo)
        self.assertEqual(pd_val, swifter_val)

    def test_nonvectorized_math_applymap_on_small_dataframe_no_progress_bar(self):
        df = pd.DataFrame({"x": np.random.normal(size=1000), "y": np.random.uniform(size=1000)})
        pd_val = df.applymap(math_foo)
        swifter_val = df.swifter.progress_bar(enable=False).applymap(math_foo)
        self.assertEqual(pd_val, swifter_val)
