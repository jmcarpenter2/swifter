import unittest
import time

import numpy as np
import pandas as pd
import swifter

print(f"Version {swifter.__version__}")


def math_vec_square(x):
    return x ** 2


def math_foo(x, compare_to=1):
    return x ** 2 if x < compare_to else x ** (1 / 2)


def math_vec_multiply(row):
    return row["x"] * row["y"]


def math_agg_foo(row):
    return row.sum() - row.min()


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
        expected = 1000
        for swifter_df in [
            pd.DataFrame().swifter,
            pd.Series().swifter,
            pd.DataFrame(
                {"x": np.arange(0, 10)}, index=pd.date_range("2019-01-1", "2020-01-1", periods=10)
            ).swifter.rolling("1d"),
        ]:
            before = swifter_df._npartitions
            swifter_df.set_npartitions(expected)
            actual = swifter_df._npartitions
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

    def test_vectorized_math_apply_on_large_series(self):
        df = pd.DataFrame({"x": np.random.normal(size=1_000_000)})
        series = df["x"]

        start_pd = time.time()
        pd_val = series.apply(math_vec_square)
        end_pd = time.time()
        pd_time = end_pd - start_pd

        start_swifter = time.time()
        swifter_val = series.swifter.apply(math_vec_square)
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
        swifter_val = series.swifter.apply(math_foo, compare_to=1)
        end_swifter = time.time()
        swifter_time = end_swifter - start_swifter

        self.assertEqual(pd_val, swifter_val)
        self.assertLess(swifter_time, pd_time)

    def test_vectorized_math_apply_on_large_dataframe(self):
        df = pd.DataFrame({"x": np.random.normal(size=1_000_000), "y": np.random.uniform(size=1_000_000)})

        start_pd = time.time()
        pd_val = df.apply(math_vec_multiply, axis=1)
        end_pd = time.time()
        pd_time = end_pd - start_pd

        start_swifter = time.time()
        swifter_val = df.swifter.apply(math_vec_multiply, axis=1)
        end_swifter = time.time()
        swifter_time = end_swifter - start_swifter

        self.assertEqual(pd_val, swifter_val)
        self.assertLess(swifter_time, pd_time)

    def test_nonvectorized_math_apply_on_large_dataframe(self):
        df = pd.DataFrame({"x": np.random.normal(size=1_000_000), "y": np.random.uniform(size=1_000_000)})

        start_pd = time.time()
        pd_val = df.apply(math_agg_foo, axis=1)
        end_pd = time.time()
        pd_time = end_pd - start_pd

        start_swifter = time.time()
        swifter_val = df.swifter.apply(math_agg_foo, axis=1)
        end_swifter = time.time()
        swifter_time = end_swifter - start_swifter

        self.assertEqual(pd_val, swifter_val)
        self.assertLess(swifter_time, pd_time)

    def test_vectorized_math_apply_on_large_rolling_dataframe(self):
        df = pd.DataFrame(
            {"x": np.arange(0, 1_000_000)}, index=pd.date_range("2019-01-1", "2020-01-1", periods=1_000_000)
        )

        start_pd = time.time()
        pd_val = df.rolling("1d").apply(sum)
        end_pd = time.time()
        pd_time = end_pd - start_pd

        start_swifter = time.time()
        swifter_val = df.swifter.rolling("1d").apply(sum)
        end_swifter = time.time()
        swifter_time = end_swifter - start_swifter

        self.assertEqual(pd_val, swifter_val)
        self.assertLess(swifter_time, pd_time)

    def test_nonvectorized_math_apply_on_large_rolling_dataframe(self):
        df = pd.DataFrame(
            {"x": np.arange(0, 1_000_000)}, index=pd.date_range("2019-01-1", "2020-01-1", periods=1_000_000)
        )

        start_pd = time.time()
        pd_val = df.rolling("1d").apply(math_agg_foo)
        end_pd = time.time()
        pd_time = end_pd - start_pd

        start_swifter = time.time()
        swifter_val = df.swifter.rolling("1d").apply(math_agg_foo)
        end_swifter = time.time()
        swifter_time = end_swifter - start_swifter

        self.assertEqual(pd_val, swifter_val)
        self.assertLess(swifter_time, pd_time)
