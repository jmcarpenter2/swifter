import sys
import unittest
import subprocess
import time
import logging

import numpy as np
import pandas as pd
import swifter

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


class TestTransformation(unittest.TestCase):
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

    def test_rolling_apply_on_empty_dataframe(self):
        LOG.info("test_rolling_apply_on_empty_dataframe")
        df = pd.DataFrame(columns=["x", "y"])
        pd_val = df.rolling(1).apply(math_agg_foo, raw=True)
        swifter_val = df.swifter.rolling(1).apply(math_agg_foo, raw=True)
        self.assertEqual(pd_val, swifter_val)  # equality test

    def test_resample_apply_on_empty_dataframe(self):
        LOG.info("test_resample_apply_on_empty_dataframe")
        df = pd.DataFrame(columns=["x", "y"], index=pd.date_range(start="2020/01/01", periods=0))
        pd_val = df.resample("1d").apply(math_agg_foo)
        swifter_val = df.swifter.resample("1d").apply(math_agg_foo)
        self.assertEqual(pd_val, swifter_val)  # equality test

    def test_nonvectorized_math_apply_on_small_rolling_dataframe(self):
        LOG.info("test_nonvectorized_math_apply_on_small_rolling_dataframe")
        df = pd.DataFrame({"x": np.arange(0, 1000)}, index=pd.date_range("2019-01-1", "2020-01-1", periods=1000))
        pd_val = df.rolling("1d").apply(math_agg_foo, raw=True)
        swifter_val = (
            df.swifter.rolling("1d").progress_bar(desc="Nonvec math apply ~ Rolling DF").apply(math_agg_foo, raw=True)
        )
        self.assertEqual(pd_val, swifter_val)  # equality test

    def test_nonvectorized_math_apply_on_small_rolling_dataframe_no_progress_bar(self):
        LOG.info("test_nonvectorized_math_apply_on_small_rolling_dataframe_no_progress_bar")
        df = pd.DataFrame({"x": np.arange(0, 1000)}, index=pd.date_range("2019-01-1", "2020-01-1", periods=1000))
        pd_val = df.rolling("1d").apply(math_agg_foo, raw=True)
        swifter_val = df.swifter.rolling("1d").progress_bar(enable=False).apply(math_agg_foo, raw=True)
        self.assertEqual(pd_val, swifter_val)  # equality test

    def test_vectorized_math_apply_on_large_rolling_dataframe(self):
        LOG.info("test_vectorized_math_apply_on_large_rolling_dataframe")
        df = pd.DataFrame(
            {"x": np.arange(0, 1_000_000)}, index=pd.date_range("2019-01-1", "2020-01-1", periods=1_000_000)
        )

        start_pd = time.time()
        pd_val = df.rolling("1d").apply(max, raw=True)
        end_pd = time.time()
        pd_time = end_pd - start_pd

        start_swifter = time.time()
        swifter_val = df.swifter.rolling("1d").progress_bar(desc="Vec math apply ~ Rolling DF").apply(max, raw=True)
        end_swifter = time.time()
        swifter_time = end_swifter - start_swifter

        self.assertEqual(pd_val, swifter_val)  # equality test
        if self.ncores > 1:  # speed test
            self.assertLess(swifter_time, pd_time)

    def test_nonvectorized_math_apply_on_large_rolling_dataframe(self):
        LOG.info("test_nonvectorized_math_apply_on_large_rolling_dataframe")
        df = pd.DataFrame(
            {"x": np.arange(0, 3_000_000)}, index=pd.date_range("2019-01-1", "2020-01-1", periods=3_000_000)
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

        self.assertEqual(pd_val, swifter_val)  # equality test
        if self.ncores > 1:  # speed test
            self.assertLess(swifter_time, pd_time)

    def test_nonvectorized_math_apply_on_small_resampler_dataframe(self):
        LOG.info("test_nonvectorized_math_apply_on_small_resampler_dataframe")
        df = pd.DataFrame({"x": np.arange(0, 1000)}, index=pd.date_range("2019-01-1", "2020-01-1", periods=1000))
        pd_val = df.resample("1M").apply(math_agg_foo)
        swifter_val = df.swifter.resample("1M").progress_bar(desc="Nonvec math apply ~ Resample DF").apply(math_agg_foo)
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
        swifter_val = df.swifter.resample("3T").progress_bar(desc="Nonvec math apply ~ Resample DF").apply(math_agg_foo)
        end_swifter = time.time()
        swifter_time = end_swifter - start_swifter

        self.assertEqual(pd_val, swifter_val)  # equality test
        if self.ncores > 1:  # speed test
            self.assertLess(swifter_time, pd_time)
