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


class TestSeries(unittest.TestCase):
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
        swifter_val = series.swifter.progress_bar(desc="Vec math apply ~ Series").apply(math_vec_square, axis=0)
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
        swifter_val = series.swifter.progress_bar(desc="Nonvec math apply ~ Series").apply(math_foo, compare_to=1)
        end_swifter = time.time()
        swifter_time = end_swifter - start_swifter

        self.assertEqual(pd_val, swifter_val)  # equality test
        if self.ncores > 1:  # speed test
            self.assertLess(swifter_time, pd_time)
