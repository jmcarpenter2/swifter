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


class TestDataFrame(unittest.TestCase):
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
        swifter_val = df.swifter.progress_bar(desc="Vec math apply ~ DF").apply(math_vec_multiply, axis=1)
        end_swifter = time.time()
        swifter_time = end_swifter - start_swifter

        self.assertEqual(pd_val, swifter_val)  # equality test
        if self.ncores > 1:  # speed test
            self.assertLess(swifter_time, pd_time)

    def test_nonvectorized_math_apply_on_large_dataframe_broadcast(self):
        LOG.info("test_nonvectorized_math_apply_on_large_dataframe_broadcast")
        df = pd.DataFrame({"x": np.random.normal(size=1_000_000), "y": np.random.uniform(size=1_000_000)})

        tqdm.pandas(desc="Pandas Nonvec math apply + broadcast ~ DF")
        start_pd = time.time()
        pd_val = df.progress_apply(math_agg_foo, axis=1, result_type="broadcast")
        end_pd = time.time()
        pd_time = end_pd - start_pd

        start_swifter = time.time()
        swifter_val = df.swifter.progress_bar(desc="Nonvec math apply + broadcast ~ DF").apply(
            math_agg_foo, axis=1, result_type="broadcast"
        )
        end_swifter = time.time()
        swifter_time = end_swifter - start_swifter

        self.assertEqual(pd_val, swifter_val)  # equality test
        if self.ncores > 1:  # speed test
            self.assertLess(swifter_time, pd_time)

    def test_nonvectorized_math_apply_on_large_dataframe_reduce(self):
        LOG.info("test_nonvectorized_math_apply_on_large_dataframe_reduce")
        df = pd.DataFrame({"x": np.random.normal(size=1_000_000), "y": np.random.uniform(size=1_000_000)})

        tqdm.pandas(desc="Pandas Nonvec math apply + reduce ~ DF")
        start_pd = time.time()
        pd_val = df.progress_apply(math_agg_foo, axis=1, result_type="reduce")
        end_pd = time.time()
        pd_time = end_pd - start_pd

        start_swifter = time.time()
        swifter_val = df.swifter.progress_bar(desc="Nonvec math apply + reduce ~ DF").apply(
            math_agg_foo, axis=1, result_type="reduce"
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
            .progress_bar(desc="Nonvec Dask text apply ~ DF")
            .apply(text_foo, axis=1)
        )
        end_swifter = time.time()
        swifter_time = end_swifter - start_swifter

        self.assertEqual(pd_val, swifter_val)  # equality test
        if self.ncores > 1:  # speed test
            self.assertLess(swifter_time, pd_time)

    def test_nonvectorized_text_modin_apply_on_large_dataframe(self):
        LOG.info("test_nonvectorized_text_modin_apply_on_large_dataframe")
        df = pd.DataFrame({"letter": ["I", "You", "We"] * 100_000, "value": ["want to break free"] * 300_000})

        tqdm.pandas(desc="Pandas Nonvec text apply ~ DF")
        start_pd = time.time()
        pd_val = df.progress_apply(clean_text_foo, axis=1)
        end_pd = time.time()
        pd_time = end_pd - start_pd

        start_swifter = time.time()
        swifter_val = (
            df.swifter.allow_dask_on_strings(False)
            .progress_bar(desc="Nonvec Modin text apply ~ DF")
            .apply(clean_text_foo, axis=1)
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
        swifter_val = df.swifter.progress_bar(desc="Vec math applymap ~ DF").applymap(math_vec_square)
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
        swifter_val = df.swifter.progress_bar(desc="Nonvec math applymap ~ DF").applymap(math_foo)
        end_swifter = time.time()
        swifter_time = end_swifter - start_swifter

        self.assertEqual(pd_val, swifter_val)  # equality test
        if self.ncores > 1:  # speed test
            self.assertLess(swifter_time, pd_time)

    def test_nonvectorized_math_applymap_on_small_dataframe(self):
        LOG.info("test_nonvectorized_math_applymap_on_small_dataframe")
        df = pd.DataFrame({"x": np.random.normal(size=1000), "y": np.random.uniform(size=1000)})
        pd_val = df.applymap(math_foo)
        swifter_val = df.swifter.applymap(math_foo)
        self.assertEqual(pd_val, swifter_val)  # equality test

    def test_nonvectorized_math_applymap_on_small_dataframe_no_progress_bar(self):
        LOG.info("test_nonvectorized_math_applymap_on_small_dataframe_no_progress_bar")
        df = pd.DataFrame({"x": np.random.normal(size=1000), "y": np.random.uniform(size=1000)})
        pd_val = df.applymap(math_foo)
        swifter_val = df.swifter.progress_bar(enable=False).applymap(math_foo)
        self.assertEqual(pd_val, swifter_val)  # equality test
