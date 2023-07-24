from setuptools import setup

setup(
    name="swifter",
    packages=["swifter"],  # this must be the same as the name above
    version="1.4.0",
    description="A package which efficiently applies any function to a pandas dataframe or series in the fastest available manner",
    author="Jason Carpenter",
    author_email="jcarpenter@manifold.ai",
    url="https://github.com/jmcarpenter2/swifter",  # use the URL to the github repo
    download_url="https://github.com/jmcarpenter2/swifter/archive/1.4.0.tar.gz",
    keywords=["pandas", "dask", "apply", "function", "parallelize", "vectorize"],
    install_requires=[
        "pandas>=1.0.0",
        "psutil>=5.6.6",
        "dask[dataframe]>=2.10.0",
        "tqdm>=4.33.0",
    ],
    extras_require={
        "groupby": ["ray>=1.0.0"],
        "notebook": ["ipywidgets>=7.0.0"],
    },
    classifiers=[],
)
