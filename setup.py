from setuptools import setup

setup(
    name="swifter",
    packages=["swifter"],  # this must be the same as the name above
    version="1.1.4",
    description="A package which efficiently applies any function to a pandas dataframe or series in the fastest available manner",
    author="Jason Carpenter",
    author_email="jcarpenter@manifold.ai",
    url="https://github.com/jmcarpenter2/swifter",  # use the URL to the github repo
    download_url=f"https://github.com/jmcarpenter2/swifter/archive/1.1.4.tar.gz",
    keywords=["pandas", "dask", "apply", "function", "parallelize", "vectorize"],
    install_requires=[
        "pandas>=1.0.0",
        "psutil>=5.6.6",
        "dask[dataframe]>=2.10.0",
        "tqdm>=4.33.0",
        "ipywidgets>=7.0.0",
        "cloudpickle>=0.2.2",
        "parso>0.4.0",
        "bleach>=3.1.1",
    ],
    classifiers=[],
)
