from setuptools import find_packages, setup, Extension

with open("requirements.txt", "rb") as f:
    requirements = [req.strip() for req in f.readlines()]

setup(
    name="swifter",
    packages=["swifter"],  # this must be the same as the name above
    version="1.0.9",
    description="A package which efficiently applies any function to a pandas dataframe or series in the fastest available manner",
    author="Jason Carpenter",
    author_email="jcarpenter@manifold.ai",
    url="https://github.com/jmcarpenter2/swifter",  # use the URL to the github repo
    download_url="https://github.com/jmcarpenter2/swifter/archive/1.0.9.tar.gz",
    keywords=["pandas", "dask", "apply", "function", "parallelize", "vectorize"],
    install_requires=requirements,
    classifiers=[],
)
