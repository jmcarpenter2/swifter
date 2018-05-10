from distutils.core import setup
setup(
  name = 'swifter',
  packages = ['swifter'], # this must be the same as the name above
  version = '0.153',
  description = 'A package which efficiently applies any function to a pandas dataframe or series in the fastest available manner',
  author = 'Jason Carpenter',
  author_email = 'jmcarpenter2@dons.usfca.edu',
  url = 'https://github.com/jmcarpenter2/swifter', # use the URL to the github repo
  download_url = 'https://github.com/jmcarpenter2/swifter/archive/0.152.tar.gz',
  keywords = ['pandas', 'apply', 'function', 'parallelize', 'vectorize'],
  install_requires=[
        'pandas',
        'psutil',
        'dask'
  ],
  classifiers = [],
)
