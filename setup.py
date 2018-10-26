from distutils.core import setup
setup(
  name = 'swifter',
  packages = ['swifter'], # this must be the same as the name above
  version = '0.252',
  description = 'A package which efficiently applies any function to a pandas dataframe or series in the fastest available manner',
  author = 'Jason Carpenter',
  author_email = 'jmcarpenter2@dons.usfca.edu',
  url = 'https://github.com/jmcarpenter2/swifter', # use the URL to the github repo
  download_url = 'https://github.com/jmcarpenter2/swifter/archive/0.252.tar.gz',
  keywords = ['pandas', 'dask', 'apply', 'function', 'parallelize', 'vectorize'],
  install_requires=[
        'pandas>=0.23.0',
        'psutil',
        'dask[complete]>=0.19.0',
	'tqdm'
  ],
  classifiers = [],
)
