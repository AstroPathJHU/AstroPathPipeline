import setuptools, site

site.ENABLE_USER_SITE = True #https://www.scivision.dev/python-pip-devel-user-install/

setupkwargs = dict(
  name = "astropath",
  packages = setuptools.find_packages(include=["astropath*"]),
  entry_points = {
    "console_scripts": [
      "aligncohort=astropath.slides.align.aligncohort:main",
      "alignsample=astropath.slides.align.alignsample:main",
      "extractlayer=astropath.hpfs.extractlayer.extractlayer:main",
      "prepdbcohort=astropath.slides.prepdb.prepdbcohort:main",
      "prepdbsample=astropath.slides.prepdb.prepdbsample:main",
    ],
  },
  install_requires = [
    "contextlib2>=0.6.0",
    "cvxpy",
    "dataclassy @ git+git://github.com/hroskes/dataclassy@262fdeff62fd401f2da83bfadafdb1a22fa16448#egg=dataclassy",
    "imagecodecs",
    "jxmlease>=1.0.2dev1",
    "matplotlib>=3.3.2",
    "methodtools",
    "more_itertools>=8.3.0",
    "networkx",
    "numba",
    "numpy>=1.17.0",
    "opencv-python",
    "pyopencl",
    "reikna",
    "seaborn",
    "scikit-image>=0.17,<0.18", #see test/testmisc.py - we want polygon.numpyarray to return reproducible results, and skimage.draw.polygon's behavior changes between 0.17 and 0.18.  Want to support python 3.6 for now so we need to stick to 0.17.
    "scikit-learn>=0.17",
    "scipy>=0.12",
    "setuptools-scm",
    "uncertainties",
  ],
  extras_require = {
    "test": ["beautifulsoup4", "flake8", "lxml", "marko[toc]", "pyflakes", "texoutparse"],
    "gdal": ["gdal>=3.2.1"],
    "vips": ["pyvips"],
  }
)

setupkwargs["extras_require"]["all"] = sum(setupkwargs["extras_require"].values(), [])

setuptools.setup(**setupkwargs)
