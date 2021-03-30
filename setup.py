import setuptools, site

site.ENABLE_USER_SITE = True #https://www.scivision.dev/python-pip-devel-user-install/

setupkwargs = dict(
  name = "astropath-calibration",
  packages = setuptools.find_packages(include=["astropath*"]),
  entry_points = {
    "console_scripts": [
      "alignmentcohort=astropath.alignment.alignmentcohort:main",
      "annowarpcohort=astropath.annowarp.annowarpcohort:main",
      "annowarpfailedfitswithconstraint=astropath.annowarp.annowarpfailedfitswithconstraint:main",
      "compareprepdbcohort=astropath.prepdb.compareprepdbcohort:main",
      "deepzoomcohort=astropath.deepzoom.deepzoomcohort:main",
      "evaluate_exposure_time=astropath.exposuretime.evaluate_exposure_time:main",
      "extractlayer=astropath.extractlayer.extractlayer:main",
      "find_exposure_time_samples=astropath.scripts.find_exposure_time_samples:main",
      "find_slide_overexposed_hpfs=astropath.overexposed_hpfs.find_slide_overexposed_hpfs:main",
      "geomcohort=astropath.geom.geomcohort:main",
      "geomcellcohort=astropath.geomcell.geomcellcohort:main",
      "prepdbcohort=astropath.prepdb.prepdbcohort:main",
      "prepdbsample=astropath.prepdb.prepdbsample:main",
      "read_annotation_xml=astropath.baseclasses.annotationpolygonxmlreader:main",
      "run_exposure_time_fits=astropath.exposuretime.run_exposure_time_fits:main",
      "run_flatfield=astropath.flatfield.run_flatfield:main",
      "run_image_correction=astropath.image_correction.run_image_correction:main",
      "run_many_warp_fits_with_pool=astropath.warping.run_many_warp_fits_with_pool:main",
      "run_warp_fitter=astropath.warping.run_warp_fitter:main",
      "run_warping_fits=astropath.warping.run_warping_fits:main",
      "stitchmaskcohort=astropath.zoom.stitchmaskcohort:main",
      "zoomcohort=astropath.zoom.zoomcohort:main",
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
    "test": ["flake8", "pyflakes", "texoutparse"],
    "gdal": ["gdal>=3.2.1"],
    "vips": ["pyvips"],
  }
)

setupkwargs["extras_require"]["all"] = sum(setupkwargs["extras_require"].values(), [])

setuptools.setup(**setupkwargs)
