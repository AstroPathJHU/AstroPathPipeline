import setuptools, site

site.ENABLE_USER_SITE = True #https://www.scivision.dev/python-pip-devel-user-install/

setupkwargs = dict(
  name = "astropath-calibration",
  packages = setuptools.find_packages(include=["astropath_calibration*"]),
  entry_points = {
    "console_scripts": [
      "alignmentcohort=astropath_calibration.alignment.alignmentcohort:main",
      "annowarpcohort=astropath_calibration.annowarp.annowarpcohort:main",
      "annowarpfailedfitswithconstraint=astropath_calibration.annowarp.annowarpfailedfitswithconstraint:main",
      "deepzoomcohort=astropath_calibration.deepzoom.deepzoomcohort:main",
      "evaluate_exposure_time=astropath_calibration.exposuretime.evaluate_exposure_time:main",
      "extractlayer=astropath_calibration.extractlayer.extractlayer:main",
      "find_exposure_time_samples=astropath_calibration.scripts.find_exposure_time_samples:main",
      "find_slide_overexposed_hpfs=astropath_calibration.overexposed_hpfs.find_slide_overexposed_hpfs:main",
      "prepdbcohort=astropath_calibration.prepdb.prepdbcohort:main",
      "prepdbsample=astropath_calibration.prepdb.prepdbsample:main",
      "run_exposure_time_fits=astropath_calibration.exposuretime.run_exposure_time_fits:main",
      "run_flatfield=astropath_calibration.flatfield.run_flatfield:main",
      "run_image_correction=astropath_calibration.image_correction.run_image_correction:main",
      "run_many_warp_fits_with_pool=astropath_calibration.warping.run_many_warp_fits_with_pool:main",
      "run_warp_fitter=astropath_calibration.warping.run_warp_fitter:main",
      "run_warping_fits=astropath_calibration.warping.run_warping_fits:main",
      "stitchmaskcohort=astropath_calibration.zoom.stitchmaskcohort:main",
      "zoomcohort=astropath_calibration.zoom.zoomcohort:main",
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
    "scikit-image",
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
