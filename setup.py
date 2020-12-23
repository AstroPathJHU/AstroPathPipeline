import setuptools

setuptools.setup(
  name = "astropath-calibration",
  packages = setuptools.find_packages(include=["astropath_calibration*"]),
  entry_points = {
    "console_scripts": [
      "alignmentcohort=astropath_calibration.alignment.alignmentcohort:main",
      "badregionscohort=astropath_calibration.badregions.cohort:main",
      "badregionssample=astropath_calibration.badregions.sample:main",
      "evaluate_exposure_time=astropath_calibration.exposuretime.evaluate_exposure_time:main",
      "run_exposure_time_fits=astropath_calibration.exposuretime.run_exposure_time_fits:main",
      "extractlayer=astropath_calibration.extractlayer.extractlayer:main",
      "run_flatfield=astropath_calibration.flatfield.run_flatfield:main",
      "run_image_correction=astropath_calibration.image_correction.run_image_correction:main",
      "prepdbsample=astropath_calibration.prepdb.prepdbsample:main",
      "find_exposure_time_samples=astropath_calibration.scripts.find_exposure_time_samples:main",
      "run_warping_fits=astropath_calibration.warping.run_warping_fits:main",
      "run_many_warp_fits_with_pool=astropath_calibration.warping.run_many_warp_fits_with_pool:main",
      "run_warp_fitter=astropath_calibration.warping.run_warp_fitter:main",
      "zoom.py=astropath_calibration.zoom.zoom:main",
      "zoomcohort.py=astropath_calibration.zoom.zoomcohort:main",
    ],
  },
  install_requires = [
    "cvxpy",
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
    "pyvips",
    "reikna",
    "seaborn",
    "scikit-image",
    "scikit-learn>=0.17",
    "scipy",
    "setuptools-scm",
    "uncertainties",
  ],
  extras_require = {
    "test": ["flake8", "texoutparse"],
  }
)
