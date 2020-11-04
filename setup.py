import setuptools

setuptools.setup(
  name = "astropath-calibration",
  packages = setuptools.find_packages(include=["astropath_calibration*"]),
  entry_points = {
    "console_scripts": [
      "alignmentcohort.py=astropath_calibration.alignment.alignmentcohort:main",
      "badregionscohort.py=astropath_calibration.badregions.cohort:main",
      "badregionssample.py=astropath_calibration.badregions.sample:main",
      "evaluate_exposure_time.py=astropath_calibration.exposuretime.evaluate_exposure_time:main",
      "run_exposure_time_fits.py=astropath_calibration.exposuretime.run_exposure_time_fits:main",
      "extractlayer.py=astropath_calibration.extractlayer.extractlayer:main",
      "run_flatfield.py=astropath_calibration.flatfield.run_flatfield:main",
      "run_for_sample.py=astropath_calibration.image_correction.run_for_sample:main",
      "prepdbsample.py=astropath_calibration.prepdb.prepdbsample:main",
      "find_exposure_time_samples.py=astropath_calibration.scripts.find_exposure_time_samples:main",
      "run_fits_for_sample.py=astropath_calibration.warping.run_fits_for_sample:main",
      "run_many_fits_with_pool.py=astropath_calibration.warping.run_many_fits_with_pool:main",
      "run_warp_fitter.py=astropath_calibration.warping.run_warp_fitter:main",
      "zoomcohort.py=astropath_calibration.zoom.zoomcohort:main",
    ],
  },
  install_requires = [
    "cvxpy",
    "imagecodecs",
    "jxmlease>=1.0.2dev1",
    "matplotlib",
    "methodtools",
    "more_itertools",
    "networkx",
    "numba",
    "numpy",
    "opencv-python",
    "pyopencl",
    "pyvips",
    "reikna",
    "seaborn",
    "scikit-image",
    "scikit-learn",
    "scipy",
    "setuptools-scm",
    "uncertainties",
  ],
  extras_require = {
    "test": ["flake8", "texoutparse"],
  }
)
