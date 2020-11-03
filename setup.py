import setuptools

setuptools.setup(
  name = "astropath-calibration",
  packages = ["astropathcalibration"],
  entry_points = {
    "console_scripts": [
      "alignmentcohort.py=astropathcalibration.alignment.alignmentcohort:main",
      "badregionscohort.py=astropathcalibration.badregions.cohort:main",
      "badregionssample.py=astropathcalibration.badregions.sample:main",
      "evaluate_exposure_time.py=astropathcalibration.exposuretime.evaluate_exposure_time:main",
      "run_exposure_time_fits.py=astropathcalibration.exposuretime.run_exposure_time_fits:main",
      "extractlayer.py=astropathcalibration.extractlayer.extractlayer:main",
      "run_flatfield.py=astropathcalibration.flatfield.run_flatfield:main",
      "run_for_sample.py=astropathcalibration.image_correction.run_for_sample:main",
      "prepdbsample.py=astropathcalibration.prepdb.prepdbsample:main",
      "find_exposure_time_samples.py=astropathcalibration.scripts.find_exposure_time_samples:main",
      "run_fits_for_sample.py=astropathcalibration.warping.run_fits_for_sample:main",
      "run_many_fits_with_pool.py=astropathcalibration.warping.run_many_fits_with_pool:main",
      "run_warp_fitter.py=astropathcalibration.warping.run_warp_fitter:main",
      "zoomcohort.py=astropathcalibration.zoom.zoomcohort:main",
    ],
  },
  install_requires = [
    "cvxpy",
    "imagecodecs",
    "jxmlease",
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
    "uncertainties",
  ],
  extras_require = {
    "test": ["flake8", "texoutparse"],
  }
)
