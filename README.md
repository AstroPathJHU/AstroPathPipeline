# Introduction

This is the main repository for the Astropath group's code that has been ported to Python. 

## Installation

To install the code, first check out the repository, enter its directory, and run
```bash
pip install .
```
If you want to continue developing the code after installing, run instead
```bash
pip install --editable .
```

Once the code is installed using either of those lines, you can run
```python
import astropath
```
from any directory.

## Dependencies

In addition to 64-bit Python 3, The following packages are required to run code in this repository:
- NumPy  ([homepage](https://numpy.org/))
- SciPy ([homepage](https://www.scipy.org/))
- scikit-learn ([homepage](https://scikit-learn.org/stable/))
- scikit-image ([homepage](https://scikit-image.org/))
- opencv-python ([homepage](https://opencv.org/), [python version PyPi page](https://pypi.org/project/opencv-python/))
- Numba ([homepage](https://numba.pydata.org/))
- PyOpenCL ([homepage](https://documen.tician.de/pyopencl/)) 
- Reikna ([homepage](http://reikna.publicfields.net/en/latest/))
- cvxpy ([homepage](https://www.cvxpy.org/))
- Matplotlib ([homepage](https://matplotlib.org/))
- seaborn ([homepage](https://seaborn.pydata.org/))
- pyvips ([homepage](https://libvips.github.io/libvips/), [python binding homepage](https://libvips.github.io/pyvips/intro.html))
- imagecodecs ([PyPi page](https://pypi.org/project/imagecodecs/))
- NetworkX ([homepage](https://networkx.org/)) 
- uncertainties ([homepage](https://uncertainties-python-package.readthedocs.io/en/latest/))
- methodtools ([PyPi page](https://pypi.org/project/methodtools/))
- more_itertools ([homepage](https://more-itertools.readthedocs.io/en/stable/))
- jxmlease ([GitHub page](https://github.com/Juniper/jxmlease))

These packages will all be installed automatically when you install this repository.

A testing environment exists for this code, with a corresponding minimal environment inside a Docker container. The current environment setup for that container can be found in a sister repository [here](https://github.com/AstroPathJHU/astropathtest/blob/master/Dockerfile).
