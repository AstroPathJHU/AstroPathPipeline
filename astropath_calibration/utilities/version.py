import pkg_resources, setuptools_scm
from .. import __name__ as package_name

try:
  astropathversion = "v"+setuptools_scm.get_version(root="../..", relative_to=__file__)
except LookupError:
  try:
    astropathversion = "v"+pkg_resources.get_distribution(package_name).version
  except pkg_resources.DistributionNotFound:
    astropathversion = "unknown version"