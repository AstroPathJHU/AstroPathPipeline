import datetime, os, pkg_resources, re, setuptools_scm
from .. import __name__ as package_name

try:
  if int(os.environ.get("_ASTROPATH_VERSION_NO_GIT", 0)):
    env_var_no_git = True
    raise LookupError
  env_var_no_git = False
  astropathversion = "v"+setuptools_scm.get_version(root="../..", relative_to=__file__)
except LookupError:
  try:
    astropathversion = "v"+pkg_resources.get_distribution(package_name).version
  except pkg_resources.DistributionNotFound:
    astropathversion = "v0.0.0.dev0+g0000000.d"+datetime.date.today().strftime("%Y%m%d")

astropathversionmatch = re.match(r"v(?P<version>[0-9]+(?:\.[0-9]+)*)(?P<dev>\.dev[0-9]+\+g[0-9a-f]+)?(?P<date>\.d[0-9]+)?", astropathversion)
if not astropathversionmatch:
  raise RuntimeError(f"got a version number '{astropathversion}' that doesn't match the desired regex")
