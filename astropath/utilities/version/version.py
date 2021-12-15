"""
Determines the version number automatically from git.
If installing with pip, the version will come from the git tag used when building
If installing with pip install --editable, it will determine the version
from the most recent git tag, the current commit, and the edits to the working
tree, if any.

Setting the _ASTROPATH_VERSION_NO_GIT environment variable to 1 will disable
running git commands.  The only reason to do this is if git is slow, which
happens when editing the repo on cygwin and running on cmd or powershell.
"""

import datetime, os, pkg_resources, re, setuptools_scm
from ... import __name__ as package_name

try:
  if int(os.environ.get("_ASTROPATH_VERSION_NO_GIT", 0)):
    env_var_no_git = True
    raise LookupError
  env_var_no_git = False
  astropathversion = "v"+setuptools_scm.get_version(root="../../..", relative_to=__file__)
  have_git = True
except LookupError:
  have_git = False
  try:
    astropathversion = "v"+pkg_resources.get_distribution(package_name).version
  except pkg_resources.DistributionNotFound:
    astropathversion = "v0.0.0.dev0+g0000000.d"+datetime.date.today().strftime("%Y%m%d")

astropathversionregex = re.compile(r"v(?P<version>[0-9]+(?:\.[0-9]+)*)(?P<dev>\.dev(?P<devnumber>[0-9]+)\+g(?P<commit>[0-9a-f]+))?(?:\.d(?P<date>[0-9]+))?")
astropathversionmatch = astropathversionregex.match(astropathversion)
if not astropathversionmatch:
  raise RuntimeError(f"got a version number '{astropathversion}' that doesn't match the desired regex")
