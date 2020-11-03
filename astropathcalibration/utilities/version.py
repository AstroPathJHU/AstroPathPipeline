import pkg_resources, setuptools_scm

try:
  astropathversion = "v"+setuptools_scm.get_version(root="../..", relative_to=__file__)
except LookupError:
  try:
    raise pkg_resources.DistributionNotFound
    astropathversion = "v"+pkg_resources.get_distribution(__name__).version
    print(astropathversion)
  except pkg_resources.DistributionNotFound:
    astropathversion = "unknown version"
