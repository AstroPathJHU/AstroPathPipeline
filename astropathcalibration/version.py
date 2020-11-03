try:
  from ._version import version as astropathversion
  astropathversion = "v"+astropathversion
except ImportError:
  astropathversion = "unknown version"
