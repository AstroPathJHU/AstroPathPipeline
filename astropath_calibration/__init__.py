try:
  from osgeo import ogr
except ImportError:
  pass
else:
  ogr.UseExceptions()
