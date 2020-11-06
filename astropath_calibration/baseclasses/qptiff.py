import collections, methodtools, tifffile

class QPTiffZoomLevel(tuple):
  @methodtools.lru_cache()
  @property
  def tags(self):
    result = {}
    for key in self[0].tags.keys():
      tags = [page.tags[key] for page in self]
      values = {tag.value for tag in tags}
      try:
        value, = values
      except ValueError:
        continue
      name, = {tag.name for tag in tags}
      result[name] = value
    return result

  @methodtools.lru_cache()
  @property
  def shape(self):
    result, = {page.shape for page in self}
    return result

class QPTiff(tifffile.TiffFile):
  @property
  def zoomlevels(self):
    pages = []
    lastwidth = None
    for page in self.pages:
      if page.tags["SamplesPerPixel"].value != 1: continue
      if page.imagewidth != lastwidth:
        lastwidth = page.imagewidth
        if pages: yield QPTiffZoomLevel(pages)
        pages = []
      pages.append(page)
    if pages: yield QPTiffZoomLevel(pages)
