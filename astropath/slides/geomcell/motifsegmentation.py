import contextlib, fractions, methodtools, numpy as np, pathlib, re, skimage.measure, tifffile
from ...shared.argumentparser import ArgumentParserWithVersionRequirement
from ...shared.logging import printlogger, ThingWithLogger
from ...utilities import units
from ...utilities.tableio import writetable
from .geomcellsample import CellGeomLoad, PolygonFinder

class MiniField(units.ThingWithPscale):
  def __init__(self, hpfid, tifffile):
    self.hpfid = hpfid
    self.tifffile = tifffile

  @property
  def n(self): return self.hpfid

  @methodtools.lru_cache()
  @property
  def imageinfo(self):
    with tifffile.TiffFile(self.tifffile) as f:
      pscales = set()
      shapes = set()
      for page in f.pages:
        xresolution = page.tags["XResolution"].value
        pscales.add(fractions.Fraction(*xresolution) / 10000)
        yresolution = page.tags["YResolution"].value
        pscales.add(fractions.Fraction(*yresolution) / 10000)
        shapes.add(page.shape)
      pscale, = pscales
      shape, = shapes
      shape = np.array(shape) * units.onepixel(pscale)
      height, width = shape
    return pscale, width, height

  @property
  def pscale(self):
    return self.imageinfo[0]
  @property
  def width(self):
    return self.imageinfo[1]
  @property
  def height(self):
    return self.imageinfo[2]

  @methodtools.lru_cache()
  @property
  def pxvec(self):
    match = re.match(r"[0-9A-Za-z_]+_\[([0-9]+),([0-9]+)\]_binary_seg_maps\.tif", self.tifffile.name)
    x = int(match.group(1)) * self.onepixel
    y = int(match.group(2)) * self.onepixel
    return np.array([x, y])
  @property
  def px(self): return self.pxvec[0]
  @property
  def py(self): return self.pxvec[1]

  @property
  def mxbox(self):
    return np.array([self.py, self.px, self.py+self.height, self.px+self.width])

class MotifGeomCell(ArgumentParserWithVersionRequirement, ThingWithLogger, units.ThingWithPscale):
  def __init__(self, *, tifffolder, logfolder, outputfolder, **kwargs):
    super().__init__(**kwargs)
    self.tifffolder = pathlib.Path(tifffolder)
    self.logfolder = pathlib.Path(logfolder)
    self.outputfolder = pathlib.Path(outputfolder)

  @property
  def logger(self):
    return printlogger("motifgeomcell")

  @property
  def tiffs(self):
    return list(self.tifffolder.glob("*_seg_maps.tif"))

  def runHPF(self, field, *, _debugdraw=(), _debugdrawonerror=False, _onlydebug=False, repair=True, minarea):
    geomload = []
    onepixel = field.onepixel
    outputfile = self.outputfolder/field.tifffile.name.with_suffix(".csv")
    if outputfile.exists(): return
    with tifffile.TiffFile(field.tifffile) as f:
      nuclei, _, membranes = f.pages
      nuclei = nuclei.asarray()
      membranes = membranes.asarray()
      for celltype, imlayer in (0, membranes), (2, nuclei):
        properties = skimage.measure.regionprops(imlayer)
        ismembranelayer = imlayer is membranes
        for cellproperties in properties:
          if not np.any(cellproperties.image):
            assert False
            continue

          celllabel = cellproperties.label
          if _onlydebug and (field.n, celltype, celllabel) not in _debugdraw: continue
          polygon = PolygonFinder(imlayer, celllabel, ismembrane=ismembranelayer, bbox=cellproperties.bbox, pxvec=field.pxvec, mxbox=field.mxbox, pscale=field.pscale, logger=self.logger, loginfo=f"{field.n} {celltype} {celllabel}", _debugdraw=(field.n, celltype, celllabel) in _debugdraw, _debugdrawonerror=_debugdrawonerror, repair=repair).findpolygon()
          if polygon is None: continue
          if polygon.area < minarea: continue

          box = np.array(cellproperties.bbox).reshape(2, 2)[:,::-1] * onepixel * 1.0
          box += field.pxvec
          box = box // onepixel * onepixel

          geomload.append(
            CellGeomLoad(
              field=field.n,
              ctype=celltype,
              n=celllabel,
              box=box,
              poly=polygon,
              pscale=field.pscale,
            )
          )

    writetable(outputfile, geomload)

  @methodtools.lru_cache()
  @property
  def fields(self):
    return [MiniField(hpfid, tifffile) for hpfid, tifffile in enumerate(sorted(self.tifffolder.glob("*_binary_seg_maps.tif")), start=1)]

  @methodtools.lru_cache()
  @property
  def pscale(self):
    pscale, = {f.pscale for f in self.fields}
    return pscale

  def run(self, minarea=None, **kwargs):
    if minarea is None:
      minarea = 3*self.onemicron**2
    for field in self.fields:
      self.runHPF(field, minarea=minarea, **kwargs)

  @classmethod
  def runfromargsdicts(cls, *, initkwargs, runkwargs, misckwargs):
    if misckwargs:
      raise TypeError(f"Some miscellaneous kwargs were not processed:\n{misckwargs}")
    sample = cls(**initkwargs)
    with sample:
      sample.run(**runkwargs)
    return sample

  @classmethod
  def initkwargsfromargumentparser(cls, parsed_args_dict):
    return {
      **super().initkwargsfromargumentparser(parsed_args_dict),
      "tifffolder": parsed_args_dict.pop("tiff_folder"),
      "outputfolder": parsed_args_dict.pop("output_folder"),
      "logfolder": parsed_args_dict.pop("log_folder"),
    }

  @classmethod
  def makeargumentparser(cls, **kwargs):
    p = super().makeargumentparser(**kwargs)
    p.add_argument("--tiff-folder", type=pathlib.Path, required=True, help="Folder with the segmented tiffs")
    p.add_argument("--output-folder", type=pathlib.Path, required=True, help="Folder for the output csvs")
    p.add_argument("--log-folder", type=pathlib.Path, required=True, help="Folder for the log files")
    return p



def main(args=None):
  return MotifGeomCell.runfromargumentparser(args=args)

if __name__ == "__main__":
  main()
