import contextlib, fractions, job_lock, methodtools, multiprocessing as mp, numpy as np, pathlib, re, skimage.measure, tifffile
from ...shared.argumentparser import ArgumentParserWithVersionRequirement, ParallelArgumentParser
from ...shared.logging import printlogger, ThingWithLogger
from ...utilities import units
from ...utilities.tableio import writetable
from .geomcellsample import CellGeomLoad, PolygonFinder

class MiniField(units.ThingWithPscale):
  def __init__(self, hpfid, tifffilename):
    self.hpfid = hpfid
    self.tifffile = tifffilename
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
    self.__pscale = pscale
    self.__width = width
    self.__height = height

  @property
  def n(self): return self.hpfid  

  @property
  def pscale(self): return self.__pscale
  @property
  def width(self): return self.__width
  @property
  def height(self): return self.__height

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

class MotifGeomCell(ArgumentParserWithVersionRequirement, ParallelArgumentParser, ThingWithLogger, units.ThingWithPscale):
  def __init__(self, *, tifffolder, logfolder, outputfolder, njobs=None, **kwargs):
    super().__init__(**kwargs)
    self.tifffolder = pathlib.Path(tifffolder)
    self.logfolder = pathlib.Path(logfolder)
    self.outputfolder = pathlib.Path(outputfolder)
    self.__njobs = njobs

  @property
  def njobs(self):
    return self.__njobs
  def pool(self):
    nworkers = mp.cpu_count()
    if self.njobs is not None: nworkers = min(nworkers, self.njobs)
    return mp.get_context().Pool(nworkers)

  @property
  def logger(self):
    return printlogger("motifgeomcell")

  @property
  def tiffs(self):
    return list(self.tifffolder.glob("*_seg_maps.tif"))

  @staticmethod
  def runHPF(i, field, *, logger, outputfolder, _debugdraw=(), _debugdrawonerror=False, _onlydebug=False, repair=True, minarea, nfields):
    geomload = []
    onepixel = field.onepixel
    outputfile = outputfolder/field.tifffile.with_suffix(".csv").name
    lockfile = outputfile.with_suffix(".lock")
    with job_lock.JobLock(lockfile, outputfiles=[outputfile]) as lock:
      if not lock: return
      logger.info(f"writing cells for field {field.n} ({i} / {nfields})")
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
            polygon = PolygonFinder(imlayer, celllabel, ismembrane=ismembranelayer, bbox=cellproperties.bbox, pxvec=field.pxvec, mxbox=field.mxbox, pscale=field.pscale, logger=logger, loginfo=f"{field.n} {celltype} {celllabel}", _debugdraw=(field.n, celltype, celllabel) in _debugdraw, _debugdrawonerror=_debugdrawonerror, repair=repair).findpolygon()
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
    runHPFkwargs = {
      "minarea": minarea,
      "logger": self.logger,
      "nfields": len(self.fields),
      "outputfolder": self.outputfolder,
      **kwargs,
    }
    print(runHPFkwargs)
    if self.njobs is None or self.njobs > 1:
      with self.pool() as pool:
        results = [
          pool.apply_async(self.runHPF, args=(i, field), kwds=runHPFkwargs)
          for i, field in enumerate(self.fields, start=1)
        ]
        for r in results:
          r.get()
    else:
      for i, field in enumerate(self.fields, start=1):
        self.runHPF(i, field, **runHPFkwargs)

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
