import fractions, job_lock, methodtools, multiprocessing as mp, numpy as np, pathlib, skimage.measure, tifffile
from ...shared.argumentparser import ArgumentParserWithVersionRequirement, ParallelArgumentParser
from ...shared.csvclasses import constantsdict
from ...shared.logging import printlogger, ThingWithLogger
from ...utilities import units
from ...utilities.tableio import writetable
from .geomcellsample import CellGeomLoad, PolygonFinder

class MiniField(units.ThingWithPscale):
  def __init__(self, hpfid, tifffilename, pscale, shift):
    self.hpfid = hpfid
    self.tifffile = tifffilename
    with tifffile.TiffFile(self.tifffile) as f:
      shapes = set()
      positions = set()
      for page in f.pages:
        xresolution = page.tags["XResolution"].value
        xresolution = float(fractions.Fraction(*xresolution) / 10000)
        yresolution = page.tags["YResolution"].value
        yresolution = float(fractions.Fraction(*yresolution) / 10000)
        np.testing.assert_allclose([pscale, pscale], [xresolution, yresolution], atol=1e-10, rtol=0)
        xposition = page.tags["XPosition"].value
        xposition = float(fractions.Fraction(*xposition))
        xposition = units.Distance(centimeters=xposition, pscale=pscale)
        yposition = page.tags["YPosition"].value
        yposition = float(fractions.Fraction(*yposition))
        yposition = units.Distance(centimeters=yposition, pscale=pscale)
        positions.add((xposition, yposition))
        shapes.add(page.shape)

      shape, = shapes
      shape = np.array(shape) * units.onepixel(pscale)
      height, width = shape
      position, = positions
      position = np.array(position) + shift
    self.__pscale = pscale
    self.__width = width
    self.__height = height
    self.__position = position

  @property
  def n(self): return self.hpfid

  @property
  def pscale(self): return self.__pscale
  @property
  def width(self): return self.__width
  @property
  def height(self): return self.__height
  @property
  def position(self): return self.__position

  @property
  def pxvec(self):
    return self.position
  @property
  def px(self): return self.pxvec[0]
  @property
  def py(self): return self.pxvec[1]

  @property
  def mxbox(self):
    return np.array([self.py, self.px, self.py+self.height, self.px+self.width])

class MotifSampleWithFields(units.ThingWithPscale):
  @methodtools.lru_cache()
  @property
  def fields(self):
    return [MiniField(hpfid, tifffile, pscale=self.pscale, shift=self.shiftqptiff) for hpfid, tifffile in enumerate(sorted(self.tifffolder.glob("*_binary_seg_maps.tif")), start=1)]

  @methodtools.lru_cache()
  @property
  def constantsdict(self):
    return constantsdict(self.dbloadfolder/"constants.csv")

  @property
  def shiftqptiff(self):
    constants = self.constantsdict
    return np.array([constants["xshift"], constants["yshift"]])

  @methodtools.lru_cache()
  @property
  def pscale(self):
    return float(self.constantsdict["pscale"])

class MotifGeomCell(MotifSampleWithFields, ArgumentParserWithVersionRequirement, ParallelArgumentParser, ThingWithLogger):
  def __init__(self, *, tifffolder, logfolder, outputfolder, dbloadfolder, njobs=None, **kwargs):
    super().__init__(**kwargs)
    self.tifffolder = pathlib.Path(tifffolder)
    self.logfolder = pathlib.Path(logfolder)
    self.outputfolder = pathlib.Path(outputfolder)
    self.dbloadfolder = pathlib.Path(dbloadfolder)
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

  @staticmethod
  def runHPF(i, field, *, logger, outputfolder, _debugdraw=(), _debugdrawonerror=False, _onlydebug=False, repair=True, minarea, nfields, unitsargs):
    with units.setup_context(*unitsargs):
      geomload = []
      onepixel = field.onepixel
      outputfile = outputfolder/field.tifffile.with_suffix(".csv").name.replace(",", ".")
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

  def run(self, minarea=None, **kwargs):
    if minarea is None:
      minarea = 3*self.onemicron**2
    runHPFkwargs = {
      "minarea": minarea,
      "logger": self.logger,
      "nfields": len(self.fields),
      "outputfolder": self.outputfolder,
      "unitsargs": units.currentargs(),
      **kwargs,
    }
    self.outputfolder.mkdir(parents=True, exist_ok=True)
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
    with units.setup_context(misckwargs.pop("units")):
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
      "dbloadfolder": parsed_args_dict.pop("dbload_folder"),
      "logfolder": parsed_args_dict.pop("log_folder"),
    }

  @classmethod
  def misckwargsfromargumentparser(cls, parsed_args_dict):
    return {
      **super().misckwargsfromargumentparser(parsed_args_dict),
      "units": parsed_args_dict.pop("units"),
    }

  @classmethod
  def makeargumentparser(cls, **kwargs):
    p = super().makeargumentparser(**kwargs)
    p.add_argument("--tiff-folder", type=pathlib.Path, required=True, help="Folder with the segmented tiffs")
    p.add_argument("--output-folder", type=pathlib.Path, required=True, help="Folder for the output csvs")
    p.add_argument("--dbload-folder", type=pathlib.Path, required=True, help="Folder with the dbload csvs")
    p.add_argument("--log-folder", type=pathlib.Path, required=True, help="Folder for the log files")
    p.add_argument("--units", choices=("safe", "fast", "fast_pixels", "fast_microns"), default="safe")
    return p

def main(args=None):
  return MotifGeomCell.runfromargumentparser(args=args)

if __name__ == "__main__":
  main()
