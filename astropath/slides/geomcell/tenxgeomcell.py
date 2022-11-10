import contextlib, csv, job_lock, methodtools, multiprocessing as mp, numpy as np, pathlib, skimage.measure
from ...shared.argumentparser import ArgumentParserWithVersionRequirement, ParallelArgumentParser
from ...shared.imageloader import ImageLoaderPng
from ...shared.logging import printlogger, ThingWithLogger
from ...utilities import units
from ...utilities.tableio import writetable
from .geomcellsample import CellGeomLoad, PolygonFinder

class MiniField(units.ThingWithPscale):
  def __init__(self, hpfid, pngfilename, csvfilename, pscale, position):
    self.hpfid = hpfid
    self.pngfilename = pngfilename
    self.__pscale = pscale
    self.__position = position
    self.__imageloader = ImageLoaderPng(filename=self.pngfilename)
    self.__width = self.__height = None

  @contextlib.contextmanager
  def using_image(self):
    with self.__imageloader.using_image() as im:
      self.__height, self.__width = np.array(im.shape) * units.onepixel(pscale=self.pscale)
      yield im

  @property
  def n(self): return self.hpfid

  @property
  def pscale(self): return self.__pscale
  @property
  def width(self):
    if self.__width is None:
      with self.using_image():
        assert self.__width is not None
    return self.__width
  @property
  def height(self):
    if self.__height is None:
      with self.using_image():
        assert self.__height is not None
    return self.__height
  @property
  def shape(self): return np.array([self.width, self.height])
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

class TenXSampleWithFields(units.ThingWithPscale, contextlib.ExitStack):
  def __init__(self, *args, mainfolder, **kwargs):
    super().__init__(*args, **kwargs)
    self.__mainfolder = pathlib.Path(mainfolder)
  @property
  def mainfolder(self): return self.__mainfolder
  @property
  def csvfolder(self): return self.__mainfolder/"tile"/"csv"
  @property
  def pngfolder(self): return self.__mainfolder/"tile"/"nuclear_mask"
  @property
  def geomfolder(self): return self.__mainfolder/"tile"/"geom"

  @methodtools.lru_cache()
  @property
  def fields(self):
    result = []
    with open(self.mainfolder/"whole_slide"/"tile_ID_and_coordinates.csv") as f:
      reader = csv.DictReader(f)
      for row in reader:
        hpfid = int(row["tile_ID"])
        coordinates = row["tile_coordinates [up_L, down_L, up_R, down_R]"]
        x1, x2, y1, y2 = (int(_)*self.onepixel for _ in coordinates.strip("[]").split(","))
        pngfilename = self.pngfolder/f"tile_nuclear_mask_{hpfid-1}.png"
        if not pngfilename.exists(): raise FileNotFoundError(f"{pngfilename} does not exist")
        csvfilename = self.csvfolder/f"tile_cell_coordinates_{hpfid-1}.csv"
        assert csvfilename.exists()
        mf = MiniField(
          hpfid=hpfid,
          pngfilename=pngfilename,
          csvfilename=csvfilename,
          pscale=self.pscale,
          position=np.array([x1, y1]),
        )
        self.enter_context(mf.using_image())
        np.testing.assert_array_equal(mf.shape, np.array([y2-y1, x2-x1]))
        result.append(mf)
    return result
        
class TenXGeomCell(TenXSampleWithFields, ArgumentParserWithVersionRequirement, ParallelArgumentParser, ThingWithLogger):
  def __init__(self, *args, njobs=None, **kwargs):
    super().__init__(*args, **kwargs)
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
    return printlogger("tenxgeomcell")

  @staticmethod
  def runHPF(i, field, *, logger, outputfolder, _debugdraw=(), _debugdrawonerror=False, _onlydebug=False, repair=True, minarea, nfields, unitsargs):
    with units.setup_context(*unitsargs):
      geomload = []
      onepixel = field.onepixel
      outputfile = outputfolder/field.pngfilename.with_suffix(".csv").name
      lockfile = outputfile.with_suffix(".lock")
      with job_lock.JobLock(lockfile, outputfiles=[outputfile]) as lock:
        if not lock: return
        logger.info(f"writing cells for field {field.n} ({i} / {nfields})")
        with field.using_image() as image:
          for celltype, imlayer in (2, image),: #2 = nuclei
            properties = skimage.measure.regionprops(imlayer)
            ismembranelayer = False
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

        writetable(outputfile, geomload, rowclass=CellGeomLoad)

  def run(self, minarea=None, **kwargs):
    if minarea is None:
      minarea = 3*self.onemicron**2
    runHPFkwargs = {
      "minarea": minarea,
      "logger": self.logger,
      "nfields": len(self.fields),
      "outputfolder": self.geomfolder,
      "unitsargs": units.currentargs(),
      **kwargs,
    }
    self.geomfolder.mkdir(parents=True, exist_ok=True)
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
      "mainfolder": parsed_args_dict.pop("main_folder"),
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
    p.add_argument("main_folder", type=pathlib.Path, help="Folder with the whole_slide and tile folders")
    p.add_argument("--units", choices=("safe", "fast", "fast_pixels", "fast_microns"), default="safe")
    return p

  @property
  def pscale(self): return 1

def main(args=None):
  return TenXGeomCell.runfromargumentparser(args=args)

if __name__ == "__main__":
  main()
