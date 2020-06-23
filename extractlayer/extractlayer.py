import abc, argparse, collections, contextlib, methodtools, numpy as np, pathlib, subprocess, tempfile
from ..baseclasses.sample import FlatwSampleBase, LogSampleBase
from ..utilities import units
from ..utilities.misc import memmapcontext

here = pathlib.Path(__file__).parent

class LayerExtractorBase(FlatwSampleBase, LogSampleBase, contextlib.ExitStack, collections.abc.Sized):
  @property
  def logmodule(self): return "extractlayer"

  @abc.abstractproperty
  def fwfiles(self): pass

  @methodtools.lru_cache()
  def __getnlayers(self):
    filename = next(self.fwfiles)
    with memmapcontext(filename, dtype=np.uint16, mode="r") as memmap:
      nlayers = len(memmap) / units.pixels(self.tiffwidth * self.tiffheight, pscale=self.tiffpscale, power=2)
    if not nlayers.is_integer():
      raise ValueError(f"file seems to have {nlayers} layers??")
    return int(nlayers)

  @property
  def shape(self):
    return (self.__getnlayers(), units.pixels(self.tiffwidth, pscale=self.tiffpscale), units.pixels(self.tiffheight, pscale=self.tiffpscale))

  def extractlayers(self, *, layers={1}, alreadyexistsstrategy="error"):
    (self.root2/self.SlideID).mkdir(parents=True, exist_ok=True)
    nfiles = len(self)
    for i, filename in enumerate(self.fwfiles, start=1):
      self.logger.info(f"{i:5d}/{nfiles} {filename.name}")
      with memmapcontext(filename, dtype=np.uint16, order="F", shape=self.shape, mode="r") as memmap:
        for layer in layers:
          outfilename = self.root2/self.SlideID/f"{filename.stem}.fw{layer:02d}"
          if outfilename.exists():
            if alreadyexistsstrategy == "error":
              raise OSError(f"{outfilename} already exists")
            elif alreadyexistsstrategy == "keep":
              continue
            elif alreadyexistsstrategy == "overwrite":
              pass
            else:
              raise ValueError(f"Invalid alreadyexistsstrategy {alreadyexistsstrategy}: options are error, keep, or overwrite")
          output = memmap[layer-1,:,:].T
          with memmapcontext(outfilename, dtype=np.uint16, order="F", shape=output.shape, mode="w+") as newmemmap:
            newmemmap[:] = output

class LayerExtractor(LayerExtractorBase):
  @property
  def fwfiles(self):
    return (self.root2/self.SlideID).glob("*.fw")
  def __len__(self):
    return len(list(self.fwfiles))

class ShredderAndLayerExtractor(LayerExtractorBase):
  def __enter__(self):
    self.__tmpdir = pathlib.Path(self.enter_context(tempfile.TemporaryDirectory()))
    return super().__enter__()

  def __exit__(self, *stuff):
    super().__exit__(*stuff)
    del self.__tmpdir

  @property
  def tmpdir(self):
    try:
      return self.__tmpdir
    except AttributeError:
      raise TypeError("Need to use ShredderAndLayerExtractor in a with statement for proper cleanup")

  @property
  def im3files(self):
    return (self.root1/self.SlideID/"im3"/"flatw").glob("*.im3")

  @property
  def fwfiles(self):
    for im3 in self.im3files:
      fw = self.shred(im3)
      yield fw
      fw.unlink()

  def __len__(self):
    return len(list(self.im3files))

  def shred(self, im3):
    rawfile = self.tmpdir/(im3.stem+".raw")
    fwfile = self.tmpdir/(im3.stem+".fw")
    if fwfile.exists(): return fwfile
    try:
      out = subprocess.check_output([str(here/"ShredIm3.exe"), str(im3), "-o", str(fwfile.parent)], stderr=subprocess.STDOUT)
      if not rawfile.exists():
        e = RuntimeError("ShredIm3.exe failed to create the fw image")
        e.output = out
        raise e
    except (subprocess.CalledProcessError, RuntimeError) as e:
      print(e.output.decode())
      raise
    rawfile.rename(fwfile)
    return fwfile

if __name__ == "__main__":
  p = argparse.ArgumentParser()
  p.add_argument("root1")
  p.add_argument("root2")
  p.add_argument("samp")
  p.add_argument("--layer", action="append", type=int)
  p.add_argument("--shred", action="store_const", const=ShredderAndLayerExtractor, default=LayerExtractor, dest="ShredderClass")
  g = p.add_mutually_exclusive_group()
  g.add_argument("--error-if-exists", action="store_const", const="error", dest="alreadyexistsstrategy", default="error")
  g.add_argument("--overwrite", action="store_const", const="overwrite", dest="alreadyexistsstrategy")
  g.add_argument("--skip-existing", action="store_const", const="keep", dest="alreadyexistsstrategy")
  args = p.parse_args()

  with args.ShredderClass(root=args.root1, root2=args.root2, samp=args.samp) as le:
    kwargs = {}
    kwargs["alreadyexistsstrategy"] = args.alreadyexistsstrategy
    if args.layer: kwargs["layers"] = args.layer
    le.extractlayers(**kwargs)
