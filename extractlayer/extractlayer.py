import logging, methodtools, numpy as np, pathlib, PIL.Image

logger = logging.getLogger("extractlayer")
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(message)s, %(funcName)s, %(asctime)s"))
logger.addHandler(handler)

class LayerExtractor:
  def __init__(self, root1, root2, samp):
    self.root1 = pathlib.Path(root1)
    self.root2 = pathlib.Path(root2)
    self.samp = samp

  @methodtools.lru_cache()
  def __getlayershape(self):
    filename = next((self.root1/self.samp/"inform_data"/"Component_Tiffs").glob("*.tif"))
    with PIL.Image.open(filename) as f:
      return f.size

  @property
  def fwfiles(self):
    return (self.root2/self.samp).glob("*.fw")

  @methodtools.lru_cache()
  def __getnlayers(self):
    filename = next(self.fwfiles)
    nlayers = len(np.memmap(filename, dtype=np.uint16)) / np.product(self.__getlayershape())
    if not nlayers.is_integer():
      raise ValueError(f"file seems to have {nlayers} layers??")
    return int(nlayers)

  @property
  def shape(self):
    return (self.__getnlayers(),) + self.__getlayershape()

  def extractlayers(self, *, layers={1}, alreadyexistsstrategy="error"):
    nfiles = len(list(self.fwfiles))
    for i, filename in enumerate(self.fwfiles, start=1):
      logger.info(f"{i:5d}/{nfiles} {filename.name}")
      with open(filename, "rb") as f:
        memmap = np.memmap(f, dtype=np.uint16, order="F", shape=self.shape, mode="r")
        for layer in layers:
          outfilename = self.root2/self.samp/f"{filename.stem}.fw{layer:02d}"
          if outfilename.exists():
            if alreadyexistsstrategy == "error":
              raise OSError(f"{outfilename} already exists")
            elif alreadyexistsstrategy == "keep":
              continue
            elif alreadyexistsstrategy == "overwrite":
              pass
            else:
              raise ValueError("Invalid alreadyexistsstrategy {alreadyexistsstrategy}: options are error, keep, or overwrite")
          output = memmap[layer-1,:,:].T
          newmemmap = np.memmap(outfilename, dtype=np.uint16, order="F", shape=output.shape, mode="w+")
          newmemmap[:] = output
