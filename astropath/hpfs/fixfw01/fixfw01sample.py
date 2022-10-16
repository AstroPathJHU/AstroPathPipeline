import contextlib, dataclassy, job_lock, numpy as np
from ...shared.argumentparser import Im3ArgumentParser, SelectLayersArgumentParser, SelectRectanglesArgumentParser
from ...shared.sample import ReadRectanglesDbloadIm3, ReadRectanglesIm3Base, ReadRectanglesIm3FromXML, SelectLayersIm3WorkflowSample, XMLLayoutReaderTissue
from ...slides.prepdb.prepdbsample import PrepDbSample
from ...utilities.miscfileio import CorruptMemmapError, memmapcontext, rm_missing_ok

class FixFW01ArgumentParser(SelectLayersArgumentParser, SelectRectanglesArgumentParser, Im3ArgumentParser):
  @classmethod
  def makeargumentparser(cls, **kwargs):
    p = super().makeargumentparser(**kwargs)
    g = p.add_mutually_exclusive_group()
    g.add_argument("--error-on-corrupt", action="store_true", help="give an error if an fw01 file already exists but is corrupt")
    g.add_argument("--remove-corrupt", action="store_true", help="remove corrupt fw01 files, but give an error if the file size is correct and the contents are wrong (default)")
    g.add_argument("--remove-disagreement", action="store_true", help="remove fw01 files whose contents disagree with the fw")
    return p

  @classmethod
  def runkwargsfromargumentparser(cls, parsed_args_dict):
    error_on_corrupt = parsed_args_dict.pop("error_on_corrupt")
    remove_corrupt = parsed_args_dict.pop("remove_corrupt")
    remove_disagreement = parsed_args_dict.pop("remove_disagreement")

    assert error_on_corrupt + remove_corrupt + remove_disagreement <= 1
    if error_on_corrupt:
      remove_corrupt = remove_disagreement = False
    elif remove_disagreement:
      remove_corrupt = remove_disagreement = True
    else:
      remove_corrupt = True
      remove_disagreement = False

    return {
      **super().runkwargsfromargumentparser(parsed_args_dict),
      "removecorrupt": remove_corrupt,
      "removedisagreement": remove_disagreement,
    }

class FixFW01SampleBase(ReadRectanglesIm3Base, SelectLayersIm3WorkflowSample, FixFW01ArgumentParser):
  def __init__(self, *args, layer=None, layers=None, **kwargs):
    if layers is not None and layer is None:
      layer, = layers
      layers = None
    super().__init__(*args, layerim3=layer, layersim3=layers, readlayerfile=False, filetype="flatWarp", **kwargs)

  @classmethod
  def logmodule(self): return "fixfw01"

  def fixfw01(self, check=True, removecorrupt=True, removedisagreement=False):
    n = len(self.rectangles)
    removecorrupt = removecorrupt or removedisagreement
    for i, r in enumerate(self.rectangles, start=1):
      self.logger.debug(f"rectangle {i}/{n}")
      kwargs = {field: getattr(r, field) for field in set(dataclassy.fields(type(r)))}
      assert not kwargs["readlayerfile"]
      kwargs["readlayerfile"] = True
      newr = type(r)(**kwargs)
      oldfile = r.im3file
      newfile = newr.im3file
      assert oldfile != newfile

      with r.using_im3() as oldim3:
        with contextlib.ExitStack() as stack:
          try:
            newim3 = stack.enter_context(newr.using_im3())
          except CorruptMemmapError:
            msg = f"{newfile.name} is corrupt"
            if removecorrupt:
              self.logger.warning(f"{msg}, removing it")
              newfile.unlink()
            else:
              raise IOError(msg)
          except FileNotFoundError:
            pass
          else:
            if check:
              if not np.all(oldim3 == newim3):
                msg = f"{newfile.name} does not agree with {oldfile.name}"
                if removedisagreement:
                  self.logger.warning(f"{msg}, removing it")
                  newfile.unlink()
                else:
                  raise ValueError(msg)

        lockfile = newfile.with_suffix(".lock")
        with job_lock.JobLock(lockfile, outputfiles=[newfile]) as lock:
          if not lock: continue
          try:
            with memmapcontext(newfile, dtype=np.uint16, shape=tuple(newr.im3loader.imageshapeininput), order="F", mode="w+") as newmemmap:
              assert newr.im3loader.imagetransposefrominput == (0, 1)
              newmemmap[:] = oldim3
          except:
            rm_missing_ok(newfile)
            raise

  def run(self, **kwargs):
    self.fixfw01(**kwargs)

  def inputfiles(self, **kwargs):
    return super().inputfiles(**kwargs) + [
      *(r.im3file for r in self.rectangles),
    ]

  @classmethod
  def getoutputfiles(cls, SlideID, *, shardedim3root, layerim3, **otherkwargs):
    return [
      filename.with_suffix(f".fw{layerim3:02d}")
      for filename in (shardedim3root/SlideID).glob(f"{SlideID}_*.fw")
    ]

class FixFW01SampleXML(FixFW01SampleBase, ReadRectanglesIm3FromXML, XMLLayoutReaderTissue):
  @classmethod
  def workflowdependencyclasses(cls, **kwargs):
    return super().workflowdependencyclasses(**kwargs)

class FixFW01SampleDbload(FixFW01SampleBase, ReadRectanglesDbloadIm3):
  @classmethod
  def workflowdependencyclasses(cls, **kwargs):
    return super().workflowdependencyclasses(**kwargs) + [PrepDbSample]

  def inputfiles(self, **kwargs):
    return super().inputfiles(**kwargs) + [
      self.csv("rect")
    ]

def main_xml(args=None):
  FixFW01SampleXML.runfromargumentparser(args)
def main_dbload(args=None):
  FixFW01SampleDbload.runfromargumentparser(args)

if __name__ == "__main__":
  main_xml()
