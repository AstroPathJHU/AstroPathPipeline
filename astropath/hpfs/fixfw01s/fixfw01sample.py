import contextlib, dataclassy, job_lock, numpy as np
from ...shared.sample import ReadRectanglesDbloadIm3, ReadRectanglesIm3Base, ReadRectanglesIm3FromXML, WorkflowSample
from ...slides.prepdb.prepdbsample import PrepDbSample
from ...utilities.miscfileio import memmapcontext, rm_missing_ok

class FixFW01SampleBase(ReadRectanglesIm3Base, WorkflowSample):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, readlayerfile=False, **kwargs)

  @classmethod
  def logmodule(self): return "fixfw01"

  def fixfw01(self, check=True, removecorrupt=True, removedisagreement=False):
    n = len(self.rectangles)
    for i, r in enumerate(self.rectangles):
      self.logger.debug(f"rectangle {i}/{n}")
      kwargs = {field: getattr(r, field) for field in set(dataclassy.fields(type(r)))}
      assert kwargs["readlayerfile"]
      kwargs["readlayerfile"] = False
      newr = type(r)(**kwargs)
      oldfile = r.im3file
      newfile = newr.im3file
      assert oldfile != newfile

      with r.using_im3() as oldim3:
        with contextlib.ExitStack() as stack:
          try:
            newim3 = stack.enter_context(newr.using_im3())
          except ValueError as e:
            if str(e) == "mmap length is greater than file size":
              msg = f"{newfile.name} is corrupt"
              if removecorrupt:
                self.logger.warning(f"{msg}, removing it")
                newfile.unlink()
              else:
                raise ValueError(msg)
            else:
              raise
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
        with job_lock.JobLock(lockfile, outputfilenames=[newfile]) as lock:
          assert lock
          try:
            with open(newfile, "xb") as newf, memmapcontext(newf, dtype=np.uint16, shape=tuple(newr.imageshapeininput), order="F", mode="r") as newmemmap:
              assert newr.imagetransposefrominput == (0, 1)
              newmemmap[:] = oldim3
          except:
            rm_missing_ok(newfile)

  def run(self, **kwargs):
    self.fixfw01(**kwargs)

  def inputfiles(self, **kwargs):
    return super().inputfiles(**kwargs) + [
      *(r.im3file for r in self.rectangles),
    ]

  @classmethod
  def getoutputfiles(cls, SlideID, *, shardedim3root, layerim3, **otherkwargs):
    return [
      filename.with_suffix(".fw{layerim3:02d}")
      for filename in (shardedim3root/SlideID).glob(f"{SlideID}_[*].fw")
    ]

class FixFW01SampleXML(FixFW01SampleBase, ReadRectanglesIm3FromXML):
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
