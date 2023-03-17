from ...hpfs.flatfield.meanimagesample import MeanImageSampleIm3Tissue
from ...shared.cohort import GeomFolderCohort, Im3Cohort, MaskCohort, PhenotypeFolderCohort
from ...shared.rectangle import AstroPathTissueMaskRectangle, GeomLoadRectangle, RectangleReadIm3SingleLayer, RectangleReadSegmentedComponentTiffMultiLayer, PhenotypedRectangle
from ...shared.sample import CellPhenotypeSampleBase, GeomSampleBase, InformSegmentationSample, MaskSampleBase, ReadRectanglesComponentTiffFromXML, ReadRectanglesIm3FromXML, TissueSampleBase

class InputCheckerRectangle(RectangleReadIm3SingleLayer, RectangleReadSegmentedComponentTiffMultiLayer, GeomLoadRectangle, PhenotypedRectangle, AstroPathTissueMaskRectangle):
  pass

class InputCheckerSampleBase(ReadRectanglesIm3FromXML, ReadRectanglesComponentTiffFromXML, GeomSampleBase, InformSegmentationSample, CellPhenotypeSampleBase, MaskSampleBase):
  rectangletype = InputCheckerRectangle
  multilayercomponenttiff = True

  def __init__(self, *args, checkintegrity=True, suppressinitwarnings=True, **kwargs):
    self.__checkintegrity = checkintegrity
    super().__init__(
      *args,
      layerscomponenttiff="setlater",
      suppressinitwarnings=suppressinitwarnings,
      **kwargs
    )
    self.setlayerscomponenttiff(
      layerscomponenttiff=[
        self.segmentationmembranelayer(seg) for seg in self.segmentationorder
      ] + [
        self.segmentationnucleuslayer(seg) for seg in self.segmentationorder
      ],
    )

  @property
  def checkintegrity(self): return self.__checkintegrity

  @property
  def segmentationorder(self):
    return sorted(
      self.segmentationids,
      key=lambda x: -2*(x=="Tumor")-(x=="Immune")
    )

  @classmethod
  def initkwargsfromargumentparser(cls, parsed_args_dict):
    parsed_args_dict["no_log"] = True
    return super().initkwargsfromargumentparser(parsed_args_dict)

  def run(self):
    if not self.annotationsxmlfile.exists():
      self.logger.warning("Missing HPF layout annotations file")
    else:
      self.logger.info("Checking im3s")
      tolog = []
      anyexist = False
      for r in self.rectangles:
        if not r.im3file.exists():
          tolog.append(f"Missing im3: {r.im3file}")
        else:
          anyexist = True
          if self.checkintegrity:
            try:
              with r.using_im3():
                pass
            except Exception:
              tolog.append(f"Corrupt im3: {r.im3file}")
      if not anyexist:
        tolog = ["ALL im3s are missing"] + [msg for msg in tolog if "Missing im3" not in msg]
      for msg in tolog:
        self.logger.warning(msg)

      tolog = []
      anyexist = False
      for r in self.rectangles:
        if not r.tissuemaskfile.exists():
          tolog.append(f"Missing mask: {r.tissuemaskfile}")
        else:
          anyexist = True
          try:
            with r.using_tissuemask():
              pass
          except Exception:
            tolog.append(f"Corrupt mask: {r.tissuemaskfile}")
      if not anyexist:
        tolog = ["ALL masks are missing"] + [msg for msg in tolog if "Missing mask" not in msg]
      for msg in tolog:
        self.logger.warning(msg)

      self.logger.info("Checking component tiffs")
      anynonsegmentedexist = False
      anysegmentedexist = False
      tolog = []
      for r in self.rectangles:
        nonsegmented = r.componenttifffile.with_name(r.componenttifffile.name.replace("_w_seg", ""))
        if not nonsegmented.exists():
          tolog.append(f"Missing component tiff: {r.componenttifffile}")
        else:
          anynonsegmentedexist = True
          if not r.componenttifffile.exists():
            tolog.append(f"Missing segmented component tiff: {r.componenttifffile}")
          else:
            anysegmentedexist = True
            if self.checkintegrity:
              try:
                with r.using_component_tiff():
                  pass
              except Exception:
                self.logger.warning(f"Corrupt segmented component tiff: {r.componenttifffile}")

      if not anynonsegmentedexist:
        tolog = ["ALL component tiffs are missing"] + [msg for msg in tolog if "Missing component tiff:" not in msg]
      elif not anysegmentedexist:
        tolog = [msg for msg in tolog if "Missing segmented component tiff:" not in msg] + ["ALL segmented component tiffs are missing"]
      for msg in tolog:
        self.logger.warning(msg)

      self.logger.info("Checking phenotype csvs")
      for r in self.rectangles:
        if not r.phenotypecsv.exists():
          self.logger.warning(f"Missing phenotype csv: {r.phenotypecsv}")

    if not self.qptifffilename.exists():
      self.logger.warning("Missing qptiff")

  @property
  def rectangleextrakwargs(self):
    return {
      **super().rectangleextrakwargs,
      "usememmap": True,
      "geomfolder": self.geomfolder,
      "phenotypefolder": self.phenotypefolder,
      "maskfolder": self.maskfolder,
    }

  @classmethod
  def getoutputfiles(cls, **kwargs): return []
  def inputfiles(self, **kwargs): return []
  @classmethod
  def logmodule(cls): return "slidesinputcheck"
  @classmethod
  def defaultim3filetype(cls): return "flatWarp"

  @classmethod
  def workflowdependencyclasses(cls):
    return [MeanImageSampleIm3Tissue]

class InputCheckerSampleTissue(InputCheckerSampleBase, TissueSampleBase):
  pass

class InputCheckerCohort(Im3Cohort, GeomFolderCohort, PhenotypeFolderCohort, MaskCohort):
  sampleclass = InputCheckerSampleTissue

  @classmethod
  def initkwargsfromargumentparser(cls, parsed_args_dict):
    parsed_args_dict["no_log"] = True
    return super().initkwargsfromargumentparser(parsed_args_dict)

  @classmethod
  def defaultim3filetype(cls): return "flatWarp"

def runsample(*args, **kwargs):
  return InputCheckerSampleTissue.runfromargumentparser(*args, **kwargs)
def runcohort(*args, **kwargs):
  return InputCheckerCohort.runfromargumentparser(*args, **kwargs)
