class Workflow:
  cohorts = PrepdbCohort, AlignmentCohort, ZoomCohort, DeepZoomCohort, AnnoWarpCohort, GeomCohort, GeomCellCohort, CsvScanCohort

  def __init__(self, root, *args, **kwargs):
    self.root = root
    self.kwargs = kwargs

  def kwargs(self, cohortclass):
    return {
      kw: kwarg for kw, kwarg in kwargs.items()
      if issubclass(cohortclass, {
        "slideidfilters": Cohort,
        "samplefilters": Cohort,
        "debug": Cohort,
        "uselogfiles": Cohort,
        "logroot": Cohort,
        "xmlfolders": Cohort,
        "root2": Im3Cohort,
        "dbloadroot": DbloadCohort,
        "zoomroot": ZoomFolderCohort,
        "maskroot": MaskCohort,
        "selectrectangles": SelectRectanglesCohort,
        "layers": SelectLayersCohort,
        "temproot": TempDirCohort,
        "geomroot": GeomFolderCohort,
        "tilepixels": AnnoWarpCohortBase,
        "mintissuefraction": AnnoWarpCohortBase,
        
      }[kw])
    }
