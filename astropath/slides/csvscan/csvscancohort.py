import re
from ...baseclasses.cohort import GeomFolderCohort, GlobalDbloadCohort, PhenotypeFolderCohort, SelectRectanglesCohort, WorkflowCohort
from ...baseclasses.csvclasses import ClinicalInfo, ControlCore, ControlFlux, ControlSample, GlobalBatch, MergeConfig
from ...baseclasses.sample import SampleDef
from .csvscansample import CsvScanBase, CsvScanSample

class CsvScanCohort(GlobalDbloadCohort, GeomFolderCohort, PhenotypeFolderCohort, SelectRectanglesCohort, WorkflowCohort, CsvScanBase):
  sampleclass = CsvScanSample
  __doc__ = sampleclass.__doc__

  def runsample(self, sample):
    return sample.runcsvscan()

  def run(self, **kwargs):
    super().run(**kwargs)
    with self.globallogger():
      self.makeglobalcsv()

  @property
  def globalcsvs(self):
    sampledefs = self.readtable(self.root/"sampledef.csv", SampleDef)
    slideids = [s.SlideID for s in sampledefs]
    for folder in {self.root, self.dbloadroot, self.geomroot, self.phenotyperoot}:
      for subfolder in folder.iterdir():
        if subfolder.name.endswith(".csv"): yield subfolder
        if not subfolder.is_dir(): continue
        if subfolder.name in slideids: continue
        yield from subfolder.rglob("*.csv")

  def makeglobalcsv(self):
    toload = []
    expectcsvs = {
      self.root/"Batch"/f"{csv}_{s.BatchID:02d}.csv"
      for csv in ("MergeConfig", "Batch")
      for s in self.sampledefs
    } | {
      csv
      for csv in (self.root/"Clinical").glob(f"Clinical_Table_Specimen_{self.Cohort}_*.csv")
      if re.match(f"Clinical_Table_Specimen_{self.Cohort}_[0-9]+.csv", csv.name)
    } | {
      self.root/"Ctrl"/f"project{self.Project}_ctrl{ctrl}.csv"
      for ctrl in ("cores", "fluxes", "samples")
    }
    ctrlsamplescsv = self.root/"Ctrl"/f"project{self.Project}_ctrlsamples.csv"
    try:
      controlcsvs = {
        self.root/sample.SlideID/"dbload"/f"{sample.SlideID}_control.csv"
        for sample in self.readtable(ctrlsamplescsv, ControlSample)
      }
    except IOError:
      controlcsvs = set()
    expectcsvs |= controlcsvs
    optionalcsvs = set()
    unknowncsvs = set()
    for csv in self.globalcsvs:
      if csv == self.csv("loadfiles"):
        continue
      if csv == self.root/"sampledef.csv":
        continue

      try:
        expectcsvs.remove(csv)
      except KeyError:
        try:
          optionalcsvs.remove(csv)
        except KeyError:
          unknowncsvs.add(csv)
          continue

      if csv.parent == self.root/"Batch":
        match = re.match("(.*)_[0-9]+[.]csv", csv.name)
        csvclass, tablename = {
          "Batch": (GlobalBatch, "Batch"),
          "MergeConfig": (MergeConfig, "MergeConfig")
        }[match.group(1)]
        extrakwargs = {}
        idx = 1
      elif csv.parent == self.root/"Clinical":
        csvclass = ClinicalInfo
        tablename = "Clinical"
        extrakwargs = {}
        idx = 2
      elif csv.parent == self.root/"Ctrl":
        match = re.match(f"project{self.Project}_(.*)[.]csv", csv.name)
        csvclass, tablename = {
          "ctrlcores": (ControlCore, "Ctrlcores"),
          "ctrlfluxes": (ControlFlux, "Ctrlfluxes"),
          "ctrlsamples": (ControlSample, "Ctrlsamples"),
        }[match.group(1)]
        extrakwargs = {}
        idx = 3
      elif csv in controlcsvs:
        continue
      else:
        assert False, csv

      toload.append({"csv": csv, "csvclass": csvclass, "tablename": tablename, "extrakwargs": extrakwargs, "SlideID": f"project{self.Project}-{idx}"})
      toload.sort(key=lambda x: (x["csv"]))

    if expectcsvs or unknowncsvs:
      errors = []
      if expectcsvs:
        errors.append("Some csvs are missing: "+", ".join(str(_) for _ in sorted(expectcsvs)))
      if unknowncsvs:
        errors.append("Unknown csvs: "+", ".join(str(_) for _ in sorted(unknowncsvs)))
      raise ValueError("\n".join(errors))

    loadfiles = [self.processcsv(**kwargs) for kwargs in toload]

    self.dbload.mkdir(exist_ok=True)
    self.writecsv("loadfiles", loadfiles, header=False)

def main(args=None):
  CsvScanCohort.runfromargumentparser(args)

if __name__ == "__main__":
  main()
