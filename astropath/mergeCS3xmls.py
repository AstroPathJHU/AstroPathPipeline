import os, pathlib, re
from astropath.slides.annowarp.mergeannotationxmls import MergeAnnotationXMLsCohort

root = pathlib.Path(r'\\bki04\Clinical_Specimen_3')
for folder in root.glob("MA*"):
  SlideID = folder.name
  if SlideID in ("MA69", "MA77", "MA80", "MA82"): continue
  logfile = folder/"logfiles"/f"{SlideID}-align.log"
  position = 0, 0
  try:
    with open(logfile) as f:
      for line in f:
        if "Some HPFs have" in line and "+ margin" not in line:
          regex = r"Some HPFs have \(x, y\) < \(xposition, yposition\), shifting the whole slide by \(([0-9.-]+), ([0-9.-]+)\)"
          match = re.search(regex, line)
          position = float(match.group(1)), float(match.group(2))
  except FileNotFoundError:
    pass
  regex1 = "MA[0-9]+_Scan[0-9]+[.]annotations.polygons.xml"
  regex2 = "M[0-9]+_[0-9]+-MA[0-9]+_annotations[.]polygons_x[.]xml"
  xmlfolder, = (root/SlideID/"im3").glob("Scan*")
  xmlfiles = list(xmlfolder.glob("*.xml"))
  assert any(re.match(regex1, f.name) for f in xmlfiles), xmlfolder
  if not any(re.match(regex2, f.name) for f in xmlfiles):
    continue
  args = [os.fspath(root), "--sampleregex", f"^{SlideID}$", "--debug", "--annotation", "good tissue", regex1, "--annotation", "tumor x", regex2, "--skip-annotation", "good tissue x", "--annotations-on-wsi", "--annotation-position", *(str(_) for _ in position)]
  MergeAnnotationXMLsCohort.runfromargumentparser(args)
