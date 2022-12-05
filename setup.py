import os, csv, pathlib, setuptools.command.build_py, setuptools.command.develop, site, subprocess

site.ENABLE_USER_SITE = True #https://www.scivision.dev/python-pip-devel-user-install/
here = pathlib.Path(__file__).parent

class build_commits_csv(setuptools.Command):
  user_options = []
  def initialize_options(self): pass
  def finalize_options(self): pass

  def run(self):
    with open(here/"astropath"/"utilities"/"version"/"commits.csv", "w", newline="") as f:
      writer = csv.DictWriter(f, ["hash", "parents", "tags"], lineterminator='\r\n')
      writer.writeheader()
      for line in subprocess.run(["git", "log", "--all", "--pretty=%H\t%P\t%D", "--no-abbrev-commit"], cwd=here, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding="ascii").stdout.split("\n"):
        if not line.strip(): continue
        hash, parents, tags = line.split("\t")
        parents = parents.split()
        tags = tags.replace(",", "").replace("->", "").split()
        writer.writerow({"hash": hash, "parents": " ".join(parents), "tags": " ".join(tags)})

class build_py(setuptools.command.build_py.build_py):
  def run(self):
    self.run_command("build_commits_csv")
    super().run()

class develop(setuptools.command.develop.develop):
  def run(self):
    self.run_command("build_commits_csv")
    super().run()

def get_nnunet_package_files():
  directory = here/'astropath'/'slides'/'segmentation'/'nnunet_models'
  paths = []
  for (path, directories, filenames) in os.walk(directory):
    for filename in filenames:
      paths.append(os.path.join('..', path, filename))
  return paths

setupkwargs = dict(
  name = "astropath",
  packages = setuptools.find_packages(include=["astropath*"]),
  cmdclass={
    'build_commits_csv': build_commits_csv,
    'build_py': build_py,
    'develop': develop,
  },
  entry_points = {
    "console_scripts": [
      "aligncohort=astropath.slides.align.aligncohort:main",
      "alignsample=astropath.slides.align.alignsample:main",
      "annowarpcohort=astropath.slides.annowarp.annowarpcohort:main",
      "annowarpsample=astropath.slides.annowarp.annowarpsample:main",
      "appliedflatfieldcohort=astropath.hpfs.flatfield.appliedflatfieldcohort:main",
      "appliedflatfieldcohortcomponenttiff=astropath.hpfs.flatfield.appliedflatfieldcohort:appliedflatfieldcohortcomponenttiff",
      "applyflatwsample=astropath.hpfs.imagecorrection.applyflatwsample:main",
      "applyflatwcohort=astropath.hpfs.imagecorrection.applyflatwcohort:main",
      "ast-gen=astropath.scans.astroidgen.ASTgen:start_gen",
      "batchflatfieldmulticohort=astropath.hpfs.flatfield.batchflatfieldmulticohort:main",
      "checkannotations=astropath.shared.annotationpolygonxmlreader:checkannotations",
      "copyannotationinfocohort=astropath.slides.annotationinfo.annotationinfo:copyannotationinfocohort",
      "copyannotationinfosample=astropath.slides.annotationinfo.annotationinfo:copyannotationinfosample",
      "csvscancohort=astropath.slides.csvscan.csvscancohort:main",
      "csvscansample=astropath.slides.csvscan.csvscansample:main",
      "deepzoomcohort=astropath.slides.deepzoom.deepzoomcohort:main",
      "deepzoomsample=astropath.slides.deepzoom.deepzoomsample:main",
      "find_slide_overexposed_hpfs=astropath.hpfs.overexposedhpfs.find_slide_overexposed_hpfs:main",
      "fixfw01cohort=astropath.hpfs.fixfw01.fixfw01cohort:main_xml",
      "fixfw01cohortdbload=astropath.hpfs.fixfw01.fixfw01cohort:main_dbload",
      "fixfw01sample=astropath.hpfs.fixfw01.fixfw01sample:main_xml",
      "fixfw01sampledbload=astropath.hpfs.fixfw01.fixfw01sample:main_dbload",
      "geomcellcohortdeepcell=astropath.slides.geomcell.geomcellcohort:deepcell",
      "geomcellcohortinform=astropath.slides.geomcell.geomcellcohort:inform",
      "geomcellcohortmesmer=astropath.slides.geomcell.geomcellcohort:mesmer",
      "geomcellsampledeepcell=astropath.slides.geomcell.geomcellsample:deepcell",
      "geomcellsampleinform=astropath.slides.geomcell.geomcellsample:inform",
      "geomcellsamplemesmer=astropath.slides.geomcell.geomcellsample:mesmer",
      "motifcsvscan=astropath.slides.csvscan.motifcsvscan:main",
      "motifdeepzoom=astropath.slides.deepzoom.motifdeepzoom:main",
      "motifgeomcell=astropath.slides.geomcell.motifgeomcell:main",
      "motifprepdb=astropath.slides.prepdb.motifprepdb:main",
      "prepdbcohort=astropath.slides.prepdb.prepdbcohort:main",
      "prepdbsample=astropath.slides.prepdb.prepdbsample:main",
      "read_annotation_xml=astropath.shared.annotationpolygonxmlreader:writeannotationcsvs",
      "makesampledef=astropath.shared.samplemetadata:makesampledef",
      "meanimagesample=astropath.hpfs.flatfield.meanimagesample:main",
      "meanimagecohort=astropath.hpfs.flatfield.meanimagecohort:main",
      "meanimagecomparison=astropath.hpfs.flatfield.meanimagecomparison:main",
      "meanimagesamplecomponenttiff=astropath.hpfs.flatfield.meanimagesample:meanimagesamplecomponenttiff",
      "mergeannotationxmlscohort=astropath.slides.annotationinfo.annotationinfo:mergeannotationxmlscohort",
      "mergeannotationxmlssample=astropath.slides.annotationinfo.annotationinfo:mergeannotationxmlssample",
      "segmentationsampledeepcell=astropath.slides.segmentation.segmentationsampledeepcell:main",
      "segmentationsamplemesmercomponenttiff=astropath.slides.segmentation.segmentationsamplemesmer:segmentationsamplemesmercomponenttiff",
      "segmentationsamplemesmerwithihc=astropath.slides.segmentation.segmentationsamplemesmer:segmentationsamplemesmerwithihc",
      "segmentationsamplennunet=astropath.slides.segmentation.segmentationsamplennunet:main",
      "segmentationcohortdeepcell=astropath.slides.segmentation.segmentationcohort:segmentationcohortdeepcell",
      "segmentationcohortmesmercomponenttiff=astropath.slides.segmentation.segmentationcohort:segmentationcohortmesmercomponenttiff",
      "segmentationcohortmesmerwithihc=astropath.slides.segmentation.segmentationcohort:segmentationcohortmesmerwithihc",
      "segmentationcohortnnunet=astropath.slides.segmentation.segmentationcohort:segmentationcohortnnunet",
      "stitchastropathtissuemaskcohort=astropath.slides.stitchmask.stitchmaskcohort:astropathtissuemain",
      "stitchastropathtissuemasksample=astropath.slides.stitchmask.stitchmasksample:astropathtissuemain",
      "stitchihctissuemaskcohort=astropath.slides.stitchmask.stitchmaskcohort:ihcmain",
      "stitchihctissuemasksample=astropath.slides.stitchmask.stitchmasksample:ihcmain",
      "stitchinformmaskcohort=astropath.slides.stitchmask.stitchmaskcohort:informmain",
      "stitchinformmasksample=astropath.slides.stitchmask.stitchmasksample:informmain",
      "tenxdeepzoom=astropath.slides.deepzoom.tenxdeepzoom:main",
      "tenxgeomcell=astropath.slides.geomcell.tenxgeomcell:main",
      "transfer-daemon=astropath.scans.transferdaemon.Daemon:launch_transfer",
      "warpingsample=astropath.hpfs.warping.warpingsample:main",
      "warpingmulticohort=astropath.hpfs.warping.warpingmulticohort:main",
      "writeannotationinfo=astropath.shared.annotationpolygonxmlreader:writeannotationinfo",
      "writeannotationinfocohort=astropath.slides.annotationinfo.annotationinfo:writeannotationinfocohort",
      "writeannotationinfosample=astropath.slides.annotationinfo.annotationinfo:writeannotationinfosample",
      "zoomcohort=astropath.slides.zoom.zoomcohort:main",
      "zoomsample=astropath.slides.zoom.zoomsample:main",
    ],
  },
  setup_requires = [
    
  ],
  install_requires = [
    "batchgenerators",
    "contextlib2>=0.6.0; python_version < '3.7'",
    "dataclassy>=0.10.0",
    "imagecodecs",
    "integv",
    "jxmlease>=1.0.2dev1",
    "matplotlib>=3.3.2",
    "methodtools",
    "more_itertools>=8.3.0",
    "networkx",
    "numba>=0.54", #require np.clip, added in numba/numba#6808
    "numpy>=1.23.0",
    "opencv-python",
    "openpyxl",
    "pathos>=0.2.8",
    "psutil;sys_platform!='cygwin'", #please note astropath is NOT been tested on cygwin
    "pyopencl",
    "rdp",
    "reikna",
    "seaborn",
    "scikit-image>=0.18",
    "scikit-learn>=0.17",
    "scipy>=0.12",
    "setuptools-scm",
    "SimpleITK>=2.1.1",
    "slurm-python-utils>=1.7",
    "uncertainties",
    "gpu": ["pyopencl", "reikna"],
  ],
  extras_require = {
    "deepcell": ["deepcell>=0.12.0"],
    "gdal": ["gdal>=3.3.0"],
    "nnunet": ["nnunet>=1.6.0"],
    "test": [
      #packages needed for running tests
      "beautifulsoup4",
      "cvxpy",
      "flake8",
      "gitpython",
      "lxml",
      "marko[toc]",
      "pyflakes",
      "texoutparse",
      #packages that are standard dependencies, but
      #a specific version is needed for reference comparison
      "deepcell==0.12.3",
    ],
    "vips": ["pyvips"],
  },
  package_data = {
    "astropath": [
      "shared/master_annotation_list.csv",
      "slides/zoom/color_matrix_8.txt",
      "slides/zoom/color_matrix_9.txt",
      "utilities/version/commits.csv",
    ]+get_nnunet_package_files(),
  },
)

setupkwargs["extras_require"]["all"] = sum(setupkwargs["extras_require"].values(), [])

setuptools.setup(**setupkwargs)
