import csv, pathlib, setuptools.command.build_py, setuptools.command.develop, site, subprocess

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
      "applyflatwsample=astropath.hpfs.imagecorrection.applyflatwsample:main",
      "applyflatwcohort=astropath.hpfs.imagecorrection.applyflatwcohort:main",
      "ast-gen=astropath.scans.astroidgen.ASTgen:start_gen",
      "astropathworkflow=astropath.shared.workflow:main",
      "batchflatfieldmulticohort=astropath.hpfs.flatfield.batchflatfieldmulticohort:main",
      "checkannotations=astropath.shared.annotationpolygonxmlreader:checkannotations",
      "csvscancohort=astropath.slides.csvscan.csvscancohort:main",
      "csvscansample=astropath.slides.csvscan.csvscansample:main",
      "deepzoomcohort=astropath.slides.deepzoom.deepzoomcohort:main",
      "deepzoomsample=astropath.slides.deepzoom.deepzoomsample:main",
      "find_slide_overexposed_hpfs=astropath.hpfs.overexposedhpfs.find_slide_overexposed_hpfs:main",
      "geomcohort=astropath.slides.geom.geomcohort:main",
      "geomsample=astropath.slides.geom.geomsample:main",
      "geomcellcohort=astropath.slides.geomcell.geomcellcohort:main",
      "geomcellsample=astropath.slides.geomcell.geomcellsample:main",
      "prepdbcohort=astropath.slides.prepdb.prepdbcohort:main",
      "prepdbsample=astropath.slides.prepdb.prepdbsample:main",
      "read_annotation_xml=astropath.shared.annotationpolygonxmlreader:main",
      "makesampledef=astropath.shared.samplemetadata:makesampledef",
      "meanimagesample=astropath.hpfs.flatfield.meanimagesample:main",
      "meanimagecohort=astropath.hpfs.flatfield.meanimagecohort:main",
      "meanimagecomparison=astropath.hpfs.flatfield.meanimagecomparison:main",
      "stitchastropathtissuemasksample=astropath.slides.stitchmask.stitchmasksample:astropathtissuemain",
      "stitchinformmasksample=astropath.slides.stitchmask.stitchmasksample:informmain",
      "stitchastropathtissuemaskcohort=astropath.slides.stitchmask.stitchmaskcohort:astropathtissuemain",
      "stitchinformmaskcohort=astropath.slides.stitchmask.stitchmaskcohort:informmain",
      "transfer-daemon=astropath.scans.transferdaemon.Daemon:launch_transfer",
      "warpingsample=astropath.hpfs.warping.warpingsample:main",
      "warpingcohort=astropath.hpfs.warping.warpingcohort:main",
      "zoomcohort=astropath.slides.zoom.zoomcohort:main",
      "zoomsample=astropath.slides.zoom.zoomsample:main",
    ],
  },
  setup_requires = [
    
  ],
  install_requires = [
    "contextlib2>=0.6.0; python_version < '3.7'",
    "cvxpy",
    "dataclassy>=0.10.0",
    "imagecodecs",
    "jxmlease>=1.0.2dev1",
    "matplotlib>=3.3.2",
    "methodtools",
    "more_itertools>=8.3.0",
    "networkx",
    "numba",
    "numpy>=1.17.0",
    "opencv-python",
    "openpyxl",
    "psutil;sys_platform!='cygwin'", #please note astropath is NOT been tested on cygwin
    "pyopencl",
    "reikna",
    "seaborn",
    "scikit-image>=0.17,<0.18", #see test/testmisc.py - we want polygon.numpyarray to return reproducible results, and skimage.draw.polygon's behavior changes between 0.17 and 0.18.  Want to support python 3.6 for now so we need to stick to 0.17.
    "scikit-learn>=0.17",
    "scipy>=0.12",
    "setuptools-scm",
    "slurm-python-utils>=1.2.14",
    "uncertainties",
  ],
  extras_require = {
    "test": ["beautifulsoup4", "flake8", "gitpython", "lxml", "marko[toc]", "pyflakes", "texoutparse"],
    "gdal": ["gdal>=3.3.0"],
    "vips": ["pyvips"],
  },
  package_data = {
    "astropath": [
      "shared/master_annotation_list.csv",
      "slides/zoom/color_matrix.txt",
      "utilities/version/commits.csv",
    ],
  },
)

setupkwargs["extras_require"]["all"] = sum(setupkwargs["extras_require"].values(), [])

setuptools.setup(**setupkwargs)
