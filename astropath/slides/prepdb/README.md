# 6.2. PrepDb

The prepdb stage of the pipeline extracts metadata for a sample from the `.xml` files
and writes it out to `.csv` files.

To run prepdb on single slide, run:
```
prepdbsample root/path/of/samples SlideID
```

To run alignment on a whole cohort of slides, run:
```
prepdbcohort root/path/of/samples
```

To see more command line arguments, run `prepdbsample --help` or `prepdbcohort --help`.

The outputs will be in `root/path/of/samples/SlideID/dbload/`.  The following files will
be created:
 - `SlideID_batch.csv` gives basic bookkeeping information about the sample scan.
 - `SlideID_constants.csv` gives basic information about the microscope's scan, such as the position of the tissue within the slide and the pixels/micron scale of the microscope.
 - `SlideID_qptiff.csv` gives basic information from the large qptiff image.
 - `SlideID_qptiff.jpg` is a thumbnail image constructed from the qptiff image.
 - `SlideID_annotations.csv` describes the annotations for tissue, tumor, or other regions of interest.
 - `SlideID_regions.csv` describes the annotations in more detail and will be modified further in the annowarp stage to fully give the boundaries of each annotation.
 - `SlideID_vertices.csv` also describes the annotations by giving each vertex of each annotation aboundary.
 - `SlideID_rect.csv` gives information about each HPFs.
 - `SlideID_overlap.csv` gives the overlaps between adjacent HPFs.
 - `SlideID_exposures.csv` gives the exposure time for each layer of each HPF.
