# 6.2. PrepDb

The prepdb stage of the pipeline extracts metadata for a sample from the `.xml` files
and writes it out to `.csv` files.

To run prepdb on a single slide, run:
```
prepdbsample \\<Dpath>\<Dname> <SlideID>
```

To run alignment on a whole cohort of slides, run:
```
prepdbcohort \\<Dpath>\<Dname>
```

(See [here](../../scans/docs/Definitions.md#43-definitions) for definitions of the terms in `<angle brackets>`.)

To see more command line arguments, run `prepdbsample --help` or `prepdbcohort --help`.

The outputs will be in `\\<Dpath>\<Dname>\<SlideID>\dbload`.  The following files will
be created:
 - `<SlideID>_batch.csv` gives basic bookkeeping information about the sample scan.
 - `<SlideID>_constants.csv` gives basic information about the microscope's scan, such as the position of the tissue within the slide and the pixels/micron scale of the microscope.
 - `<SlideID>_qptiff.csv` gives basic information from the large qptiff image.
 - `<SlideID>_qptiff.jpg` is a thumbnail image constructed from the qptiff image.
 - `<SlideID>_annotations.csv` describes the annotations for tissue, tumor, or other regions of interest.
 - `<SlideID>_regions.csv` describes the annotations in more detail and will be modified further in the annowarp stage to fully give the boundaries of each annotation.
 - `<SlideID>_vertices.csv` also describes the annotations by giving each vertex of each annotation aboundary.
 - `<SlideID>_rect.csv` gives information about each HPFs.
 - `<SlideID>_overlap.csv` gives the overlaps between adjacent HPFs.
 - `<SlideID>_exposures.csv` gives the exposure time for each layer of each HPF.
