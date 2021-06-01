# 6.7. Tumor and field geometry

The geom step of the pipeline writes out the boundaries of the HPF
primary regions and the boundaries of the tumor region determined
by inform to csv files.

To run `geom` on single slide, run:
```
geomsample \\<Dpath>\<Dname> <SlideID>
```

To run alignment on a whole cohort of slides, run:
```
geomcohort \\<Dpath>\<Dname>
```
(See [here](../../scans/docs/Definitions.md#43-definitions) for definitions
of the terms in `<angle brackets>`.)

To see more command line arguments, run `geomsample --help` or `geomcohort --help`.

The outputs will be in `\\<Dpath>\<Dname>\<SlideID>\dbload`.  The following files will
be created:
 - `<SlideID>_fieldGeometry.csv` contains the boundaries of each HPF primary region
 - `<SlideID>_tumorGeometry.csv` contains the boundaries of the tumor regions in each HPF.
