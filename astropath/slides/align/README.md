# Alignment

This document describes how to technically run alignment.  For a more detailed mathematical
explanation, see [this document](README.pdf).

To run alignment on single slide, run:
```
alignsample root/path/of/samples root/path/of/images SlideID
```

To run alignment on a whole cohort of slides, run:
```
aligncohort root/path/of/samples root/path/of/images
```

To see more command line arguments, run `alignsample --help` or `aligncohort --help`.

The outputs will be in `root/path/of/samples/SlideID/dbload/`.  The following files will
be created:
 - `SlideID_imstat.csv` contains statistics about the pixel fluxes in each HPF.
 - `SlideID_align.csv` contains the results of the individual alignments.
 - `SlideID_fields.csv` contains the final stitched positions of the HPFs.
 - `SlideID_fieldoverlaps.csv` contains the covariance matrix elements for the corresponding to the positions of adjacent HPFs, which can be used to get the error on the relative positions of those HPFs.
 - `SlideID_affine.csv` contains the affine matrix and its error.
