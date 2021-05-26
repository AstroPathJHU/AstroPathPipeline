# 6.3. Align

This document describes how to technically run the align stage of the pipeline.
For a more detailed mathematical explanation, see [this document](README.pdf).

To run alignment on single slide, run:
```
alignsample \\<Dpath>\<Dname> \\<FWpath>\<Dname> <SlideID>
```

To run alignment on a whole cohort of slides, run:
```
aligncohort \\<Dpath>\<Dname> \\<FWpath>\<Dname>
```

(See [here](../../scans/docs/Definitions.md#43-definitions) for definitions of the terms in `<angle brackets>`.)

To see more command line arguments, run `alignsample --help` or `aligncohort --help`.

The outputs will be in `\\<Dpath>\<Dname>\<SlideID>\dbload`.  The following files will
be created:
 - `<SlideID>_imstat.csv` contains statistics about the pixel fluxes in each HPF.
 - `<SlideID>_align.csv` contains the results of the individual alignments.
 - `<SlideID>_fields.csv` contains the final stitched positions of the HPFs.
 - `<SlideID>_fieldoverlaps.csv` contains the covariance matrix elements for the corresponding to the positions of adjacent HPFs, which can be used to get the error on the relative positions of those HPFs.
 - `<SlideID>_affine.csv` contains the affine matrix and its error.
