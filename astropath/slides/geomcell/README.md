# 6.9. Cell geometries

The `geomcell` step of the pipeline writes out the boundaries of the
segmented cells by reading from the component tiff.

To run `geomcell` on a single slide, run:
```
geomcellsample \\<Dpath>\<Dname> <SlideID>
```

To run `geomcell` on a whole cohort of slides, run:
```
geomcellcohort \\<Dpath>\<Dname>
```
(See [here](../../scans/docs/Definitions.md#43-definitions) for definitions
of the terms in `<angle brackets>`.)

To see more command line arguments, run `geomcellsample --help` or `geomcellcohort --help`.

The outputs will be in `\\<Dpath>\<Dname>\<SlideID>\geom`.  A csv file will
be created for each HPF containing information about the cells in that HPF.

`geomcell` does significant cleanup of the cells:
- in the membrane layers, sometimes the membrane is broken.  In the primary
  region of the HPF (defined by mx1, my1, mx2, my2), `geomcell` tries to
  connect the broken pieces.  Outside the primary region, _no repairs are done_,
  and the cell will likely be discarded by one of the subsequent cleanup steps.
- Membranes are filled in.
- Narrow tails coming off of a cell, with a width of one pixel, are discarded.
  If a narrow bridge connects two components, that bridge is also discarded.
- If there are disconnected components (including ones that are now disconnected
  after discarding a bridge), only the larger one is kept.
- If the cell's area is less than (3 microns)^2, it is discarded.
