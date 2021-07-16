# 6.7. Warp annotations

The `annowarp` step of the pipeline writes out the pathologist annotations,
which were drawn on the `qptiff` image, in `im3` coordinates.  It does this
by aligning tiles of the whole slide image, created in the [`zoom` step](../zoom#64-zoom),
to the qptiff and creating a warp map.

The alignment only uses tissue regions of the image, as determined by the
astropath mask.  For this reason, you have to first run the [`stitchmask` step](../stitchmask#66-stitch-mask)
before running annowarp.

To run `annowarp` on a single slide, run:
```
annowarpsample \\<Dpath>\<Dname> <SlideID> --zoomroot \\<Zpath>
```

To run `stitchmask` on a whole cohort of slides, run:
```
annowarpcohort \\<Dpath>\<Dname> --zoomroot \\<Zpath>
```
(See [here](../../scans/docs/Definitions.md#43-definitions) for definitions
of the terms in `<angle brackets>`.)

To see more command line arguments, run the scripts with `--help`.

The outputs are in the `\\<Dpath>\<Dname>\<SlideID>\dbload` folder.
They include:
- `<SlideID>_annotations.csv` - metadata about the annotations
- `<SlideID>_annowarp.csv` - alignment results for each tile of the wsi and qptiff
- `<SlideID>_annowarp-stitch.csv` - the result of the stitching
- `<SlideID>_vertices.csv` - the vertices of each of the annotations, including the original and warped coordinates
- `<SlideID>_regions.csv` - the polygons of each of the annotations
