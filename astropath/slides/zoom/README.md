# 6.4. Zoom

The zoom step of the pipeline creates a single, stitched image for the
whole slide based on the output of the align step.  From each HPF, its
primary region as defined by (mx1, my1, mx2, my2) is taken from the
component tiff and inserted into the large image.

To run `zoom` on a single slide, run:
```
zoomsample \\<Dpath>\<Dname> <SlideID> --zoomroot \\<Zpath>
```

To run `zoom` on a whole cohort of slides, run:
```
zoomcohort \\<Dpath>\<Dname> --zoomroot \\<Zpath>
```
(See [here](../../scans/docs/Definitions.md#43-definitions) for definitions
of the terms in `<angle brackets>`.)

To see more command line arguments, run `zoomsample --help` or `zoomcohort --help`.

The outputs will be in `\\<Zpath>\<SlideID>\wsi\`, which will contain one file
for each unmixed layer.  The files are named `<SlideID>-Z9-L<layer>-wsi.png`,
where Z9 refers to the fact that this is the most zoomed in version of the image.
The zoomed out versions will be created in the `deepzoom` step.
