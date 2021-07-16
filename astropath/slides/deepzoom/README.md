# 6.5. Deepzoom

The `deepzoom` step of the pipeline runs on the output of `zoom` and creates
image tiles at various zoom levels, each one more zoomed out than the previous
one.  These image tiles will be used to create the viewable images in cellview.

To run `deepzoom` on a single slide, run:
```
deepzoomsample \\<Dpath>\<Dname> <SlideID> --zoomroot \\<Zpath> --deepzoomroot \\<DZpath>
```

To run `deepzoom` on a whole cohort of slides, run:
```
deepzoomcohort \\<Dpath>\<Dname> --zoomroot \\<Zpath> --deepzoomroot \\<DZpath>
```
(See [here](../../scans/docs/Definitions.md#43-definitions) for definitions
of the terms in `<angle brackets>`.)

To see more command line arguments, run `deepzoomsample --help` or `deepzoomcohort --help`.

The outputs will be in `\\<DZpath>\<SlideID>\`.  `zoomlist.csv` contains
a list of all the images, where each image is 256x256 pixels and contains
a tile of the zoomed image.  `Z0` is the most zoomed out and `Z9` is the
most zoomed in.
