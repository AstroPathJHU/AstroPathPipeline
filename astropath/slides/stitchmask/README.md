# 6.6. Stitch mask

The `stitchmask` step of the pipeline creates a single, stitched mask
whole slide based on the output of the align step.  From each HPF, its
primary region as defined by (mx1, my1, mx2, my2) is inserted into the
large mask.

There are currently two types of masks that can be stitched: the Inform
mask, which is stored as the first layer of the `_w_seg` component tiff
after the image layers, and the [AstroPath tissue mask](../../shared/image_masking).

To run `stitchmask` on a single slide, run:
```
stitchinformmasksample \\<Dpath>\<Dname> <SlideID>
stitchastropathtissuemasksample \\<Dpath>\<Dname> <SlideID>
```

To run `stitchmask` on a whole cohort of slides, run:
```
stitchinformmaskcohort \\<Dpath>\<Dname>
stitchastropathtissuemaskcohort \\<Dpath>\<Dname>
```
(See [here](../../scans/docs/Definitions.md#43-definitions) for definitions
of the terms in `<angle brackets>`.)

To see more command line arguments, run the scripts with `--help`.

The outputs will be in `\\<Dpath>\<Dname>\<SlideID>\im3\meanimage\image_masking\`.
By default, the mask will be saved in `.npz` format, which does a good job of
compressing the data.  You can also save it in `.bin` format by specifying
`--mask-file-suffix .bin` on the command line.  In that case, you can read
the mask using the [`ImageMask.unpack_tissue_mask` function](../../shared/image_masking/image_mask.py#L150-L156)
or an equivalent routine in another language.
