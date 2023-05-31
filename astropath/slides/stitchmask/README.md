# 6.4. Stitch mask

The `stitchmask` step of the pipeline creates a single, stitched mask
whole slide based on the output of the align step.  From each HPF, its
primary region as defined by (mx1, my1, mx2, my2) is inserted into the
large mask.

As part of the AstroPath pipeline, we use the tissue mask
produced by the [meanimage](../../hpfs/flatfield) step of the pipeline.

To run `stitchmask` on a single slide, run:
```
stitchastropathtissuemasksample \\<Dpath>\<Dname> <SlideID>
```

To run `stitchmask` on a whole cohort of slides, run:
```
stitchastropathtissuemaskcohort \\<Dpath>\<Dname>
```
(See [here](../../scans/docs/Definitions.md#43-definitions) for definitions
of the terms in `<angle brackets>`.)

To see more command line arguments, run the scripts with `--help`.

The outputs will be in `\\<Dpath>\<Dname>\<SlideID>\im3\meanimage\image_masking\<SlideID>_tissue_mask.npz`.

You can also save it in `.bin` format by specifying
`--mask-file-suffix .bin` on the command line.  In that case, you can read
the mask using the [`ImageMask.unpack_tissue_mask` function](../../shared/image_masking/image_mask.py#L169-L175) 
or an equivalent routine in another language.  Also, you can use `stitchinformmasksample` or `stitchinformmaskcohort`
to stitch the inform mask from the `_w_seg` component tiff.
