# 6.7. Annotation info

The `writeannotationinfo` and `copyannotationinfo` steps of the pipeline
write metadata about the annotations.  The information written in these
steps is needed for the subsequent `annowarp` step, which records the position
of the annotations in `im3` coordinates.

# 6.7.1. Write annotation info

The first step, `writeannotationinfo`, writes a `csv` file corresponding to
the annotation `xml` file.  It contains the hash of the `xml` file as well as
the type of image the annotations were drawn on: the `qptiff` or `wsi`. If the
annotations were drawn on the `wsi`, this file also records the position of
the wsi when the annotations were drawn, in case a later version of the `wsi`
has a different shift.

All of this information needs to be provided on the command line.
For a single slide:
```
writeannotationinfosample \\<Dpath>\<Dname> <SlideID> --annotations-on-qptiff
writeannotationinfosample \\<Dpath>\<Dname> <SlideID> --annotations-on-wsi --annotation-position 100 200  #annotation position (x, y) given in pixels
writeannotationinfosample \\<Dpath>\<Dname> <SlideID> --annotations-on-wsi --annotation-position-from-affine-shift  #annotations were drawn on the wsi in its current position
```
or for a whole cohort:
```
writeannotationinfocohort \\<Dpath>\<Dname> --annotations-on-qptiff
writeannotationinfocohort \\<Dpath>\<Dname> --annotations-on-wsi --annotation-position 100 200  #annotation position (x, y) given in pixels
writeannotationinfocohort \\<Dpath>\<Dname> --annotations-on-wsi --annotation-position-from-affine-shift  #annotations were drawn on the wsi in its current position
```

If there are multiple annotation xml files in the `im3/ScanX` folder, you need
to indicate which one is the desired one by including the `--annotations-xml-regex`
on the command line.  This regex should match exactly one of the files.  If you
later want to take annotations from two or more files, you have to run
`writeannotationinfo` for each of them.

The output of this step is a `csv` file in the `im3/ScanX` folder, with the same
name as the `xml` file but with the suffix `.annotationinfo.csv`.

# 6.7.2. Copy annotation info

The `copyannotationinfo` step copies the `annotationinfo.csv` from the previous
step into `<SlideID>_annotationinfo.csv` in the `dbload` folder.  The steps are
separated so that later versions of the annotation xmls, with their own info files,
can be created and used, while keeping a record of the original annotation xml
and its info.

To run:
```
copyannotationinfosample \\<Dpath>\<Dname> <SlideID>
copyannotationinfocohort \\<Dpath>\<Dname>
```
As before, if there are multiple annotation xml files, you can provide the
`--annotations-xml-regex` argument to choose between them.

If you want to take annotations from multiple different xml files, instead run the
`mergeannotationxmlssample` or `mergeannotationxmlscohort` commands:
```
mergannotationxmlssample \\<Dpath>\<Dname> <SlideID> --annotation "good tissue" ".*regex1[.]xml" --annotation "tumor" ".*regex2[.]xml" --skip-annotation "other annotation"
mergannotationxmlscohort \\<Dpath>\<Dname> --annotation "good tissue" ".*regex1[.]xml" --annotation "tumor" ".*regex2[.]xml" --skip-annotation "other annotation"
```
This will also create `dbload/<SlideID>_annotationinfo.csv` with the selected
annotations from each xml file.
