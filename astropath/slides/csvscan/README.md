# 6.10. csvscan

csvscan is the last step in the preparation for loading the database.
It compiles all the csv files created in the other steps into a single
csv that drives the database load.

To run `csvscan` on a single slide, run:
```
csvscansample \\<Dpath>\<Dname> <SlideID>
```

To run `csvscan` on a whole cohort of slides, run:
```
csvscancohort \\<Dpath>\<Dname>
```
(See [here](../../scans/docs/Definitions.md#43-definitions) for definitions
of the terms in `<angle brackets>`.)

To see more command line arguments, run `csvscansample --help` or `csvscancohort --help`.

The output is `\\<Dpath>\<Dname>\<SlideID>\dbload\<SlideID>_loadfiles.csv`.

Additionally, when running a cohort, `\\<Dpath>\<Dname>\dbload\<SlideID>_loadfiles.csv`.
This file lists the global csv files that get loaded into the database.
