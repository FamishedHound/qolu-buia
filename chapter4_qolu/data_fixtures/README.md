# Local raster fixtures (not tracked)

Drop a UKCEH Land Cover Map "1 km percentage cover, aggregate class" GeoTIFF in
this directory and the test-suite will pick it up automatically (first `*.tif`
in alphabetical order).

Resolution order used by `chapter4_qolu/tests/conftest.py`:

1. the `CHAPTER4_LCM_TIF` environment variable;
2. the first `*.tif` in this directory;
3. otherwise the raster-dependent tests skip with an explanatory message.

A small product is enough for the tests, for example *Land Cover Map 2015 (1km
percentage aggregate class, N. Ireland)*, whose UKCEH download
(`362feaea-0ccf-4a45-b11f-980c6b89a858.zip`) contains
`data/lcm2015_ni_1km_percent_cover_aggregate_class.tif`.  The
Northern-Ireland product is a **placeholder**: it is not the study region of
the paper (south-east UK).
`load_lcm_raster` logs and raises a `UserWarning` whenever the file name looks
like an NI product, so the substitution cannot pass unnoticed.

Rasters are deliberately not committed: the UKCEH licence is non-commercial /
personal use and the files are better fetched from
<https://catalogue.ceh.ac.uk/> by whoever runs the experiments.
