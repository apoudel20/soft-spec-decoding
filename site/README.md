# Project page sources

The shared project page is generated from these files. `template.html` is the page with a `__SUMMARY__`
placeholder that is replaced by `summary.json` (the failure-explorer data) to produce the published page.
The figures are built by the `make_*.py` scripts and spliced in by the matching `apply_*.py` scripts.
`examples.json` (the failure-explorer examples, about 450 KB) is published alongside the page and is not kept here.
