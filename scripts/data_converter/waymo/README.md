# NCore Waymo

Data conversion specification and tooling for converting waymo-open data to NCore.

## Converting Waymo Data to NCore

`//scripts/data_converter/waymo` converts waymo-open data to NCore and expects `.tfrecords` as input.

Run the following command

```
bazel run  //scripts/data_converter/waymo -- --root-dir <directory-to-input-tfrecords> --output-dir <directory-to-ouput-ncore-data> waymo-v4
```

Waymo-open data can be downloaded from https://waymo.com/intl/en_us/open/download/ in form of tfrecords files.
