# Intensity support v0.6

Point-cloud TXT/CSV loading now explicitly supports LinkNex voxel files with four columns:

```text
x y z intensity
```

The fourth column is preserved as VTK point-data array `Intensity` and its original numeric range is retained.

Appearance now includes:

- `Color by intensity`
- intensity availability and min-max range
- mutual exclusion between file RGB and intensity coloring

For four-column voxel data, intensity coloring is enabled by default.
