Advanced use and batch analysis
==============================================================

Continue from existing results
------------------------------------------------------------

An existing binary mask can be opened in Instance segmentation. Existing instance point clouds and skeletons can be opened in Network analysis. For three-dimensional images and animations, open Visualization directly.

1. Keep each sample's original image, mask, and ``processed_output`` in neighboring directories.
2. Open the file or folder directly in module 3 or 4.
3. Check the detected sample name and status, then run or inspect the result.
4. After changing an upstream mask, update instances and networks in order so that images and tables describe the same revision.

Batch operations
--------------------------------

All in static modules processes the loaded list. Processing scope in dynamics switches between Current dataset and All loaded datasets. For a new batch, process a representative sample first and inspect dimensions, calibration, output location, and logs before processing all inputs.

Use separate folders for experimental groups and consistent sample IDs to link images, masks, and tables. Include biological replicate, time-point, and channel information in filenames. Automatically generated instance IDs identify objects within the analysis; record biological replicates separately.

Consistent parameters and reproducibility
----------------------------------------------------------------------------------

Record source version, environment, model file, segmentation threshold, spatial calibration, network distances, time interval, tracking radius, and input manifest for each batch. Select Overwrite existing network results after changing network parameters. Rerunning dynamics clears some previous downstream outputs; preserve earlier results before rerunning comparisons.

Model learning and data augmentation
------------------------------------------------------------------------

``semantic_models.py`` manages segmentation architectures. GUI Backbone, architecture, and weights must match. A complete segmentation checkpoint includes the task decoder. DINOv2 encoder weights alone require downstream training.

``ssl/`` provides a separate self-supervised training script. Set the input/output directories and backbone in its USER CONFIG section, check the local paths, and run the script with Python.

``paired_tiff_segmentation_augmentation_full.py``, ``microscopy_segmentation_augmentation_full_v2.py``, and ``evaluate_masks_with_panels_v2.py`` are independent augmentation/evaluation tools. Inspect their settings and input conventions, then verify them on data copies. Keep augmented variants of each original sample in the same training/test group to prevent leakage between groups.

Selecting a model
----------------------------------

Compare architectures under the same split, considering Dice, boundaries, small objects, runtime, and GPU memory. Save failure cases and independent test results alongside training curves. Test previews viewed each epoch can influence manual model selection; reserve additional independent samples for formal performance reporting.

Before processing a batch
------------------------------------------------------------

Check one representative sample at each stage: interpreted dimensions → saved mask → instance labels and morphology → network outputs. Keep model, prediction threshold, calibration, and network distance settings consistent for comparisons. Save the settings with the input sample list.

After editing a mask, rerun the affected instance analysis and then its network analysis; previously saved tables still describe the earlier mask. After changing network distances or scale, recompute network outputs. Display brightness, colour, and camera changes affect presentation without updating quantitative tables.

