Guide for new users
======================================

For your first analysis, use the GUI to process one representative sample and check its outputs before running a full batch. The following module tutorials explain each operation and parameter in more detail.

Starting LinkODT
--------------------------------

1. Configure the environment as described in :doc:`setup-python`.
2. Open the project folder containing ``LinkODT.py`` and run ``python LinkODT.py``.
3. On the home screen, select the module that matches your available data below.

Choose a starting point
----------------------------------------------

.. list-table:: Start from the data you have
   :header-rows: 1
   :widths: 34 33 33

   * - Available input
     - Starting module
     - Next outputs
   * - Raw multichannel TIFF
     - Image preprocessing
     - Image review records, cell masks, single-cell single-channel TIFFs
   * - Single-channel images and a segmentation model
     - Semantic segmentation
     - Binary masks and editable instance labels
   * - Three-dimensional binary masks
     - Instance segmentation
     - Physical/Social point clouds and morphology tables
   * - Existing instance outputs
     - Network analysis
     - Hierarchical networks and metric tables
   * - Punctate organelle time series
     - Punctate organelle dynamics
     - Tracks, temporal networks, ROI tables, and videos
   * - OBJ / point clouds
     - Visualization
     - Presentation images, camera paths, and animations

Worked example without model weights
------------------------------------------------------------------------

Use this example to learn loading, instance-result inspection, and three-dimensional viewing before preparing a segmentation model. This section uses the synthetic files marked DEMO.

1. Copy ``demo_data/`` from this manual's folder to a writable working directory. Keep ``DEMO_static.tif``, ``DEMO_static_mask.tif``, and ``processed_output/`` together.
2. Open **Instance segmentation** and load ``DEMO_static_mask.tif``. Existing outputs can be inspected directly. To practise calculation, use **Run selected** in the working copy.
3. Compare the **Physical** and **Social** views and open **Morphology table**. The supplied example contains five instances in each representation; each corresponding morphology CSV has five data rows and 39 columns.
4. Open **Visualization**, load ``processed_output/DEMO_static.obj``, fit the view, and rotate the sample. The supplied surface has 6,062 points and 12,104 mesh cells.
5. Inspect the saved files described in :doc:`outputs`. Continue with your own representative sample once loading and output inspection work.

These counts identify the supplied example. Changing the masks or algorithms can change the results.

Your first static analysis
----------------------------------------------------

1. Start ``LinkODT.py`` and open **Image preprocessing**. Load a TIFF, confirm its dimensions, mark it GOOD, draw and save the cell regions, then use Apply cell mask and Split channels.
2. Return Home and open **Semantic segmentation**. Load the separated channel images and a complete segmentation model. Set Predict threshold, predict one image, and inspect several Z slices.
3. Revise the mask with the tools on the right. Click **Save masks** and confirm that ``*_mask.tif`` in the image directory has been updated.
4. Open **Instance segmentation**, load the three-dimensional ``*_mask.tif``, and click **Run selected**. Inspect Physical / Social point clouds and Morphology table after processing.
5. Open **Network analysis** and select the instance output's ``processed_output`` folder. Set XYZ scale and distance thresholds, run the selected sample, and inspect the Physical, Whole Physical, and Social graph views and hierarchical metrics.
6. For presentation images, load an OBJ or instance TXT into **Visualization**, adjust appearance and camera settings, and export PNG.

With an existing mask, start at step 4. Existing instance and network results can be loaded directly for inspection.

Your first time-series analysis
--------------------------------------------------------------

1. Open **Punctate organelle dynamics** and load a time-lapse TIFF.
2. Inspect the original frames and click **Generate mask**. The application prepares unprocessed sequences and generates masks automatically.
3. Inspect masks in **Mask review**. The current dynamics page has no available add/erase brush controls; see :doc:`dynamics` for manual revisions.
4. Enter the actual frame interval, XY/Z calibration, and matching radius, then click **Run dynamic analysis**.
5. Inspect tracks in **Dynamic overlay**. Draw ROIs and export ROI Dynamics CSV. Use Export overlay MP4 when a video is required.

Parameter-setting order
----------------------------------------------

Check input dimensions and calibration first, segmentation second, and network distances last. For a new experiment, process a few representative samples and inspect missed objects, merged objects, track jumps, and adjacency changes. Keep settings consistent within a comparison batch and save the sample list, weight filename, and parameter record.

Confirming completion
------------------------------------------

After the progress bar finishes, check the output folder for the expected files and reload one result. For training, inspect the log and test table. For three-dimensional instances, inspect point clouds and morphology tables. For dynamics, check the final log messages, tracking tables, and overlays at several time points.

Main-window menus
----------------------------------------------------------------------

**File** opens data or saves masks for the active module. **View** returns Home, fits the image/scene, and changes the interface theme. **Analysis** opens the six modules and runs the current selection when inputs are ready. **Help → User guide** or ``F1`` opens this guide. Actions that do not apply to the current module are disabled.

Appearance and Home
----------------------------------------------------------------------

The sun/moon button at the upper right switches the interface theme. Hover over it to see the target theme and shortcut (``Ctrl+Shift+T``). Dark mode uses neutral charcoal surfaces; image and object colours are preserved. Home presents the six analysis modules. Developer details are available under **Help → About LinkODT** and in :doc:`sources`.

Real-data examples used in the screenshots
----------------------------------------------------------------------

The supplied example contains a four-channel HeLa Z stack, separate mitochondrial, lipid-droplet and lysosomal channels, and matching masks and three-dimensional results. It contains no time axis. Mitochondrial data was chosen to show mask editing and organelle geometry; a small crop keeps the instance and network demonstrations easy to inspect.

``demo_data/real_examples/`` includes ``HeLa_mito_ROI.tif``, its existing binary mask and results recalculated by the current backend. Copy this folder to a writable work directory. Load ``HeLa_mito_ROI_mask.tif`` in **Instance segmentation**, inspect **Morphology table**, open its ``processed_output`` in **Network analysis**, or load ``HeLa_mito_ROI_pcd_instance_physical.txt`` in **Visualization**. The crop is 256×256×42, with XY origin (857, 507) in the full image. It uses voxel coordinates; apply the correct spatial calibration when analysing your own experiments.

``HeLa_4channel_ROI.tif`` preserves the original intensities of all four channels in the same crop and is used for the preprocessing screenshot. The full four-channel image and mitochondrial stack are not duplicated in the guide. Training windows show configuration only. The supplied model files were not used to produce new predictions for these screenshots. Dynamics continues to use the synthetic DEMO time series.
