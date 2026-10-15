Setting up LinkODT and input data
==================================================================

Windows standalone package
--------------------------------------------------------------------

Extract the entire supplied ZIP, then double-click ``LinkODT.exe``. Keep ``_internal/`` and ``UserGuide/`` beside the executable. No Python installation is needed. Open the guide through **Help → User guide** or ``UserGuide/index.html``. This review package uses CPU processing; load your trained checkpoint for model prediction.

The steps below apply to the source-code edition.

1. Keep the project files together
--------------------------------------------------------------------

Retain at least ``LinkODT.py``, ``LinkNex.py``, ``ui_theme.py``, ``gui_tasks.py``, ``semantic_models.py``, ``punctate_dynamics_module.py``, ``linknex_module34_directload_patch.py``, ``backends/``, ``visualization/``, and ``dinov2-main/`` when using DINOv2. The production backends for modules 3/4 and the point-cloud editor are embedded in the main program; ``backend_sources/`` contains readable reference copies.

2. Start the main interface
------------------------------------------------------

.. code-block:: powershell

   python LinkODT.py

The supplied ``RUN_LinkODT.bat`` selects an available project environment. Set ``LINKODT_PYTHON`` to choose a specific Python executable, or activate your environment and use the command above.

The root-level ``main.py`` is an IDE example script. The independent visualization entry point is ``visualization/main.py``.

3. Prepare static images
------------------------------------------------

Start with a small set of single-channel, three-dimensional TIFF stacks. Process raw multichannel images in Image preprocessing first. Check the displayed channel count, Z depth, width, and height against the microscope export information.

.. code-block:: text

   static_data/
     sample01.tif
     sample01_mask.tif
     sample02.tif
     sample02_mask.tif
     processed_output/       ← subsequent instance and network outputs

Semantic binary masks use ``<image_name>_mask.tif``, with background 0 and positive foreground values. Instance-label files use ``<image_name>_mask_instance.tif``. Images and masks must have matching spatial dimensions. The full module 3 pipeline requires Z ≥ 2.

4. Prepare training data
------------------------------------------------

.. code-block:: text

   training_dataset/
     image/
       sample01.tif
       sample02.tif
     mask/
       sample01_mask.tif
       sample02_mask.tif

Select ``training_dataset`` as Dataset. Masks may also use the same filename as their images. The current pairing checks accept two-dimensional samples, validate dimensions, skip invalid pairs, and write reports. Prepare two-dimensional image slices and matching masks before training from volumetric data.

5. Prepare time series
--------------------------------------------

Use TIFF and establish the T, Z, Y, X, and channel axes. Inspect the time axis with the frame slider in Punctate organelle dynamics before clicking Generate mask, which automatically prepares frames. The current splitting code treats the first axis of a conventional three-dimensional array as time. For TZYX arrays, it makes a Z maximum projection at each time point. Keep static ZYX stacks out of the time-series route. Prepared frames use a shared prefix and numeric frame index, such as ``cell01_01.tif`` and ``cell01_02.tif``.

6. Prepare model weights
------------------------------------------------

Image browsing, mask editing, and existing point-cloud inspection can be used without a segmentation model. To download DINOv2 initialization weights:

.. code-block:: powershell

   python tools/download_pretrained_models.py --model dinov2_vits14

The default GUI backbone is ViT-S/14. For ViT-B/14, download ``dinov2_vitb14`` and select the same backbone in the GUI. Predict requires a complete LinkODT segmentation checkpoint. DINOv2 encoder weights initialize training.

7. Use the supplied examples
--------------------------------------------------------

``demo_data/`` contains ``DEMO_static.tif``, ``DEMO_static_mask.tif``, and ``DEMO_time.tif``. The static mask can be used directly in module 3. The time series demonstrates frame preparation and mask generation. All examples were generated with a fixed random seed and are identified as synthetic data.

Check dimensions before loading
--------------------------------------------------------------

.. list-table:: Recommended input arrangements
   :header-rows: 1

   * - Input
     - Arrangement
     - Check
   * - Static single-channel volume
     - ZYX TIFF
     - Z is depth. Module 3 requires at least two Z slices.
   * - Semantic training pair
     - YX image and YX grayscale mask
     - Matching height/width; mask background 0 and foreground > 0.
   * - Time series
     - TYX TIFF
     - The first dimension is time; confirm the displayed frame count.
   * - Volumetric time series
     - TZYX TIFF
     - Current dynamics preparation produces a Z maximum projection per time point.

The dynamics splitter uses array shape and does not provide an axis-reassignment editor. Separate microscopy channels before analysis and verify the interpreted dimensions against acquisition metadata. Its 3D/4D shape heuristics treat a final dimension of 3 or 4 as RGB/RGBA; avoid ambiguous arrangements. Five-dimensional multichannel time series require preparation outside this route.

Keep training files directly inside ``image/`` and ``mask/``; pairing does not recursively search subfolders. TIFF or PNG masks preserve exact labels. Positive mask values are treated as foreground for semantic training, so individual instance IDs do not define separate training classes. Give each sample a unique filename stem and use a writable working directory.

