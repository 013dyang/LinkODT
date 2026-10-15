Output files and directories
========================================================

``name`` below denotes a sample stem. Optional files depend on the modules run, input content, and successful completion. Check the log first, then the relevant files. One existing output does not establish completion of the full analysis.

Static instance results
----------------------------------------------

A typical layout is shown below. Additional skeleton and mapping intermediates are also generated:

.. code-block:: text

   sample/
     name.tif
     name_mask.tif
     processed_output/
       name.obj
       name_pcd_instance_physical.txt
       name_pcd_instance_social.txt
       name_voxel_instance_phy.txt
       name_instance_morphology_physical.csv
       name_instance_morphology_social.csv

.. list-table:: File purposes
   :header-rows: 1
   :widths: 47 53

   * - File
     - Purpose
   * - ``name.obj``
     - Foreground surface mesh for Visualization
   * - ``*_pcd_instance_physical.txt``
     - Surface points labeled by branch instance
   * - ``*_pcd_instance_social.txt``
     - Surface points labeled by connected object
   * - ``*_voxel_instance_phy.txt``
     - Foreground voxel-to-Physical-label mapping
   * - ``*_instance_morphology_*.csv``
     - Measurements at the corresponding instance level, one row per instance

Network results
------------------------------

Outputs remain in the corresponding ``processed_output`` folder. Main files include ``*_hierarchical_network.pkl``, ``*_hierarchical_network_metrics.csv``, ``*_hierarchical_network_metrics_FULL.csv``, and ``*_instance_properties.csv``, plus Whole Physical node/edge and Social edge tables. Batch summaries use the ``all_images`` prefix.

PKL supports network-object reloading; CSV supports external statistics. Compact and FULL tables contain different metric sets, so use one consistent version for aggregation. Node tables connect instance identity to network properties; edge tables describe node pairs. Retain sample names to distinguish reused instance numbers across images.

Training outputs
--------------------------------

Semantic training generates model files, ``Semantic_split_manifest.csv``, dimension-check/skipped-pair reports, ``Semantic_training_metrics.csv``, loss/Dice curves, and ``Semantic_test_results.csv``. Models use timestamped directories or filenames. Locate outputs using the paths in the training log.

The split manifest records input assignment. Dimension reports explain excluded pairs. Training curves describe optimization; interpret test results in light of sample count, splitting strategy, and manual inspection during model development.

Dynamic outputs
------------------------------

.. code-block:: text

   name.tif
   name/
     name_01.tif
     name_02.tif
     ...
     processed_output/
       name_01_mask.tif
       viewer_original_stack.tif
       viewer_mask_stack.tif
       dynamic_analysis.log
       run_config.json
       tables/
         instances_tracked_all_frames.csv
         tracking_links.csv
         tracking_ambiguous_rejections.csv
         tracks_summary.csv
         frame_manifest.csv
         network_edges_by_frame.csv
         near_pair_distances_by_frame.csv
         network_metrics_by_frame.csv
         node_network_roles_by_frame.csv
         edge_events.csv
         dynamic_network_summary.csv
       tracked_surface_pointclouds/
       roi_definitions.json
       ROI_results/
         roi_dynamic_metrics.csv
         roi_organelle_count.png
         roi_mean_intensity.png

``instances_tracked_all_frames.csv`` records persistent identities by frame/instance. ``tracking_links.csv`` records matches between frames, and ``tracks_summary.csv`` summarizes tracks. ``near_pair_distances_by_frame.csv`` retains proximity distances; ``network_edges_by_frame.csv`` and ``edge_events.csv`` describe edge changes over time. Use the exported CSV headers for exact column names.

The current GUI disables some automatic trajectory/network plots provided by the backend. Their absence alone does not indicate failure. The two ROI PNGs are created by ROI export. ``tif_instance_overlay_tracks.mp4`` is generated through the video-export button.

Display caches and analysis inputs
--------------------------------------------------------------------

``viewer_*_stack.tif`` accelerates display. Dynamic calculations use individual frame ``*_mask.tif`` files. Synchronize display caches after external edits to those masks. Replacing a viewing cache alone leaves the analysis masks unchanged.

Keeping your data
----------------------------------

Retain original images, final masks, models, instance/network tables, settings, and logs. Presentation copies may collect PNG, MP4, and camera-path JSON files separately. Include version and input/output directory information when sharing results.

Reading instance measurements
------------------------------------------------------------

Use ``image_name`` together with ``instance_id`` to identify a row; IDs are local to an image and to its Physical or Social representation. ``voxel_count`` counts foreground voxels, while ``pcd_surface_point_count`` counts sampled surface points. They describe different quantities.

In the current module 3 morphology backend, ``voxel_size_xyz`` is fixed to ``(1, 1, 1)``. Length, surface area, and volume therefore use pixel/voxel coordinate units, their squares, and their cubes. Changing Network analysis XYZ scale does not recalibrate these saved morphology tables. Record the microscope calibration and any external conversion alongside your results.

Instance surface TXT files are comma-separated, without a header, in the order ``x,y,z,r,g,b,nx,ny,nz,instance_id``. The normals ``nx,ny,nz`` are surface directions, and RGB values are display colours. Quantitative analysis should use labels and measurement tables; rendered colours and screenshots are presentation outputs.

