Analysis methods and interpretation
======================================================================

This chapter explains the data relationships between modules. See the module tutorials for operations and :doc:`outputs` for file formats.

Image quality and cell regions
------------------------------------------------------------

The quality classifier assists in separating usable and low-quality images. Check GOOD / BAD decisions against experimental criteria. Cell polygons define two-dimensional regions that are applied throughout the Z stack. They select projected cell areas; inspect depth-dependent boundaries separately.

Semantic segmentation
------------------------------------------

Models predict foreground in single-channel, two-dimensional images. Volumes are processed slice by slice. Sliding-window tiles accommodate larger images within GPU memory, with overlapping regions contributing to fusion. Binary masks describe foreground class; downstream instance analysis defines object identity. Slice-view colors alone do not establish three-dimensional identity.

Three-dimensional geometry and instances
--------------------------------------------------------------------------------

Foreground voxels generate voxel sets, surface point clouds, meshes, and skeletons. Skeleton branches, connections, and narrow-bridge processing generate Physical and Social labels, which are mapped back to point clouds/voxels.

Physical describes branches; Social describes connected objects. Comparing masks with skeleton connections helps locate excessive separation and incorrect connections. Spatial adjacency and projected overlap affect segmentation, so inspect final instances against the original slices.

Morphology
--------------------

Outputs include voxel volume, voxel-surface area, shape, and intensity measurements. Current morphology calculations use unit voxel calibration. True areas and lengths in anisotropic images require the appropriate calibrated calculation. When no matching original image is available, the backend constructs an intensity input from the mask; those intensity columns do not represent microscopy fluorescence.

Static networks
------------------------------

Physical networks use branches as nodes and shared junctions as edges. Whole Physical networks extend branch adjacency using minimum skeleton-point distance. Social networks use objects as nodes and threshold minimum surface distance. Cross statistics associate the branch and object layers.

Edge counts, degrees, and connected components characterize the defined graph. Thresholds and calibration directly affect graph structure. Use consistent definitions across experimental groups and inspect threshold sensitivity. Some costly metrics are limited by node count. Preserve NaN outputs and check their calculation conditions.

Tracking and dynamic networks
----------------------------------------------------------

The dynamic workflow establishes instances in each frame, then matches them within spatial search constraints to assign persistent track_id values. Ambiguous matches are recorded in a rejection table. Track appearance/disappearance may reflect movement, field-of-view entry/exit, or segmentation changes; inspect overlays to interpret events.

Networks are built for each frame using spatial distances, then node and edge changes are compared over time. The current GUI makes a Z maximum projection for TZYX input. Its tracks and ROI measurements therefore describe the projected time series. Speed units depend on actual pixel size and Δt.

Visualization and measurement
----------------------------------------------------------

Object colors, opacity, backgrounds, lighting, and cameras control presentation. Changing them does not automatically update segmentation or quantitative inputs. Keep original CSVs, calibration, and model information alongside exported PNG/MP4 files so that displays remain traceable to analysis results.
