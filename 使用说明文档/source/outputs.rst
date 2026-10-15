输出文件与目录
==============

文件中的 ``name`` 表示样本基名。可选输出是否出现取决于执行的模块、输入内容和运行是否完成。首先检查日志，再检查对应文件；单个文件存在不能证明整个分析已完成。

静态实例结果
------------

典型目录如下，实际运行还会生成骨架和映射的其他中间文件：

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

.. list-table:: 文件用途
   :header-rows: 1
   :widths: 47 53

   * - 文件
     - 用途
   * - ``name.obj``
     - 前景表面网格，可载入 Visualization
   * - ``*_pcd_instance_physical.txt``
     - 带分支实例标签的表面点云
   * - ``*_pcd_instance_social.txt``
     - 带连通对象标签的表面点云
   * - ``*_voxel_instance_phy.txt``
     - 前景体素与 Physical 标签映射
   * - ``*_instance_morphology_*.csv``
     - 对应层级的实例定量表，每行一个实例

网络结果
--------

网络输出仍位于对应 ``processed_output``。主要包括 ``*_hierarchical_network.pkl``、``*_hierarchical_network_metrics.csv``、``*_hierarchical_network_metrics_FULL.csv``、``*_instance_properties.csv``，以及 Whole Physical 节点/边和 Social 边表。批量汇总文件使用 ``all_images`` 前缀。

PKL 用于重新载入网络对象，CSV 用于外部统计。简版与 FULL 表的指标集合不同，汇总时固定使用同一版本。节点表连接实例身份与网络属性，边表描述两节点关系；请保留样本名，防止不同图像的相同编号混淆。

训练输出
--------

语义训练产生模型文件、``Semantic_split_manifest.csv``、维度检查/跳过记录、``Semantic_training_metrics.csv``、损失与 Dice 曲线及 ``Semantic_test_results.csv`` 等。模型使用带时间信息的目录或文件名。查找结果时以训练日志给出的路径为准。

划分清单记录输入归属；维度检查解释哪些配对被排除。训练曲线描述优化过程，测试结果需结合样本数量、划分方式和人工查看情况解释。

动态结果
--------

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

``instances_tracked_all_frames.csv`` 按帧/实例记录持续轨迹身份；``tracking_links.csv`` 记录帧间匹配；``tracks_summary.csv`` 汇总轨迹。``near_pair_distances_by_frame.csv`` 保留邻近距离，``network_edges_by_frame.csv`` 与 ``edge_events.csv`` 用于跟踪网络边随时间的变化。精确列名以导出的 CSV 表头为准。

当前 GUI 调用关闭了部分后端自动轨迹图/网络图输出，因此不要将缺少这些图判定为分析失败。ROI 两张 PNG 由 ROI 导出步骤生成。视频 ``tif_instance_overlay_tracks.mp4`` 需点击导出按钮后生成。

查看缓存与分析输入
------------------

``viewer_*_stack.tif`` 加速显示。动态计算的掩膜输入为逐帧 ``*_mask.tif``。外部编辑掩膜后需同时维护显示缓存；仅替换查看缓存无法更新分析输入。

数据保存建议
------------

保留原图、最终掩膜、模型、实例/网络表、参数和日志。制作展示副本时可以单独收集 PNG、MP4 与相机路径 JSON。移交给其他用户时附上版本及输入输出目录说明。

读取实例测量结果
------------------------------------------------------------

使用 ``image_name`` 与 ``instance_id`` 共同识别一行；编号仅在相应图像及 Physical 或 Social 表示中有效。``voxel_count`` 统计前景体素，``pcd_surface_point_count`` 统计采样表面点，两者表示不同数量。

当前模块 3 形态后端将 ``voxel_size_xyz`` 固定为 ``(1, 1, 1)``，因此长度、表面积和体积采用像素／体素坐标单位及其平方、立方。在 Network analysis 中修改 XYZ scale 不会重新标定已保存的形态表。记录显微采集标定，并随结果说明任何外部单位转换。

实例表面 TXT 使用逗号分隔且无表头，列顺序为 ``x,y,z,r,g,b,nx,ny,nz,instance_id``。``nx,ny,nz`` 是表面法向方向，RGB 是显示颜色。定量分析使用标签与测量表；渲染颜色和截图用于结果呈现。

