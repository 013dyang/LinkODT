新用户指南
==========

首次使用时，建议先选择一个代表性样本，通过 GUI 完成分析并检查输出，再处理整批数据。各模块的详细步骤和参数在后续教程中说明。

启动 LinkODT
--------------------

1. 按 :doc:`setup-python` 配置运行环境。
2. 打开包含 ``LinkODT.py`` 的项目文件夹，运行 ``python LinkODT.py``。
3. 根据已有材料，在首页选择下面列出的模块。

选择起点
--------

.. list-table:: 按已有材料选择模块
   :header-rows: 1
   :widths: 34 33 33

   * - 你已有的材料
     - 从哪里开始
     - 接下来得到什么
   * - 原始多通道 TIFF
     - Image preprocessing
     - 图像检查 记录、细胞掩膜、单细胞单通道 TIFF
   * - 单通道图像及分割模型
     - Semantic segmentation
     - 二值掩膜与可编辑实例标签
   * - 三维二值掩膜
     - Instance segmentation
     - 物理/社会实例点云与形态学表
   * - 已有实例输出
     - Network analysis
     - 层级网络、指标表
   * - 点状细胞器时间序列
     - Punctate organelle dynamics
     - 轨迹、时间网络、ROI 表与视频
   * - OBJ / 点云
     - Visualization
     - 展示图、相机路径、动画

无需模型权重的入门示例
------------------------------------------------------------

先用此示例熟悉载入、实例结果检查和三维浏览，再准备分割模型。本节使用标为 DEMO 的合成数据。

1. 将手册目录中的 ``demo_data/`` 复制到可写的工作目录，保持 ``DEMO_static.tif``、``DEMO_static_mask.tif`` 与 ``processed_output/`` 的相对位置。
2. 打开 **Instance segmentation**，载入 ``DEMO_static_mask.tif``。可直接检查已有输出；练习计算时，在工作副本中点击 **Run selected**。
3. 对比 **Physical** 与 **Social** 视图，打开 **Morphology table**。随附示例在两种表示下各有五个实例，对应形态 CSV 各有五行数据、39 列。
4. 打开 **Visualization**，载入 ``processed_output/DEMO_static.obj``，适配视野并旋转样本。随附表面有 6,062 个点和 12,104 个网格单元。
5. 按 :doc:`outputs` 检查保存文件。确认载入与结果查看正常后，再处理自己的代表性样本。

这些数量便于识别随附示例。修改掩膜或算法可能改变结果。

第一次完成静态分析
------------------

1. 启动 ``LinkODT.py``，在首页进入 **Image preprocessing**。载入 TIFF，确认维度，标记 GOOD，圈定细胞区域并保存；执行 Apply cell mask 和 Split channels。
2. 回到 Home，进入 **Semantic segmentation**。载入拆分后的通道图像，加载完整分割模型，设置 Predict threshold，单张预测后检查多个 Z 层。
3. 用右侧工具修订掩膜，点击 **Save masks**。确认同目录下的 ``*_mask.tif`` 已更新。
4. 进入 **Instance segmentation**，载入三维 ``*_mask.tif``，点击 **Run selected**。完成后检查 Physical / Social 点云及 Morphology table。
5. 进入 **Network analysis**，选择实例分析生成的 ``processed_output``。填写 XYZ scale 与距离阈值，运行选定样本，查看 Physical、Whole Physical、Social 三种图视图及分层指标表。
6. 需要展示图时，将 OBJ 或实例 TXT 载入 **Visualization**，调整外观、视角并导出 PNG。

已有掩膜可从第 4 步开始。已有实例或网络结果可直接加载浏览；无需重新计算前面的步骤。

第一次完成时间序列分析
----------------------

1. 进入 **Punctate organelle dynamics**，打开一个时间序列 TIFF。
2. 浏览原始帧，点击 **Generate mask**；软件自动拆分尚未准备的序列并生成掩膜。
3. 在 **Mask review** 中检查掩膜。当前动态页面没有可用的添加/擦除笔刷入口；人工修订说明见 :doc:`dynamics`。
4. 填写真实帧间隔、XY/Z 尺度和匹配半径，点击 **Run dynamic analysis**。
5. 用 **Dynamic overlay** 检查轨迹，画 ROI 后导出 ROI Dynamics CSV；需要视频时点击 Export overlay MP4。

选择参数的顺序
--------------

先检查输入维度与标定，再检查分割，最后调整网络距离。对新实验先运行少量代表样本，查看漏分、误合并、轨迹跳转和邻接变化。确定参数后，保持同一比较批次的设置一致，并保存样本清单、权重文件名和参数记录。

完成的判断
----------

进度条完成后，再检查输出目录是否包含预期文件，并重新加载一例结果。对训练检查训练日志与测试表；对三维实例检查点云与形态学表；对动态分析检查日志末段、轨迹表和至少几个时间点的叠加效果。

主窗口菜单
----------------------------------------------------------------------

**File** 根据当前模块打开数据或保存掩膜；**View** 返回首页、适配查看区并切换界面主题；**Analysis** 提供六个模块入口，并在输入就绪时运行选中样本；**Help → User guide** 或 ``F1`` 打开本手册。当前模块不适用的操作显示为禁用。

外观与首页
----------------------------------------------------------------------

右上角的太阳／月亮图标按钮用于切换深浅主题。悬停可查看将切换的主题与快捷键（``Ctrl+Shift+T``）。深色界面采用中性炭黑，图像与对象颜色保持不变。首页提供六个分析模块入口。开发者与团队信息见 **Help → About LinkODT** 和 :doc:`sources`。

截图中使用的真实数据
----------------------------------------------------------------------

example 包含 HeLa 四通道 Z 图像、独立的线粒体／脂滴／溶酶体通道，以及对应掩膜和三维结果，没有时间轴。线粒体数据用于展示掩膜编辑与细胞器几何；实例和网络演示采用局部裁剪，便于查看结构。

``demo_data/real_examples/`` 包含 ``HeLa_mito_ROI.tif``、已有的二值掩膜，以及当前后端重新计算的结果。先复制到可写的工作目录。在 **Instance segmentation** 载入 ``HeLa_mito_ROI_mask.tif``，查看 **Morphology table**，将其 ``processed_output`` 载入 **Network analysis**，或在 **Visualization** 打开 ``HeLa_mito_ROI_pcd_instance_physical.txt``。局部区域尺寸为 256×256×42，XY 起点为原图中的 (857, 507)。示例使用体素坐标；分析实验数据时需设置对应物理标定。

``HeLa_4channel_ROI.tif`` 保留同一局部视野的四通道原始强度，用于预处理截图。手册没有重复打包完整四通道原图和完整线粒体图像。训练窗口仅展示配置，example 提供的模型文件未用于生成本轮截图的新预测。动态模块继续使用合成 DEMO 时间序列。
