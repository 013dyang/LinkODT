配置 LinkODT 与输入数据
================================

Windows 独立运行包
------------------------------------

完整解压交付 ZIP 后，双击 ``LinkODT.exe``。将 ``_internal/`` 与 ``UserGuide/`` 保持在可执行文件旁，无需安装 Python。通过 **Help → User guide** 或 ``UserGuide/index.html`` 打开手册。本审阅包使用 CPU 处理；模型预测需载入你自己的训练权重。

下面的环境配置与启动步骤适用于源码版。

1. 保持项目文件完整
----------------------

至少保留 ``LinkODT.py``、``LinkNex.py``、``ui_theme.py``、``gui_tasks.py``、``LinkNex.py``、``ui_theme.py``、``gui_tasks.py``、``semantic_models.py``、``punctate_dynamics_module.py``、``linknex_module34_directload_patch.py``、``backends/``、``visualization/``，以及用于 DINOv2 的 ``dinov2-main/``。模块 3/4 与点云编辑器的生产后端嵌在主程序中；``backend_sources/`` 保存可读参考副本。

2. 启动主界面
----------------

.. code-block:: powershell

   python LinkODT.py

随附 ``RUN_LinkODT.bat`` 自动选择可用的项目环境。可通过 ``LINKODT_PYTHON`` 指定 Python 可执行文件，或激活环境后使用上面的命令。

项目根目录的 ``main.py`` 是 IDE 示例脚本。独立可视化的入口则是 ``visualization/main.py``。

3. 准备静态图像
------------------

推荐先准备一个小批次的单通道 TIFF 三维堆栈。原始多通道图像先在 Image preprocessing 中处理。检查软件显示的通道数、Z 层数和宽高，确认与显微镜导出信息一致。

.. code-block:: text

   static_data/
     sample01.tif
     sample01_mask.tif
     sample02.tif
     sample02_mask.tif
     processed_output/       ← 后续实例与网络输出

语义二值掩膜使用 ``<图像名>_mask.tif``，背景为 0，前景为正值；实例标签文件使用 ``<图像名>_mask_instance.tif``。输入图像与掩膜须具有相同的空间尺寸。模块 3 的完整实例分析要求 Z ≥ 2。

4. 准备训练数据
------------------

.. code-block:: text

   training_dataset/
     image/
       sample01.tif
       sample02.tif
     mask/
       sample01_mask.tif
       sample02_mask.tif

选择 ``training_dataset`` 作为 Dataset。掩膜文件也可与图像同名。当前训练配对检查接受二维样本，自动检查尺寸、跳过无效配对并写出报告。三维体数据用于训练前需要准备二维切片与对应掩膜。

5. 准备时间序列
------------------

时间序列使用 TIFF；明确 T、Z、Y、X 和通道轴。先在 Punctate organelle dynamics 中用帧滑条检查时间轴，再执行 Generate mask（自动拆帧）。当前拆帧代码把普通三维数组首轴当作时间；TZYX 数组在每个时间点进行 Z 最大投影。因此不要把静态 ZYX 堆栈当作时间序列直接载入。已拆分的数据使用同名前缀加数字帧号，例如 ``cell01_01.tif``、``cell01_02.tif``。

6. 准备权重
--------------

仅浏览图像、编辑掩膜、加载已有点云不需要分割模型。需要 DINOv2 初始化权重时：

.. code-block:: powershell

   python tools/download_pretrained_models.py --model dinov2_vits14

默认 GUI backbone 为 ViT-S/14。选择 ViT-B/14 时下载对应 ``dinov2_vitb14``，并在 GUI 中同步选择相同 backbone。用于 Predict 的文件应为完整 LinkODT 分割 checkpoint；DINOv2 编码器权重用于初始化训练。

7. 使用随附演示数据
----------------------

手册的 ``demo_data/`` 包含 ``DEMO_static.tif``、``DEMO_static_mask.tif`` 和 ``DEMO_time.tif``。静态掩膜可以直接用于模块 3，时间序列可以用于拆帧和掩膜生成。所有文件由固定随机种子构造，并明确标为合成数据。

载入前检查数据维度
------------------------------------------------------------

.. list-table:: 推荐的输入排列
   :header-rows: 1

   * - 输入
     - 排列
     - 检查项目
   * - 静态单通道体积
     - ZYX TIFF
     - Z 表示深度；模块 3 至少需要两层 Z 切片。
   * - 语义训练配对
     - YX 图像与 YX 灰度掩膜
     - 高度和宽度一致；掩膜背景为 0，前景大于 0。
   * - 时间序列
     - TYX TIFF
     - 第一维表示时间；检查界面显示的帧数。
   * - 三维时间序列
     - TZYX TIFF
     - 当前动态预处理对每个时间点进行 Z 最大投影。

动态拆分按数组形状判断，界面没有轴重指定编辑器。分析前分离显微通道，并对照采集元数据检查实际维度。其三维／四维形状判断会将末维为 3 或 4 的数组视为 RGB/RGBA，应避免含义不明确的排列。五维多通道时间序列需先在此流程外整理。

训练文件直接放入 ``image/`` 与 ``mask/``，配对过程不递归查找子目录。TIFF 或 PNG 掩膜可保留准确标签。语义训练将所有正值视为前景，独立实例编号不会形成不同训练类别。为每个样本使用唯一文件名主干，并选择可写的工作目录。

