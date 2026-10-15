第三方组件与来源
================

LinkODT 使用以下组件实现界面、计算和可视化。具体依赖及版本见 :doc:`requirements`。

.. list-table:: 主要组件
   :header-rows: 1

   * - 组件
     - 用途
   * - PySide6 / Qt
     - 图形界面
   * - PyTorch、timm、DINOv2
     - 分割模型与特征编码器
   * - NumPy、SciPy、scikit-image、pandas
     - 图像处理、数值计算与结果表
   * - NetworkX
     - 网络分析
   * - VTK、PyVista、Open3D
     - 三维几何处理与可视化
   * - imageio-ffmpeg、Pillow
     - 图像与视频导出

第三方组件保留各自的许可证和来源说明。Windows 交付包中的相关许可文件位于 ``_internal/THIRD_PARTY_LICENSES/``。使用外部模型或分析方法发表结果时，请引用相应原始论文，见 :doc:`citing`。
