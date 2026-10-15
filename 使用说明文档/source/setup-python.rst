配置 Python 环境
========================

启动 LinkODT 前，先配置 Python 运行环境。图形界面使用 PySide6 和 VTK，深度学习训练与预测使用 PyTorch。

1. 确认项目目录
------------------

进入包含 ``LinkODT.py``、``semantic_models.py``、``punctate_dynamics_module.py`` 和 ``visualization/`` 的文件夹。外层同名项目副本文件夹只起收纳作用。后文命令均从真正的项目根目录执行。

2. 创建环境
--------------

有可用 Conda 时，可依据项目随附的环境文件创建独立环境：

.. code-block:: powershell

   conda env create -f environment.yml
   conda activate linkodt

``environment.yml`` 指定 Python 3.11 并列出基础依赖。PyTorch / torchvision 需要根据本机 GPU 与驱动选择匹配的安装方案；使用 `PyTorch 官方安装选择器 <https://pytorch.org/get-started/locally/>`_ 生成命令。

如已使用可工作的 PyTorch 环境，在该环境中安装项目依赖：

.. code-block:: powershell

   python -m pip install -r requirements_linknex.txt

3. 检查环境
--------------

.. code-block:: powershell

   python tools/check_environment.py
   python tools/project_self_check.py

检查 Python 路径是否指向预期环境；检查 numpy、scipy、PySide6、vtk、pyvista、pyvistaqt、torch 等导入结果。需要 GPU 时，确认 ``CUDA available`` 为真。环境检查还会指出缺少的 DINOv2 权重。

4. 安装可选功能
------------------

使用 SegFormer-B2 时：

.. code-block:: powershell

   python -m pip install -r requirements_semantic_optional.txt

Visualization 的基础功能依赖 PySide6、VTK、numpy、imageio 和 imageio-ffmpeg。使用本地 Qwen AI Director 时，另需 ``llama-cpp-python`` 与兼容的 GGUF 模型；详细步骤见项目 ``visualization/INSTALL_LOCAL_QWEN.md``。

5. 本机运行记录
------------------

本手册截图在 Windows 上使用 PySide6 6.8.3 采集。阅读 HTML 手册只需浏览器；运行 LinkODT 时，按前述步骤配置自己的环境，并在分析前检查依赖。
