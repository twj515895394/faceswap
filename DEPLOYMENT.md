# FaceSwap 项目部署文档

## 📋 目录
- [系统要求](#系统要求)
- [环境准备](#环境准备)
- [安装步骤](#安装步骤)
- [启动脚本](#启动脚本)
- [测试验证](#测试验证)
- [常见问题](#常见问题)

---

## 系统要求

### 硬件要求

#### GPU 模式（推荐）
- **GPU**: Nvidia GPU，支持 CUDA Compute Capability 3.5+
  - RTX 20xx 及以上: 需要 CUDA 13.0
  - GTX 9xx - GTX 10xx: 需要 CUDA 12.6
  - GTX 7xx - GTX 8xx: 需要 CUDA 11.8
- **显存**: 建议 8GB 以上
- **内存**: 建议 16GB 以上
- **磁盘空间**: 至少 10GB 可用空间

#### CPU 模式
- **CPU**: 多核处理器
- **内存**: 建议 32GB 以上
- **磁盘空间**: 至少 10GB 可用空间

### 软件要求

| 软件 | 版本要求 | 说明 |
|------|---------|------|
| 操作系统 | Windows 10/11 (64位) | 必须是64位系统 |
| Python | 3.13 | 必须是3.13版本 |
| Anaconda/Miniconda | 最新版 | 用于创建虚拟环境 |
| Git | 最新版 | 用于克隆代码仓库 |

---

## 环境准备

### 1. 安装必要软件

#### 1.1 安装 Anaconda
```bash
# 下载地址
https://www.anaconda.com/download/

# 安装步骤
1. 下载 Anaconda for Windows
2. 运行安装程序
3. 使用默认选项安装
4. 安装完成后重启电脑
```

#### 1.2 安装 Git
```bash
# 下载地址
https://git-scm.com/download/win

# 安装步骤
1. 下载 Git for Windows
2. 运行安装程序
3. 使用默认选项安装
```

### 2. 检查硬件配置

#### 2.1 检查 GPU 信息
```bash
# 打开命令提示符，输入
nvidia-smi

# 查看 CUDA Version
# 如果没有安装显卡驱动，需要先安装
```

#### 2.2 确定使用的依赖文件

根据你的 GPU 型型选择对应的依赖文件：

| GPU 型号 | 依赖文件 | CUDA 版本 |
|---------|---------|-----------|
| RTX 20xx 及以上 | requirements_nvidia_13.txt | CUDA 13.0 |
| GTX 9xx - GTX 10xx | requirements_nvidia_12.txt | CUDA 12.6 |
| GTX 7xx - GTX 8xx | requirements_nvidia_11.txt | CUDA 11.8 |
| 无 GPU / CPU 模式 | requirements_cpu.txt | 无 |

---

## 安装步骤

### 步骤 1: 创建虚拟环境

#### 方式 A: 使用 Anaconda Navigator（图形界面）
```
1. 打开 Anaconda Navigator
2. 点击左侧 "Environments"
3. 点击底部的 "Create" 按钮
4. 在弹出的对话框中：
   - Name: faceswap
   - Python version: 3.13
   - 点击 "Create"
5. 等待环境创建完成
```

#### 方式 B: 使用命令行
```bash
# 打开 Anaconda Prompt
conda create -n faceswap python=3.13 -y

# 激活虚拟环境
conda activate faceswap
```

### 步骤 2: 克隆项目代码

```bash
# 进入你想要安装的目录
cd h:\dfl

# 克隆项目（如果还没有克隆）
git clone --depth 1 https://github.com/deepfakes/faceswap.git

# 进入项目目录
cd faceswap
```

### 步骤 3: 安装依赖

#### 3.1 激活虚拟环境
```bash
# 如果使用 Anaconda Prompt
conda activate faceswap

# 如果使用 Anaconda Navigator
# 点击 Environments -> faceswap -> 右侧 ">" -> Open Terminal
```

#### 3.2 安装基础依赖
```bash
# 安装 tkinter（GUI 需要）
conda install tk -y

# 升级 pip
python -m pip install --upgrade pip
```

#### 3.3 安装 PyTorch 和其他依赖

**根据你的 GPU 选择对应的命令：**

**Nvidia GPU (RTX 20xx 及以上):**
```bash
pip install -r ./requirements/requirements_nvidia_13.txt
```

**Nvidia GPU (GTX 9xx - GTX 10xx):**
```bash
pip install -r ./requirements/requirements_nvidia_12.txt
```

**Nvidia GPU (GTX 7xx - GTX 8xx):**
```bash
pip install -r ./requirements/requirements_nvidia_11.txt
```

**CPU 模式（无 GPU）:**
```bash
pip install -r ./requirements/requirements_cpu.txt
```

**注意**: 安装过程可能需要 10-30 分钟，取决于网络速度。

### 步骤 4: 配置 Keras 后端

```bash
# 运行配置脚本
python setup.py

# 按照提示选择你的后端：
# - nvidia: 如果使用 Nvidia GPU
# - cpu: 如果使用 CPU
```

或者手动配置：

```bash
# 创建配置文件
python -c "
import json
import os
keras_dir = os.path.expanduser('~/.keras')
os.makedirs(keras_dir, exist_ok=True)
conf_file = os.path.join(keras_dir, 'keras.json')
config = {'backend': 'torch'}
with open(conf_file, 'w') as f:
    json.dump(config, f, indent=4)
print(f'Keras config written to: {conf_file}')
"
```

### 步骤 5: 验证安装

```bash
# 检查 Python 版本
python --version
# 应该输出: Python 3.13.x

# 检查 PyTorch 安装
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
# 应该输出: PyTorch version: 2.9.x 或 2.10.x

# 检查 CUDA 可用性（GPU 模式）
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
# GPU 模式应该输出: CUDA available: True
# CPU 模式应该输出: CUDA available: False

# 检查 Keras 版本
python -c "import keras; print(f'Keras version: {keras.__version__}')"
# 应该输出: Keras version: 3.12.x 或 3.13.x
```

---

## 启动脚本

### 方式 1: 创建批处理脚本（推荐）

在项目根目录 `h:\dfl\faceswap` 下创建以下脚本：

#### 1.1 启动 GUI（图形界面）

创建文件 `start_gui.bat`:
```batch
@echo off
echo ========================================
echo FaceSwap GUI Launcher
echo ========================================
echo.

REM 激活虚拟环境
call conda activate faceswap

REM 检查是否激活成功
if errorlevel 1 (
    echo [ERROR] Failed to activate conda environment 'faceswap'
    echo Please make sure Anaconda is installed and the environment exists.
    pause
    exit /b 1
)

echo [INFO] Activated conda environment: faceswap
echo.

REM 启动 FaceSwap GUI
echo [INFO] Starting FaceSwap GUI...
python faceswap.py gui

REM 如果出错，暂停以便查看错误信息
if errorlevel 1 (
    echo.
    echo [ERROR] FaceSwap GUI exited with an error.
    pause
)
```

#### 1.2 启动 Extract（提取人脸）

创建文件 `start_extract.bat`:
```batch
@echo off
echo ========================================
echo FaceSwap Extract Launcher
echo ========================================
echo.

REM 激活虚拟环境
call conda activate faceswap

REM 检查是否激活成功
if errorlevel 1 (
    echo [ERROR] Failed to activate conda environment 'faceswap'
    echo Please make sure Anaconda is installed and the environment exists.
    pause
    exit /b 1
)

echo [INFO] Activated conda environment: faceswap
echo.

REM 提示用户输入参数
set /p INPUT_DIR="Enter input directory (images or video file): "
set /p OUTPUT_DIR="Enter output directory: "

REM 启动 Extract
echo [INFO] Starting FaceSwap Extract...
echo Input: %INPUT_DIR%
echo Output: %OUTPUT_DIR%
python faceswap.py extract -i "%INPUT_DIR%" -o "%OUTPUT_DIR%"

REM 如果出错，暂停以便查看错误信息
if errorlevel 1 (
    echo.
    echo [ERROR] Extract exited with an error.
    pause
)
```

#### 1.3 启动 Train（训练模型）

创建文件 `start_train.bat`:
```batch
@echo off
echo ========================================
echo FaceSwap Train Launcher
echo ========================================
echo.

REM 激活虚拟环境
call conda activate faceswap

REM 检查是否激活成功
if errorlevel 1 (
    echo [ERROR] Failed to activate conda environment 'faceswap'
    echo Please make sure Anaconda is installed and the environment exists.
    pause
    exit /b 1
)

echo [INFO] Activated conda environment: faceswap
echo.

REM 提示用户输入参数
set /p FACE_A_DIR="Enter face A directory: "
set /p FACE_B_DIR="Enter face B directory: "
set /p MODEL_DIR="Enter model output directory: "

REM 启动 Train
echo [INFO] Starting FaceSwap Train...
echo Face A: %FACE_A_DIR%
echo Face B: %FACE_B_DIR%
echo Model: %MODEL_DIR%
python faceswap.py train -A "%FACE_A_DIR%" -B "%FACE_B_DIR%" -m "%MODEL_DIR%"

REM 如果出错，暂停以便查看错误信息
if errorlevel 1 (
    echo.
    echo [ERROR] Train exited with an error.
    pause
)
```

#### 1.4 启动 Convert（转换视频）

创建文件 `start_convert.bat`:
```batch
@echo off
echo ========================================
echo FaceSwap Convert Launcher
echo ========================================
echo.

REM 激活虚拟环境
call conda activate faceswap

REM 检查是否激活成功
if errorlevel 1 (
    echo [ERROR] Failed to activate conda environment 'faceswap'
    echo Please make sure Anaconda is installed and the environment exists.
    pause
    exit /b 1
)

echo [INFO] Activated conda environment: faceswap
echo.

REM 提示用户输入参数
set /p INPUT_DIR="Enter input directory (images or video file): "
set /p OUTPUT_DIR="Enter output directory: "
set /p MODEL_DIR="Enter model directory: "

REM 启动 Convert
echo [INFO] Starting FaceSwap Convert...
echo Input: %INPUT_DIR%
echo Output: %OUTPUT_DIR%
echo Model: %MODEL_DIR%
python faceswap.py convert -i "%INPUT_DIR%" -o "%OUTPUT_DIR%" -m "%MODEL_DIR%"

REM 如果出错，暂停以便查看错误信息
if errorlevel 1 (
    echo.
    echo [ERROR] Convert exited with an error.
    pause
)
```

### 方式 2: 使用命令行直接启动

```bash
# 激活虚拟环境
conda activate faceswap

# 进入项目目录
cd h:\dfl\faceswap

# 启动 GUI
python faceswap.py gui

# 启动 Extract
python faceswap.py extract -i <输入目录> -o <输出目录>

# 启动 Train
python faceswap.py train -A <人脸A目录> -B <人脸B目录> -m <模型目录>

# 启动 Convert
python faceswap.py convert -i <输入目录> -o <输出目录> -m <模型目录>
```

---

## 测试验证

### 1. 测试 GUI 启动

```bash
# 双击 start_gui.bat 或在命令行运行
conda activate faceswap
cd h:\dfl\faceswap
python faceswap.py gui
```

**预期结果**:
- GUI 窗口成功打开
- 可以看到各个功能选项卡（Extract、Train、Convert）

### 2. 测试 Extract 功能

```bash
# 准备测试数据
# 创建测试目录
mkdir h:\dfl\test_data
mkdir h:\dfl\test_data\input
mkdir h:\dfl\test_data\output

# 放入一些测试图片到 input 目录

# 运行 extract
conda activate faceswap
cd h:\dfl\faceswap
python faceswap.py extract -i h:\dfl\test_data\input -o h:\dfl\test_data\output
```

**预期结果**:
- 开始检测人脸
- 在 output 目录生成提取的人脸图片
- 生成 alignments.json 文件

### 3. 测试 GPU 加速

```bash
# 检查 GPU 使用情况
python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU name: {torch.cuda.get_device_name(0)}')
    print(f'GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB')
"
```

**预期结果（GPU 模式）**:
```
CUDA available: True
CUDA version: 13.0
GPU name: NVIDIA GeForce RTX 3080
GPU memory: 10.00 GB
```

**预期结果（CPU 模式）**:
```
CUDA available: False
```

---

## 常见问题

### 问题 1: conda 命令找不到

**症状**:
```
'conda' is not recognized as an internal or external command
```

**解决方案**:
```bash
# 1. 检查 Anaconda 是否正确安装
# 2. 将 Anaconda 添加到系统 PATH
#    右键 "此电脑" -> 属性 -> 高级系统设置 -> 环境变量
#    在系统变量中找到 Path，添加以下路径：
#    C:\Users\<你的用户名>\anaconda3
#    C:\Users\<你的用户名>\anaconda3\Scripts
#    C:\Users\<你的用户名>\anaconda3\Library\bin
# 3. 重启命令提示符
```

### 问题 2: CUDA 相关错误

**症状**:
```
RuntimeError: CUDA out of memory
或
RuntimeError: CUDA error: no kernel image is available for execution on the device
```

**解决方案**:
```bash
# 1. 检查 CUDA 版本是否匹配
nvidia-smi

# 2. 重新安装正确版本的 PyTorch
# 根据你的 CUDA 版本选择对应的 requirements 文件

# 3. 如果显存不足，可以减小批次大小
# 编辑 config/extract.ini 文件，调整 batch_size 参数
```

### 问题 3: 依赖安装失败

**症状**:
```
ERROR: Could not find a version that satisfies the requirement
```

**解决方案**:
```bash
# 1. 升级 pip
python -m pip install --upgrade pip

# 2. 使用国内镜像源
pip install -r ./requirements/requirements_nvidia_13.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 3. 单独安装失败的包
pip install <包名> -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 问题 4: GUI 无法启动

**症状**:
```
ImportError: No module named 'tkinter'
```

**解决方案**:
```bash
# 安装 tkinter
conda install tk -y

# 或者在虚拟环境中重新安装
conda activate faceswap
conda install tk -y
```

### 问题 5: 提取速度慢

**症状**:
- Extract 过程非常慢

**解决方案**:
```bash
# 1. 检查是否使用了 GPU
python -c "import torch; print(torch.cuda.is_available())"

# 2. 如果使用 CPU，考虑使用 GPU
# 3. 调整配置文件中的批次大小
#    编辑 config/extract.ini
#    增加 batch_size 参数

# 4. 使用更快的检测器
#    在 GUI 中选择 S3FD 检测器（默认）
#    或者在命令行指定: python faceswap.py extract -D s3fd ...
```

---

## 目录结构说明

安装完成后，项目目录结构如下：

```
h:\dfl\faceswap\
├── faceswap.py              # 主程序入口
├── setup.py                 # 安装脚本
├── config/                  # 配置文件目录
│   ├── extract.ini          # Extract 配置
│   ├── train.ini            # Train 配置
│   └── convert.ini          # Convert 配置
├── plugins/                 # 插件目录
│   ├── extract/             # Extract 插件
│   ├── train/               # Train 插件
│   └── convert/            # Convert 插件
├── scripts/                 # 脚本目录
│   ├── extract.py           # Extract 脚本
│   ├── train.py             # Train 脚本
│   └── convert.py          # Convert 脚本
├── lib/                     # 核心库
├── requirements/            # 依赖文件目录
│   ├── requirements_nvidia_13.txt
│   ├── requirements_nvidia_12.txt
│   ├── requirements_nvidia_11.txt
│   └── requirements_cpu.txt
├── start_gui.bat            # GUI 启动脚本
├── start_extract.bat         # Extract 启动脚本
├── start_train.bat           # Train 启动脚本
└── start_convert.bat        # Convert 启动脚本
```

---

## 下一步

安装完成后，你可以：

1. **学习基本用法**: 阅读 [USAGE.md](USAGE.md) 了解基本工作流程
2. **准备训练数据**: 收集两个人脸的图片或视频
3. **开始提取人脸**: 使用 Extract 功能提取人脸
4. **训练模型**: 使用 Train 功能训练模型
5. **转换视频**: 使用 Convert 功能进行人脸替换

---

## 技术支持

如果遇到问题，可以：

1. 查看 [官方文档](https://faceswap.readthedocs.io/)
2. 访问 [FaceSwap 论坛](https://faceswap.dev/forum)
3. 加入 [Discord 服务器](https://discord.gg/FC54sYg)
4. 查看项目 [GitHub Issues](https://github.com/deepfakes/faceswap/issues)

---

**文档版本**: 1.0
**最后更新**: 2026-01-14
**适用项目版本**: FaceSwap (基于 PyTorch + Keras 3)
