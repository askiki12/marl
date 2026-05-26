# **MarL：多智能体强化学习（Switch4-v0）**

>单人队伍
>
>231880038 张国良

本仓库包含用于 Switch4-v0 多智能体强化学习实验的训练、评估与渲染工具。下面说明如何配置环境、运行代码，以及仓库主要文件的作用。

**环境配置**
- **Python**: 推荐使用 Python 3.8-3.10
- **创建 conda 环境并安装依赖**:

```bash
conda create -n marl python=3.8 -y
conda activate marl
pip install -r requirements.txt
```

- 注意: `requirements.txt` 中包含带 CUDA 的 `torch` 轮子（`torch==1.13.1+cu117`）。如无 CUDA，请替换为合适的 CPU 版本或本地可用的 CUDA 版本。

**运行说明**
- 在 `marl/marl` 目录下运行训练：

```bash
python train.py --algorithm iql --device cpu
```

- 评估（对已有检查点进行批量评估）：

```bash
python eval.py --algorithm iql --seeds 0 1 2 3 4 --episodes 100 --device cpu
```

- 渲染已训练模型（打开窗口或保存 GIF）：

```bash
python show.py --algorithm iql --checkpoint-name best.pt --save-gif results/gif/iql_demo.gif
```

- 如果要在无显示的服务器上运行渲染并只保存 GIF，请不要设置 `DISPLAY` 环境变量；若要看到窗口渲染，请确保 `DISPLAY` 已设置并可用。

**输出与目录约定**
- 运行后结果保存在 `results/` 下，常见布局：
	- `results/models/<algorithm>/seed_<seed>/`：模型检查点（`best.pt`、`final.pt`）
	- `results/logs/<algorithm>/seed_<seed>/metrics.jsonl`：逐步训练/评估日志
	- `results/figures/<algorithm>/seed_<seed>/`：学习曲线图片
	- `results/gif/<algorithm>/`：由 `show.py` 生成的 GIF

**项目结构与主要文件说明**

- **algorithm**：算法实现目录
  - **iql.py**：iql 算法实现代码
  - **vdn.py**：vdn 算法实现代码

- **result**：运行结果保存目录
  - **eval**：评估结果
  - **figures**：绘制曲线
  - **gif**：gif图像
  - **logs**：运行日志
  - **models**：保存的模型权重

- **utils**：工具目录，包括绘图代码，dqn 基类代码
- **train.py**：训练逻辑的代码
- **eval.py**：评估逻辑的代码
- **show.py**：直观展示智能体的运动

**运行建议与常见问题**
- 若在服务器上无显示但需要渲染 GIF，使用 `--save-gif` 参数并确保 Pillow 可用
- 当出现 Gym 相关导入错误，确认 `ma_gym` 已成功安装（`requirements.txt` 中以 editable 方式从 Git 安装）
- 若遇到 CUDA/torch 版本不匹配，请用与你的 CUDA 驱动相匹配的 PyTorch 版本替换 `requirements.txt` 中的 torch/vision/torchaudio 条目
