# G1 Sim2Real Deployment

G1 机器人 sim2real 部署包，从 [Teleopit](https://github.com/BotRunner64/Teleopit) 提取。

支持两种输入模式：
- **离线 BVH 回放**：从 BVH 动捕文件驱动机器人
- **Pico4 VR 遥操作**：通过 Pico4 头显实时控制

## 安装

```bash
# 基础安装
pip install -e .

# 含 sim2real 依赖（OpenCV + Unitree SDK）
pip install -e ".[sim2real]"

# 含 Pico4 VR 遥操作支持
pip install -e ".[pico4]"
```

Unitree G1 DDS SDK 需额外安装：

```bash
bash scripts/setup/setup_g1_bridge.sh
```

GMR 重定向资产需下载：

```bash
python scripts/setup/download_assets.py --only gmr
```

## 使用

```bash
# 离线 BVH 回放
python scripts/run/run_sim2real.py \
    controller.policy_path=/path/to/policy.onnx \
    input.provider=bvh \
    input.bvh_file=/path/to/motion.bvh

# Pico4 实时遥操作
python scripts/run/run_sim2real.py \
    controller.policy_path=/path/to/policy.onnx \
    input.provider=pico4
```

## 许可

MIT
