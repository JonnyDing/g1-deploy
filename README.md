# G1 Sim2Real Deployment

G1 机器人 sim2real 部署包，从 [Teleopit](https://github.com/BotRunner64/Teleopit) 提取。

支持两种输入模式：
- **离线 BVH 回放**：从 BVH 动捕文件驱动机器人
- **Pico4 VR 遥操作**：通过 Pico4 头显实时控制
- **Noitom / Axis Studio 实时动捕**：通过 BVH UDP 桥接到 G1 控制机

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

### Noitom / Axis Studio 实时动捕


1. 在 Axis Studio 中开启 BVH Broadcasting，协议选择 UDP，并把目标 IP/端口设置为运行 bridge 的电脑。我不确定软件是否可以单独运行，如果可以，就不需要下面的第一个脚本


```bash
python scripts/run/noitom_to_udp_bvh_bridge.py \
    --listen-host 0.0.0.0 \
    --listen-port 7012 \
    --destination-host <G1_CONTROL_PC_IP> \
    --destination-port 1118
```

1. 在 G1 控制电脑上启动 sim2real，并使用 UDP BVH + Noitom 输入：

```bash
python scripts/run/run_sim2real.py \
    controller.policy_path=/path/to/policy.onnx \
    input.provider=udp_bvh \
    input.bvh_format=noitom \
    input.udp_host=0.0.0.0 \
    input.udp_port=1118
```

`noitom_to_udp_bvh_bridge.py` 接收 Axis Studio 的 BVH 文本帧后会输出 `UDPBVHInputProvider` 需要的空格分隔 float 数据。默认输出 180 个 float：`root_xyz + 59 * euler_zyx`，与 `data/sample_bvh/run.bvh` 这类 Noitom BVH 帧一致。如果 Axis Studio 输出 177 个旋转值，bridge 会自动补 `[0, 0, 0]` 根节点平移；如果输出 181 个值，bridge 会把第一个值当作帧号丢弃；如果输出更多值，默认取前 180 个，也可以用 `--trim last` 改为取末尾 180 个。

## 许可

MIT
