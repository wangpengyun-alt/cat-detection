# 小猫奔跑视频目标检测（YOLOv5s）

## 1. 视频来源

| 项 | 值 |
|---|---|
| 来源 | Wikimedia Commons，**Ocicat on Cat Wheel** |
| 原始文件 | `Ocicat_on_Cat_Wheel.webm`（480p 转码版，6.4 MB） |
| 原始规格 | 854 × 480，23.92 fps，63.8 s（1527 帧） |
| 使用片段 | 第 1–16 s，共 358 帧（15.0 s），用于推理演示 |

`frame_01.png` / `frame_02.png` / `frame_03.png` 是**检测结果视频**在 4.5 s、8.7 s、12.3 s 处的截帧，框中数字为该帧置信度。

## 2. 类别与标注数量

单类检测，类别名 `cat`（id = 0）。原始标注来自 COCO 预训练模型的类 15，已重映射为 0。

| 用途 | 图片数 | 标注框数 | 平均框/图 |
|---|---|---|---|
| 总计 | 176 | 209 | 1.19 |
| train | 140 | 168 | 1.20 |
| val | 36 | 41 | 1.14 |

标注格式为 YOLO：每行 `class cx cy w h`，坐标按宽高归一化到 0–1。

## 3. 数据划分

- 抽帧：整段视频每 8 帧取 1 帧，得到 191 张；其中 15 张因猫完全模糊、无有效检测被剔除，保留 176 张
- 划分比例：**train : val = 8 : 2**，随机打乱（seed = 0）后按序切分
- 标注方式：用 COCO 预训练的 YOLOv5m 做模型辅助标注（置信度阈值 0.3，中位置信度 0.76）

## 4. 验证集指标（36 张图 / 41 个实例）

| 指标 | 数值 |
|---|---|
| Precision | 0.971 |
| Recall | 0.902 |
| **mAP@0.5** | **0.959** |
| mAP@0.5:0.95 | 0.740 |

训练配置：yolov5s（COCO 预训练权重），imgsz 320，batch 16，epochs 40，SGD，CPU 训练 14 分钟，`best.pt` 13.6 MB（已剥离优化器状态）。

## 5. 遇到的问题

1. **系统策略拦截 OpenCV**：本机 Smart App Control 为强制模式，`cv2.pyd` 5.1 版被判定无信誉而拒绝加载（`应用程序控制策略已阻止此文件`）。固定使用 `opencv-python==4.10.0.84` 解决。
2. **pip 无法安装**：解包临时目录被拒（`Errno 13`）。改为直接下载 wheel 解包（`tools/fetch_wheel.py`）。
3. **素材下载 429**：Wikimedia 拒绝伪装浏览器的数据中心请求，改用带联系方式的 UA 后正常。
4. **多进程受限**：训练时 `multiprocessing` 建管道被拒，`utils/dataloader` 的 `ThreadPool`/`Pool` 全部改为串行（`tools/train_serial.py`）。
5. **运动模糊导致漏检**：猫在跑轮中高速奔跑，`yolov5s` 在部分帧完全漏检，改用 `yolov5m` 后置信度稳定在 0.8 左右。
6. **指标偏乐观**：验证集标签与训练集同源（均由 YOLOv5m 生成），因此本表的 mAP 不代表对人工标注数据的真实泛化能力；样本仅单场景、单只猫，跨场景使用需重新标注与验证。

## 6. 复现步骤

```powershell
.\setup.cmd                                                     # 环境（全部依赖 + 仓库）
python tools\geturl.py <视频URL> data-in\cat_wheel_480p.webm     # 下载素材
python tools\cut_clip.py data-in\cat_wheel_480p.webm data-in\cat_running_15s.mp4 1 15
# 抽帧 + 模型辅助标注 + 建集见 tools/build_cat_dataset.py 文件头说明
python check_dataset.py --root datasets\cat --data datasets\cat\data.yaml
python tools\train_serial.py --weights yolov5s.pt --data datasets\cat\data.yaml `
    --img 320 --batch 16 --epochs 40 --device cpu --workers 0 --cache ram
```

数据集不随本包提交，图片与标签留在本地 `yolov5-project\datasets\cat\`，按上述步骤可重新生成同分布的训练/验证集。
