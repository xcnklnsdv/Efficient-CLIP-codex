# HMDB51 当前运行诊断（2026-10-03）

## 已确认的结论

2026-10-04 更新：仓库补齐了完整 30 轮日志。最佳结果仍是
epoch 0 的 **70.6536%**；epoch 29 训练为 **100%**，验证为
**70.3922%**。验证分类损失从 0.9704 上升到 2.0613。
这种趋势支持过拟合或训练/验证失配；它不能单独区分所有成因。

原先附件仅包含 epoch 0–15 和部分 epoch 16。本次以仓库中的
`output_dir/emclip/hmdb51_mpeg4_emclip_T16_K8_20261003_213752/train_20261003_213752.log`
完整记录为准；70.39% 是最后结果，最高为 70.65%。

| 日志 epoch（从 0 开始） | Train Top-1 | Val Top-1 | Train L_ME | Val L_ME |
| --- | ---: | ---: | ---: | ---: |
| 0 | 64.6875% | **70.6536%** | 1.2363 | **0.9704** |
| 5 | 96.7330% | 67.7124% | 0.0989 | 1.4470 |
| 10 | 99.0909% | 68.4967% | 0.0301 | 1.5434 |
| 15 | 99.7159% | 68.0392% | 0.0094 | 1.8568 |
| 29 | 100.0000% | 70.3922% | 0.0001 | 2.0613 |

## 实际设置

- 原始 CLIP ViT-B/16 初始化；resume=None、init_checkpoint=None。
- I/R/MV/Text 的适用预训练权重加载覆盖率均为 100%。这验证了张量加载，
  不能替代实际 checkpoint 来源、数据或精度核验。
- paper 完整模型，T=16、K=8、输入 256、tau=0.01。
- 4 个 rank，每卡 batch=16，全局 batch=64；不是启动脚本的默认每卡 4。
  每轮 55 个 optimizer 批次与 train=3570、drop_last=True 相符。
  batch 改变也会改变 L_MG 的全局候选和正负样本组成，不能视为纯吞吐设置。
- full 模式，可训练参数 392,462,336；训练样本数 3570，验证数 1530。
- MGSE train=ground_truth、eval=class_bank、class aggregation=mean。
- AdamW、lr=8e-6、weight_decay=0.2、warmup=0、AMP、梯度裁剪 1。
  optimizer、batch、冻结策略等仍属于论文未完整说明的工程选择。
- 快速验证 1 temporal view × 1 spatial crop。
- 完整 30 轮共 1650 个批次，epoch 5 和 17 各跳过一次 AMP 更新。
  日志没有持续梯度溢出或学习率归零的证据。

## 与论文的比较

本机论文第 7 页 Table 3：EM-CLIP diamond 在 HMDB51 报告 **78.6%**。
第 8 页 Table 5 / Table 6：完整 EM-CLIP K=8 为 **78.9%**，K=16 为
**81.2%**。当前 K=8 最佳快速验证与 78.9% 数值相差约 8.25 个百分点。
这不是受控实验测出的模块贡献，也不能保证多视图会补足该差距。

Table 4 明确列出了 K400 / SSV2 的 4×3 视图。HMDB51 的 Table 3
没有单独列出该行的视图数，论文也未充分交代该数据集的冻结和 split
汇总细节。当前日志是 split 1，不能直接宣称与作者的全部实验协议一致。
补做 4×3 的目的首先是测量快验与多视图差距。

尤其需要区分预训练来源：第 7 页 Table 2 明确将 Breakfast 的 EM-CLIP
L/14 预训练列为 Kinetics-400；Table 3 的 HMDB51 行没有 Pre-training 栏，
正文和附录也没有补充它是否使用 K400 迁移。第 8 页 Table 4 的 CLIP-400M
只对应 K400/SSV2 那张表，不能据此确定 HMDB51 的完整初始化流程。
不能将“原始 CLIP 起点即可达到 78.9%”或“必须先用 K400 才能达到 78.9%”
写成论文已确认的事实。详见 `EMCLIP_FOLLOWUP_AUDIT_20261004.md`。

## 需要隔离的机制

### 1. 类别条件选帧与无标签推理的差异

训练通过真实类别的 label tokens 得到 saliency；验证在全部 51 类各自
执行 temporal softmax 后平均。两者都是当前明确配置的行为。
验证没有读取真实标签选帧。论文未充分定义未知标签推理时的 MGSE 策略，
因此 class_bank 是工程选择，不是已经核验过的作者实现。

调用当前 `_class_bank_saliency` 的合成检查：两个类别分别偏好两个相反的
motion 特征时，ground_truth 类别 0 的 saliency 是 `[1, 0]`，全部类别平均
为 `[0.5, 0.5]`。这证明平均可能抵消显著性，不能证明训练好的 HMDB51
特征也恰好如此；当前验证日志没有 saliency/选帧一致性数据可作量化。

### 2. 短视频重复 GOP 的权重

当前运行每视频平均约 7.76 个有效候选、选中约 6.51 个不同 GOP。
实际选择了 K=8 个张量位置，但并非每视频都是 8 个不同 GOP。
这些是均值，不能据此给出短视频比例或断言所有样本的 GOP 数。

当前代码合成检查：6 个 GOP、T=16、K=8 时，会先保留全部 6 个不同 GOP，
再重复 saliency 最高的 GOP 补齐。若最高是第一个 GOP，结果为
`[0,0,0,1,2,3,4,5]`；若最高是最后一个，结果为
`[0,1,2,3,4,5,5,5]`。所以即使不同 GOP 的集合相同，训练与验证仍可能
通过重复次数改变 MELSC 的有效输入权重。没有真实 checkpoint 消融数据，
不应把此机制直接归因为全部精度差距。

### 3. 小样本全参数训练

3570 个样本训练约 3.92 亿参数，训练 CE 很快接近零，而验证 CE 持续上升。
这是显著的泛化风险。冻结文本、冻结原始 CLIP、增强策略、loss 权重等可以
做独立消融，但不能凭这一份日志替换默认值后称其为论文明确设置。
L_MG 在后期占总损失绝大部分，不代表其梯度一定占相同比例。

## 可运行的下一步

在服务器 GPU 空闲时，用本次最佳权重评估；不要用 latest 替代 best。

```bash
NPROC_PER_NODE=1 BATCH_SIZE=1 MICRO_BATCH_SIZE=1 MASTER_PORT=29511 \
RESUME=output_dir/emclip/hmdb51_mpeg4_emclip_T16_K8_20261003_213752/model_best.pth \
TEMPORAL_VIEWS=4 SPATIAL_CROPS=3 \
bash scripts/eval_emclip_hmdb51.sh
```

下一次优先训练原始 CLIP 起点的 diamond 对照，以隔离 MGSE、L_MG 和选帧
带来的共同影响。保持本次每卡 batch=16、其余学习率等设置不变。
diamond 只使用 I/R，不应通过完整模型 checkpoint 初始化。

```bash
RESUME="" INIT_CHECKPOINT="" K400_CHECKPOINT="" \
BATCH_SIZE=16 MICRO_BATCH_SIZE=16 \
IMPLEMENTATION=paper VARIANT=diamond T=8 K=8 \
bash scripts/train_emclip_hmdb51.sh
```

另一个独立对照是 full 模型训练和验证都用 class_bank；保持 T=16/K=8，
从相同原始 CLIP 重新开始，而不是混用本次 optimizer/scheduler 状态。

```bash
RESUME="" INIT_CHECKPOINT="" K400_CHECKPOINT="" \
BATCH_SIZE=16 MICRO_BATCH_SIZE=16 \
IMPLEMENTATION=paper VARIANT=emclip T=16 K=8 \
MGSE_TRAIN_TEXT_MODE=class_bank MGSE_EVAL_TEXT_MODE=class_bank \
bash scripts/train_emclip_hmdb51.sh
```

新实验须保留单独输出目录、记录完整命令，并使用相同的验证协议比较。
不建议把真实标签用于正式验证选帧来提高成绩。

## 本机验证范围

相关测试：`test_emclip_mgse.py`、`test_emclip_paper_protocol.py`、
`test_emclip_engine.py` 共 **32 passed**；上述合成机制检查实际运行。
这证明对应实现行为和测试约束，不证明复现精度。

本机不能读取服务器的真实 HMDB51 文件、这次 checkpoint 或 native CoViAR
扩展，所以尚未核验原始列表交集、完整 label 映射、真实 GOP 编码间隔、
MV/R 数值范围和 checkpoint 的多视图成绩。未进行完整训练，也未修改训练
默认值或原始数据。
