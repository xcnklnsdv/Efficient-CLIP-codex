# EM-CLIP 完整日志、代码与预训练核查（2026-10-04）

依据用户提供的原论文 PDF 和仓库最新日志；论文优先于 AGENTS.md。
本次没有执行完整训练，也没有下载模型、修改原始数据或提交 Git。

## 预训练：能确认什么

| 论文位置 | 实验 | 已披露的预训练 |
| --- | --- | --- |
| 第 6 页 §4.2 | 通用实现说明 | 预训练 ViT-B/16 运动编码器；I/R 用 CLIP ViT-B/16 或 L/14；未说明所有数据集的迁移链 |
| 第 7 页 Table 2 | Breakfast，EM-CLIP L/14 | 明确为 **Kinetics-400** |
| 第 7 页 Table 3 | HMDB51/UCF101 等压缩域比较 | **没有 Pre-training 栏**；HMDB51 diamond 为 78.6% |
| 第 8 页 Table 4 | K400/SSV2，B/16 | 明确为 **CLIP-400M** |
| 第 8 页 Table 5/6 | HMDB51 完整模型 | K=8 为 78.9%，K=16 为 81.2%；没有补充预训练链 |

CLIP-400M 指 4 亿图文对的图文预训练，并不是 Kinetics-400 视频预训练。
原始定义见 [CLIP 论文](https://proceedings.mlr.press/v139/radford21a.html)。
当前代码加载原始 CLIP 复用三视觉分支和文本编码器的权重；GSPL、LMPL
和 temporal aggregator 是新增模块。先训练 K400 EM-CLIP 再迁移，会把这些
新增模块也换成已经训练的参数，两条初始化流程不同。

全文相关实现说明和附录没有解决 HMDB51 是否 K400→HMDB51 的问题。
Table 4 的 CLIP-400M 不能替代 Table 3 未披露的 HMDB51 设置；Breakfast
使用 K400 也不能自动推广到 HMDB51。作者是否用了 K400，需要作者配置或
checkpoint 来源进一步确认。此前将原始 CLIP 起点直接对应 HMDB51 的
78.9% 目标，表述过于确定；这里纠正。

建议做 K400 迁移对照，因为当前小样本训练已显著过拟合；这属于实验建议，
不是论文已确认的必要条件，也不能保证达到 78%。两组需要相同模型结构、
split、batch、学习率、选帧模式与验证视图。若 K400 来源是 legacy，而
CLIP 起点是修复后的 paper，两组还同时改变了结构，不能只归因预训练。

## 新上传日志的实际结果

| 数据集 | 完整验证轮数 | 最高 Top-1 | 最后完整 Top-1 | 最后训练 Top-1 | 视图 |
| --- | ---: | ---: | ---: | ---: | --- |
| HMDB51 split 1 | 30 | **70.6536%**，epoch 0 | **70.3922%**，epoch 29 | **100%** | 1×1 |
| SSV2 | 2，后续 epoch 2 部分训练 | **47.5522%**，epoch 1 | **47.5522%** | **62.8564%** | 1×1 |

HMDB51 的最后结果约 70.39%，最高约 70.65%。验证 CE 0.9704→2.0613，
训练 CE 1.2363→0.000147。30 轮共 1650 个批次只记录 2 次 AMP 跳步；
没有持续溢出或训练前学习率被错误归零的证据。日志支持过拟合或训练验证
失配，不能量化哪一个模块导致了多少百分点下降。

SSV2 只有两轮完整验证，40.6264%→47.5522%，不能当作 30 轮最终精度。
其每视频平均 valid_candidates≈4.1358，selected_unique_gops≈4.1358。
K=8 实际大量使用重复 GOP；这两个均值不能推出每条视频的时长或短视频比例。
当全部不同 GOP 都能被保留时，MGSE 仍可能通过“重复哪个最高分 GOP”
改变 MELSC 的输入权重。训练 ground_truth 与验证 class_bank 的不同会放大
这种风险，详见 HMDB51_LOG_AUDIT_20261003.md。尚未得到实际 checkpoint
在服务器上的消融结果，不能把它认定为全部精度差距的唯一原因。

日志、配置和逐轮数据解析保存在
`output_dir/emclip_audit_20261004/training_log_summary.json` 和两个 CSV 中。
HMDB51 最佳对应原日志第 57 行，最后对应第 609 行；SSV2 当前最佳第 1471 行。
原始日志未修改。

## 已复现并修正的问题

上一轮源码核查的三个修复仍在工作区，详细公式依据和回归测试见
EMCLIP_SOURCE_CODE_AUDIT_20261003.md：

1. MELSC 原先在 temporal MHSA 前额外使用视觉 ln_post；新 paper 默认从
   式（23）的最后层原始 CLS 开始，时间聚合后使用 final LN/projection。
   原论文未完整展开聚合头，post-pool LN/projection 及初始化仍明确记录为
   CLIP 公共空间适配选择。历史 checkpoint 保留 pre_and_post。
2. 加载新权重后评估文本缓存未清空，可能沿用旧权重的类别表示；已修复。
3. 可选 predicted_class 曾在与 L_MG 不同的标准化空间预测类别；已修复。
   本次两组日志使用 class_bank，所以该问题不是已确认的本次精度原因。

本次补充发现并修复：

4. HMDB51/SSV2 启动脚本硬编码 GPU ID / 进程数，覆盖用户环境；脚本测试
   复现了 3 个失败。现在统一复用 _gpu_env.sh，默认 4 进程、每卡 batch 4，
   支持外部选择 GPU 和覆盖 batch；保留可选 K400 初始化。
5. 末 P-frame 位置依赖固定 GOP_SIZE，但原先未检查 native GOP 数是否与
   native frame 数相容。错用其他 GOP 大小时，后面的真实 GOP 甚至可能
   被当作没有 P-frame而静默清零。现在在加载图像/MV/R 前明确报错，含
   数据集、原始列表行、路径及两种计数；不改写数据。I-only 最后 GOP
   仍明确返回零运动，已有独立回归测试。

计数一致只能排除部分编码不一致，不能证明所有 I-frame 边界都恰好间隔 12；
真实数据的编码边界、native 解码输出范围仍要在服务器验证。

## 运行边界与对照命令

从原始 CLIP 检查新结构：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,4 NPROC_PER_NODE=4 \
BATCH_SIZE=16 MICRO_BATCH_SIZE=16 \
RESUME="" INIT_CHECKPOINT="" K400_CHECKPOINT="" \
bash scripts/train_emclip_hmdb51.sh --melsc-norm-order post_pool
```

K400 权重迁移（自行替换为真实且结构兼容的 checkpoint）：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,4 NPROC_PER_NODE=4 \
BATCH_SIZE=16 MICRO_BATCH_SIZE=16 \
RESUME="" INIT_CHECKPOINT=/path/to/compatible_k400/model_best.pth \
bash scripts/train_emclip_hmdb51.sh
```

用 INIT_CHECKPOINT 加载训练后模型，目标 optimizer/scheduler/scaler/epoch
重建。RESUME 用于同一 HMDB51 运行继续训练。auto 会恢复源 checkpoint
结构和旧归一化顺序；不要把旧 checkpoint 恢复误写成已使用新结构修复。

旧 HMDB51 最佳权重先补做 4×3 评估：

```bash
CUDA_VISIBLE_DEVICES=0 NPROC_PER_NODE=1 BATCH_SIZE=1 MICRO_BATCH_SIZE=1 \
MASTER_PORT=29511 \
RESUME=output_dir/emclip/hmdb51_mpeg4_emclip_T16_K8_20261003_213752/model_best.pth \
TEMPORAL_VIEWS=4 SPATIAL_CROPS=3 bash scripts/eval_emclip_hmdb51.sh
```

论文明确列出了 K400/SSV2 的 4×3；HMDB51 的具体视图及 split 汇总没有完整
披露。该命令用于测量评估视图的影响，不保证补足 8 个百分点。

本机没有 CUDA、native CoViAR、服务器数据和实际训练 checkpoint，不能
在这里确认原生数据读取、GPU 数值结果或最终精度。

本次检查：`python -B -m pytest -q -p no:cacheprovider tests`：**120 passed**，
包括 full/diamond 两进程 CPU/Gloo 的两次梯度与 AdamW 更新对照；30 个
Python 文件的 AST 检查及主要模块 import 通过，`git diff --check` 通过。
前轮 synthetic full/diamond forward/backward 也已通过。上述结果不替代
CUDA/NCCL、真实 CoViAR 或完整精度复现。
