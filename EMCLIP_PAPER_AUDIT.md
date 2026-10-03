# EM-CLIP 论文与 SSV2 训练审计

审计日期：2026-10-03。代码基线：`5c90b79`。

**本报告记录修复前的基线及其训练日志。后续代码已修复共享 Residual 特征、固定分类温度、ACG 标准化、SSV2 翻转和短 GOP 重复选择等问题；当前默认设置及验证结果见 `IMPLEMENTATION_NOTES_EMCLIP.md` 的“2026-10-03 修复”章节。历史日志和结论保留，不能将其误读为修复后已重新训练的结果。**

论文：用户提供的《Efficient motion-centric CLIP for compressed video action recognition》，Pattern Recognition 172 (2026), 112696，DOI `10.1016/j.patcog.2025.112696`。本次直接读取该 PDF，核对了方法公式、图 1–3、Table 4 和附录。

## 结论

当前代码确实实现了 MGSE → 选取 I/R → MELSC → 分类的主要流程，不是空壳；但**不能认定为严格按论文实现，也不能把测试通过等同于复现精度成立**。

存在一个确定的 SSV2 标签增强错误，以及数处明确的公式/实验协议差异。另外，当前跨 GOP 模块具有顺序不敏感性，短视频补采样会重复选择 GOP。这些问题需要处理或做消融。

新拉取的 SSV2 日志显示的是明显的训练/验证分离：训练 Top-1 接近 96%，验证约 41%。因此不宜首先归因于模型没有训练、预训练权重没有加载、学习率归零或训练轮数单纯不足。**日志证明了泛化问题，但没有提供足够数据来量化每个代码问题贡献了多少精度差距。**

本次没有修改模型、训练器、数据读取器或现有启动脚本。新增审计探针、日志汇总、曲线和本报告，保留训练基线供后续对照。

## 1. 那次 SSV2 训练实际发生了什么

主日志：`output_dir/emclip/ssv2_mpeg4_emclip_T16_K8_20260823_230914/train_20260823_230914.log`。

| 项目 | 日志中的实际值 |
| --- | --- |
| 模型 | EM-CLIP B/16，T=16，K=8，输入 256 |
| 训练策略 | `full`，399,568,897 个可训练参数 |
| 并行与 batch | 3 GPU，每 GPU 8，global batch=24；没有拆分对比 batch |
| 类别文本 | 仓库 `something_v2_labels.csv`，174 类 |
| 初始化 | 原始 CLIP 加载后，再加载 K400 EM-CLIP `model_best.pth` |
| K400 来源 | source epoch=13，source best Top-1=77.7834% |
| 续训状态 | `resume=None`；目标 optimizer/scheduler/scaler 从 epoch 0 开始 |
| LR / 优化器 | 初始 8e-6；AdamW，weight decay=0.2，warmup=0 |
| MGSE | `class_bank` + `mean`，motion pooling=`saliency`，tau=0.01 |
| 损失 | lambda_mg=lambda_me=1；L_MG + L_ME |
| 压缩域 | MV accumulate=False；Residual accumulate=True |
| 验证 | **1 temporal view × 1 spatial crop** |
| 数据数量 | train=168913，val=24777 |
| 可见日志范围 | 完成 epoch 0–13；末尾停在 epoch 14 的 step 5630/7038 |

日志行 10–20：I/R/MV/Text 预训练加载覆盖率都是 100%，无缺失键或 shape mismatch。行 26：K400 只加载模型权重，目标训练状态重建。因此这次训练没有证据支持“CLIP 整体随机初始化”或“K400 resume 导致 LR 归零”。注意覆盖率验证的是张量加载完整性，不能证明 CLIP 文件的训练来源或预训练质量完全正确。

| 完成轮次（日志 epoch） | Train Top-1 | Val Top-1 | Train L_ME | Val L_ME |
| --- | ---: | ---: | ---: | ---: |
| 1（0） | 34.97 | 34.65 | 2.566 | 2.595 |
| 3（2） | 60.62 | 41.12 | 1.361 | 2.328 |
| 7（6） | 81.85 | 41.53 | 0.549 | 2.907 |
| 11（10） | 91.82 | 39.83 | 0.242 | 3.564 |
| 14（13） | 95.60 | **41.64** | **0.136** | **3.702** |

最佳可见结果是日志行 **10064** 的 41.6354%。从 epoch 2 起验证分类损失整体上升，训练分类损失持续下降，是明显过拟合/泛化失配的表现。标签或训练/验证数据预处理差异也能造成类似曲线，因此真实数据核验仍然必要。

完整 14 轮共记录 35 次 rank 0 AMP step skip，约占 98,532 个 batch 槽位的 **0.0355%**。这不能支持“持续 AMP 溢出使大部分训练失效”。末尾 LR 仍约 4.09e-6，未归零。

![SSV2 训练与验证曲线](<C:/Users/Frank/Downloads/Efficient CLIP/output_dir/emclip_audit_20261003/ssv2_training_curve.png>)

完整逐轮数据：`output_dir/emclip_audit_20261003/ssv2_epoch_metrics.csv`；结构化摘要：同目录 `ssv2_log_summary.json`。

## 2. 确定的错误：SSV2 翻转不处理方向标签

位置：`dataset_coviar.py:651`、`dataset_coviar.py:778`。

所有训练数据以 50% 概率水平翻转，I/MV/R 同步翻转且 MV x 分量取负，这部分几何处理是正确的。但 `__getitem__` 最后仍返回原始 `item.label`。

SSV2 至少有如下明确的方向类：

| 原类别 ID | 水平翻转后的对应 ID |
| --- | --- |
| 86：Pulling something from left to right | 87：Pulling something from right to left |
| 93：Pushing something from left to right | 94：Pushing something from right to left |
| 166：Turning the camera left while filming something | 167：Turning the camera right while filming something |

审计探针强制调用现有增强和标签返回逻辑：图像确实翻转，MV x 从 +1 变成 -1，但类别 86 仍返回 86。于是模型会收到“右到左”的视觉运动和“左到右”的监督。

建议首先对 SSV2 禁用水平翻转，或建立经过类别核实的标签置换；不要只修 MV 符号。作为独立参考，TSM 作者代码对 Something/Jester 直接关闭水平翻转：[TSM main.py](https://github.com/mit-han-lab/temporal-shift-module/blob/master/main.py#L67)。这不是 EM-CLIP 论文给出的增强细节，但支持该任务对翻转标签语义的要求。

影响判断：这是确定应修的问题；只涉及少数明确方向类时，**不能单独解释全部约 25 个百分点的差距**。

## 3. 确定的论文结构差异：GSPL 与 LMPL 没有使用同一 Residual 属性特征

位置：`models/emclip_melsc.py:49`、`:102`、`:108`。

论文第 5 页 Eq. (15)–(16) 生成同一个 `a_R` 和 `LN(a_R)`；Eq. (13) 明确说 GSPL 的 `K_R/V_R` 来自 Eq. (16)。图 3 中也只有一条 Residual FC → Norm 路径，分给 GSPL 的 K/V 和 LMPL 的 Q/K/V。

当前代码分别创建：

```python
a_r = gs_r_proj[l](cls_r)
kv_r = gs_r_ln[l](a_r)       # GSPL

a_lm = lm_r_proj[l](cls_r)
q_lm = lm_ln[l](a_lm)        # LMPL
```

两套 FC/LN 参数独立。探针捕获同一层进入两个 attention 的 Residual K/V，最大差异为 **3.26855**，不是同一组特征。

应当区分“共用 Eq. (16) 的输入特征”和“共用两种 attention 的内部投影”：论文允许两个 attention 各自有 W_Q/W_K/W_V，但它们的公共 Residual 输入应相同。

建议逐层只生成一次 Residual 属性特征，让 GSPL 和 LMPL 共同使用。该修改涉及 checkpoint 参数键和训练行为，旧 checkpoint 不能无说明地当作新结构训练的权重；需要保留旧结构兼容路径或明确迁移规则。

## 4. 确认的表示局限：跨 GOP 顺序没有进入模型

位置：`models/emclip_melsc.py:96`、`:158`；空间位置编码在 `models/emclip_layers.py:144`。

三个视觉编码器使用每帧相同的二维 patch 位置编码。GSPL、LMPL、temporal aggregator 都是无时间位置编码、无时间方向 mask 的全注意力，最后做时间平均。`selected_indices` 及 GOP 时间位置没有作为数值特征传入 MELSC。

因此，对完整 I/R GOP 对做同一个任意排列，跨帧 attention 是排列等变的，AvgPool 后视频特征是排列不变的。仅把 top-k 索引按时间排序，并不能让这种架构感知先后顺序。

实际小模型探针：

- 任意重排 GOP 对，video feature 最大差异：**5.96e-8**。
- 倒序排列 GOP 对，最大差异：**6.71e-8**。
- 把 Residual 改成全零，最大差异：0.00365；说明 Residual 分支确实生效。

限定：这里重排的是已经解码的 GOP 元组，不是将原视频物理倒放并重新编码。MV/R 仍保留 GOP 内的有符号运动，所以不能说模型完全没有运动方向信息；可以确定的是，它不能区分这些 GOP 元组的跨 GOP 先后顺序。这对 SSV2 的动作过程、状态转换和细粒度区分是重要风险。

**论文 Eq. (11)–(23) 同样没有明确给出 temporal positional embedding；因此这是已确认的当前架构局限和论文实现细节缺口，不能直接宣称作者一定用了某种时间编码。** 后续可以增加时间编码作明确标记的消融，不能把该扩展未经验证称为原论文实现。

## 5. 确认的短视频采样风险：K 个槽位不一定是 K 个不同 GOP

位置：`dataset_coviar.py:424`；`models/emclip_mgse.py:34`。

GOP 数不足 T 时，均匀重复 GOP，valid_mask 全为 True。重复项都对应真实 GOP，不能简单一律认定为坏数据；但后续 top-k 只知道候选槽位，不知道哪些槽位是同一个 GOP。

以 8 个 GOP、T=16、K=8 为例：

```text
candidate GOPs: [0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7]
selected GOPs:  [4,4,5,5,6,6,7,7]
unique GOPs:   4
```

该例的 saliency 为每个 GOP 设置不同分数，同一 GOP 的重复槽位共享相同分数，说明重复 top-k 可以确定损失实际时间覆盖。在无随机 dropout 的当前编码器中，重复解码、同一空间增强后的 GOP 通常也得到相同特征。

此外，GOP<T 的采样分支没有使用 temporal_view；同例设置 4 temporal views，得到的不同时间采样只有 **1** 种。平方画面做 3 个空间 crop 时也可能完全相同。因此设置 4×3 参数不保证短视频具有 12 个不同观察。

建议对实际 SSV2 数据统计 GOP 数分布、候选重复率、selected unique GOP 数、各 view 相同率；优先选不同 GOP，数量确实不足 K 时再明确重复。这里没有真实视频，不能声称上述例子发生于多少比例的 SSV2 样本。

## 6. MGSE 默认协议是工程扩展，不是论文 Eq. (7) 的直接实现

位置：`models/emclip_mgse.py:151`；本次训练日志行 22。

论文用某个类别描述的有效词做时间 softmax，再在词维平均。代码的 `ground_truth` 路径基本对应这种算法，但默认 `class_bank` 对 174 个类别分别做 softmax 后又平均一次。训练阶段也用该 class_bank 平均，与类别条件选择不同。

两类相反文本分别偏向两个不同帧时，探针得到：

```text
单类别 saliency: [1.0, 0.0]
全部类别 mean:  [0.5, 0.5]
```

这是平均可能抵消类别选择信号的反例，不是对真实 trained SSV2 saliency 的定量证明。日志中的平均 saliency entropy 约 2.61–2.66，接近 log(16)=2.773，说明候选权重整体较分散；重复候选同样会影响这个统计，不能据此直接认定 MGSE 完全失效。

还存在较小的公式差异：Eq. (5) 是无仿射参数的逐特征标准化；代码对 motion 使用可训练 `feature_ln`，对 text 使用无仿射 `F.layer_norm`。初始状态近似一致，学习后不再严格等同 Eq. (5)。

论文没有解释未知真实标签时如何获得该类别条件文本。现有 class_bank 是合理的防泄漏工程选择，但必须标记为额外假设。**不能通过测试时喂真实标签来换取更高数字，并将其作为正式精度。**

建议先以 diamond（无 MGSE，TSN 采样）隔离 MELSC 能力；再比较 class_bank、predicted_class，以及训练/评估分别选择文本策略的明确协议。当前单一 text_mode 让 ground_truth 训练的首次普通 validation 直接抛出防泄漏异常，不能默认完成这种混合协议。

## 7. 分类温度不是强制固定为论文 0.01

位置：`models/emclip.py:282`、`:538`。

论文 Eq. (25) 和 Section 4.2 指定 tau=0.01。当前 L_MG/ACG 使用配置 0.01，而 L_ME 使用可训练的 `exp(logit_scale).clamp(max=100)`。不加载 checkpoint 的默认有效分类温度为 0.07；加载后来自 checkpoint，可以继续训练，并非恒等于 0.01。

**这次真实日志中的原始 CLIP logit_scale=4.60517025，即约 100 倍相似度，对应 0.01；因此不能把“本次从 0.07 开始”当作低精度原因。** 随后的 K400 初始化会覆盖参数，其真实值和 SSV2 训练过程数值没有记录。应检查 checkpoint 并补记 effective temperature，而不是只看 `--mgse-temperature`。

正的全局 logit scale 本身不改变同一 forward 的 Top-1 argmax；差异主要会改变训练损失、梯度和概率校准，不能仅改评估温度就指望修复 25 个点的精度。

该策略符合仓库 AGENTS 要求沿用 CLIP logit_scale，但与论文“固定 tau”的严格字面协议不同；这是需求与论文之间的差别。

## 8. 输入预处理需要核验的差异

位置：`dataset_coviar.py:560`、`:585`、`:657`。

- I 从 CoViAR BGR 转 RGB、除 255、CLIP mean/std：正确。
- MV 是真正 CoViAR 2 通道输出，resize 校正 x/y 量级，flip 修正 x 符号，clamp/20：几何处理正确；具体数值尺度是工程选择。
- R 除 255，clamp 到 [-1,1]，没有 CLIP std 归一化，也没有 BGR→RGB。

仓库原生 C 解码器 `pytorch-coviar/data_loader/coviar_data_loader.c:143`–`:170` 用 BGR 数据产生有符号残差；当前 R 分支却复制 RGB CLIP conv1 初始化。探针输入通道 [10,20,30]：I 返回 [30,20,10]/255，R 返回 [10,20,30]/255。两者通道语义不同，对使用 RGB 预训练权重的 R 分支初始化造成失配。

这不表示残差必须当普通 RGB 归一化，也不证明论文采用了某个固定尺度。原 CoViAR 数据集用 residual 加 128/clamp 后中心化并除 ImageNet std，和这里的 signed/255 不同；所以原文档“沿用已验证归一化”的表述不足以证明复现一致性。需要核验服务器实际加载的 `/home/fuh/m2clip/Coviar/data_loader` 原生实现，以及真实 R 的值域、通道和 patch embedding 激活量级，再做受控归一化消融。

CoViAR accumulate 语义按仓库 C 源码核实：

```text
I:  load(path, gop_idx, 0,      0, False)
MV: load(path, gop_idx, last_p, 1, False)
R:  load(path, gop_idx, last_p, 2, True)
```

R=True 使用累计运动对应关系，将目标帧减去经累计运动映射的 GOP I-frame，体现 GOP 内积累变化；不是把 R1…R11 在同一像素直接相加，也不是未经运动补偿的 P_last-I。MV=False 表示最后 P-frame 相对直接参考帧的位移；MV=True 则追踪到 I-frame 的累计对应关系。论文未明示 MV accumulate 的布尔值，当前 False 应标为实现假设。最后不完整 GOP 的 last_p 使用实际帧数限制，在固定 GOP=12、无 B-frame 前提下逻辑正确。

## 9. 论文实验协议与那次运行不同

第 8 页 Table 4：

| 模型 | T / K | SSV2 Top-1 | 测试视图 | 表中预训练来源 |
| --- | --- | ---: | --- | --- |
| EM-CLIP diamond B/16 | K=8 | 67.1% | 4×3 | CLIP-400M |
| EM-CLIP B/16 | T=16 / K=8 | **67.3%** | **4×3** | CLIP-400M |
| EM-CLIP diamond B/16 | K=16 | 70.4% | 4×3 | CLIP-400M |
| EM-CLIP B/16 | T=32 / K=16 | **70.5%** | **4×3** | CLIP-400M |

当前应比较 K=8 的 67.3%，不能把 K=16 的 70.5% 当作相同帧数目标。那次日志是 1×1，并额外使用 K400 EM-CLIP 权重作初始化；Table 4 列出的是 CLIP-400M，未交代 SSV2 使用这种 K400 迁移。

可以对现有 SSV2 最佳 checkpoint 先进行 4×3 评估，量化视图差异，且必须指定 **SSV2 checkpoint**：

```bash
RESUME=/path/to/ssv2_mpeg4_emclip_T16_K8_20260823_230914/model_best.pth \
TEMPORAL_VIEWS=4 SPATIAL_CROPS=3 \
bash scripts/eval_emclip_ssv2.sh
```

不设 RESUME 时，现有 SSV2 评估脚本会默认加载 K400 checkpoint 并直接评估 SSV2；那是跨数据集迁移评估，不能当作训练好的 SSV2 模型成绩。

原始 CLIP 起点的对照训练应显式 `INIT_CHECKPOINT=""`。现有 diamond 示例也应清空 full EM-CLIP 的 K400 初始化：diamond 根本不创建 MGSE，严格 state_dict 加载 full checkpoint 会出现 MGSE unexpected keys。该问题不影响主日志中的 full→full 初始化。

图 1 在 Motion Encoder 和 Text Encoder 旁画了雪花，在 MELSC 旁画火焰；这通常暗示冻结/训练范围不同。Section 4.3.3 又写端到端微调 CLIP，而没有详细分支冻结表。**这是图文信息待澄清项，不足以断言作者全程冻结 text/MV，也不足以宣称当前 full 策略就是唯一正确协议。** 可以将 freeze_text/冻结 MV backbone 作为有记录的消融。

论文没有完整交代 optimizer、batch size、weight decay、warmup、数据增强和 MGSE 正式推理实现。本仓库工程默认值可运行，但不等于这些值经作者确认。

## 10. 已核对基本正确的部分

- 独立 I、R、MV module；预训练 loader 复制权重且检查参数 storage 不共享。
- MV patch 是两通道，RGB channel mean/repeat×1.5 初始化；位置编码 bicubic 插值，测试支持动态分辨率。
- 官方 CLIP BPE 词表、SOT/EOT、文本 causal mask；类别词 mask 排除前缀/padding/SOT/EOT。论文“词”与 BPE 子词的对应仍属工程解释。
- 训练文本不会长期复用旧特征；train() 清除 eval cache。
- ACG 的 softmax 沿 T；top-k 时间排序；按 batch gather 同步决定 MELSC I/R 输入。
- GSPL 的 query 来自 I，key/value 来自 R；LMPL 的 attention 沿 K；SAG 每层拼入两个 prompt token，层后丢弃 prompt 输出。
- Residual 分支逐层执行；最后 R block 的输出没有分类路径，冻结这段尾部避免 DDP unused 参数，符合当前仅 I-CLS 聚合的实现。
- L_MG 是 mask-aware motion pooling + batch-wise 双向 KL，同类别 multi-positive；使用 batchmean；有梯度 all-gather。
- L_ME 是视频特征对 class EOT 特征的 CE；total 只包含 lambda_mg*L_MG + lambda_me*L_ME，无重复旧 CLIP CE。
- 正式 evaluation 不向 MGSE 传真实标签；有显式 ground_truth 防泄漏错误。
- 训练 DistributedSampler.set_epoch、AMP/scaler、grad clipping、latest/best、恢复 optimizer/scheduler/scaler 路径存在。
- validation 先按视频平均 view logits，再累计正确样本数和总数，all_reduce 后求准确率；非 padding 分布式 eval sampler 不重复计数。

硬 top-k 不可微，因此 L_ME 对 MGSE 参数没有梯度；探针实际确认是 0 个参数。这与论文 argmax_K 的离散选择一致，不应作为额外“实现 bug”。MGSE 实际由 L_MG 训练，MELSC 的 CE 不会直接教它选哪些帧。

## 11. 正确的公式对应表

现有 `IMPLEMENTATION_NOTES_EMCLIP.md:21`–`:27` 的公式分组编号错误，例如它把 Eq. (15)–(18) 写成 KL，实际上那些式子属于 LMPL。

| 论文公式 | 内容 | 当前实现位置 / 差异 |
| --- | --- | --- |
| 1 | MV patch、CLS、位置编码 | PatchTokenEncoder.embed_patches |
| 2–4 | MV Transformer、最后 CLS、LN/projection | MGSE._encode_motion；PatchTokenEncoder.forward |
| 5–6 | 特征标准化、cosine | _prepare_features / _normalize_token_features；motion affine LN 是差异 |
| 7 | 时间 softmax、类别词平均 | _ground_truth_saliency；class_bank 是扩展 |
| 8 | top-K | select_topk_indices；gather_temporal |
| 9–10 | I/R embedding | MELSC._initial_tokens |
| 11–14 | I 属性特征、GSPL、ISIA | MELSC._layer_prompts；Residual 输入没有按 Eq.16 共享 |
| 15–18 | R 属性特征、LMPL、IMIA | MELSC._layer_prompts |
| 19–22 | I prompt SAG、I/R MHSA+FFN | MELSC.forward；视觉 TransformerBlock |
| 23 | frame CLS temporal MHSA/FFN/AvgPool | MELSC.temporal_blocks 和最后 pooling；norm/proj 属于工程补充 |
| 24 | 双向 motion/text KL | motion_text_kl_loss |
| 25 | 分类 CE / tau | EMCLIP.forward；learned logit_scale 不是强制固定 tau |
| 26 | 两项总损失 | EMCLIP.forward:559 |
| 27–28 | GSPL 多头 cross attention 展开 | gs_attn；nn.MultiheadAttention |
| 29–30 | batch-wise 双向相似度分布 | motion_text_kl_loss 的双向 logits / log_softmax |

## 12. 验证结果和边界

- `python -m pytest tests -q`：**82 passed**，19.93s。
- `python -B scripts/audit_emclip_paper.py`：小模型 synthetic forward/backward 有限；可训练参数无 None gradient、无非有限 gradient；上述结构问题均可重现。
- `python -B scripts/audit_emclip_paper.py --ddp-smoke`：双进程 CPU Gloo、FileStore、find_unused_parameters=False、连续两个 optimizer step 通过；两 rank 参数最大差异 0.0，rank 0 loss 为 8.2597、6.8526。Windows 当前 PyTorch 缺 libuv，torchrun 的 TCP rendezvous 两次尝试失败，改用 FileStore 验证了 DDP 计算路径。这不等于通过 CUDA/NCCL 或 GPU fp16 检查。
- 28 个项目 Python 文件 AST 语法检查、`git diff --check`：通过。
- 本机 PyTorch 2.7.0+cpu，无 CUDA、无 coviar 原生扩展、无本地 CLIP/EM-CLIP checkpoint。
- 四个数据集配置的真实 root/list 在本机均不存在。**真实 CoViAR 样本读取、SSV2 GOP 分布、真实类别 JSON 与列表逐视频对应、真实 checkpoint 数值、4×3 精度、CUDA AMP 均未在本机实测**。
- 未进行完整训练，未联网下载模型，未修改数据集原始文件。

## 13. 建议的排查顺序

1. 先保留当前 SSV2 `model_best.pth`，用它补做 4×3；不要误评 K400 权重。核实当前 best 和日志 epoch 13 是否一致，日志只覆盖了 14 个完整 epoch。
2. 修复 SSV2 flip 标签语义，核验 train/val ID 与官方 JSON 对应。对真实视频输出 I/MV/R 值域、实际编码/GOP 间隔、候选重复率和 selected unique GOP 数。特别核对服务器实际使用的 CoViAR 扩展。
3. 实现 GSPL/LMPL 共用 Eq. (16) Residual 特征的明确变体；记录旧 checkpoint 迁移方法。核对 R 通道和初始化分布，逐项做对照。
4. 在数据核验完成后，以原始 CLIP 起点的 diamond T=K=8 为一个清晰基线，再比较 full class_bank。比较可以先用固定小规模训练子集检验趋势；完整精度仍需要独立训练。
5. 单独验证时间位置编码、freeze_text、初始化来源、增强强度等影响；每次只改变一个因素。时间编码应标为扩展/消融，论文未给出的 optimizer 或冻结规则也应明确记录。

现有日志并不支持直接用“再训练十几轮”“加大学习率”或“测试时用真实类别选帧”解决问题。应先修正确定错误并隔离泛化原因；本报告不承诺修正任一项后必然达到论文 67.3%。
