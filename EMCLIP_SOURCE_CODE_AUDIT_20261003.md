# 按原始论文检查训练与模型（2026-10-03）

以用户指定的 PDF 为依据：第 3–6 页正文、Algorithm 1、Figure 2/3、
公式（1）–（26），第 12 页附录 B、公式（27）–（30），以及第 13 页
Table C.1。已实际提取原文并渲染查看 Figure 3；AGENTS.md 仅代表仓库
要求，不能替代论文事实。

## 查到并修复的问题

### 1. MELSC 时间聚合前额外的归一化

公式（23）直接把最后一层的 I CLS `z_I[L][:,:,0,:]` 送入时间 MHSA/FFN。
旧代码先对全部 I token 使用 `i_encoder.ln_post`，再送入时间 Transformer，
并在平均池化后另外执行 `final_ln`。额外的预归一化没有正文/图示依据。
合成检查中原始 CLS 与实际时间聚合输入的最大差为 1.35193，不是数值舍入。

新 paper 训练使用 `melsc_norm_order=post_pool`：原始末层 CLS → 标准
pre-norm 时间 Transformer → AvgPool → 最终 LN/CLIP projection。
聚合后的 LN/projection、标准 Transformer 内部的残差/pre-norm 和最终 LN
初始化为 CLIP visual ln_post，均作为桥接 CLIP 语义空间的工程选择记录，
不宣称论文展开了这些实现细节。

修改 `models/emclip_melsc.py`、`models/emclip.py`、`main_emclip.py`。
未使用的原 I ln_post 参数冻结，optimizer/DDP 不接收无损失路径参数。

**适用于当前 HMDB51 日志的结构差异，但精度影响尚未实测。**

### 2. 加载新权重后复用旧文本特征

在一个已做过 eval 的模型上加载另一个 checkpoint 后，旧代码未清空
`PromptTextEncoder._eval_cache`。可复现：加载后的输出仍与加载前完全相同，
与重新编码的文本特征最大差为 0.720833。

现在在递归 state_dict 加载、设备/dtype 转换时清空缓存；仅真正的 no_grad
推理使用缓存。需要梯度的 eval forward 不复用缓存，从而避免断开文本梯度
或第二次 backward 复用已释放计算图。

修改 `models/emclip.py`。常规每轮训练本来会清空缓存，当前 HMDB51 也从
CLIP 新启动，因此没有证据把这次验证平台期归因于此 bug。

### 3. predicted_class 在未被对齐训练的空间预测类别

旧代码用公式（5）标准化后的 motion 预测 EOT 类别；`L_MG` 对齐的则是
公式（4）projected motion / EOT 特征。两者可能给出不同的类别。
合成检查中对齐空间预测 class 0，旧标准化空间预测 class 1。

现在先用有效 projected motion 的均值预测类别，再用该类别的 label-word
tokens 做公式（5）–（7）的相关性计算。无需真实标签。

修改 `models/emclip_mgse.py`。predicted_class 本身是无标签推理的工程扩展，
不是论文 Algorithm 1 明确给出的测试策略。当前 HMDB51 用 class_bank，
因此这一 bug 不影响那次运行的选帧。

## 已核对的主结构

| 论文内容 | 当前代码检查 |
| --- | --- |
| （1）–（4）MV patches、CLS、视觉编码及语义投影 | 2-channel stem，独立 CLIP 视觉参数，LN/projection；位置编码插值 |
| （5）–（8）标准化、cosine、沿 T softmax、top-k | paper 无仿射标准化；词 mask；float32；排序；batch gather |
| （9）–（10）I/R embedding | 独立参数与预训练权重，不共享 Module/storage |
| （11）–（14）、（27）–（28）GSPL | I query、R key/value；R 属性与（16）共用；注意力 key 维为 K |
| （15）–（18）LMPL | R CLS 的 K 帧 self-attention，未误用单帧 patch 维 |
| （19）–（22）SAG/I/R 更新 | 每层加入两个新 prompt；I/R block 独立；丢弃本层 prompt；视觉全注意力 |
| （23）时间聚合 | 修复输入的额外预归一化；其内部细节仍需标为假设 |
| （24）–（26）损失 | 仅一次 L_ME CE，加一次双向 L_MG KL；没有重复 CLIP CE |
| （29）–（30）batch 概率 | DDP 全局样本、同类正样本；float32 温度；带梯度 gather |
| 第 4.2 节 | 30 epochs、LR=8e-6、cosine、256、tau=0.01、dropout/stochastic depth=0 |
| Table C.1 | full T16/K8、T32/K16；diamond 仅 I/R、无 MGSE |

Residual 最后一层的结果没有通向 I-CLS 分类头：Figure 3 /（11）–（22）
在第 l 层生成 prompt 时读取的是 R 的第 l−1 层。因此最后的 R block 仍
执行，但冻结它和未使用的 R projection/LN，避免 DDP 未使用可训练参数。
这不是吞掉异常，也不意味着 R 分支整体未训练。

## 训练逻辑与梯度核验

- backward、AMP unscale、梯度裁剪、optimizer.step、scheduler.step 的顺序正确。
- AMP overflow 时跳过 update，并不推进 scheduler；正常参数更新实际发生。
- 训练文本重新计算，没有长期缓存旧训练文本；validation 无标签选帧。
- DDP validation 按正确数/总样本数汇总，不直接平均 rank 百分比；同视频
  多视图先平均 logits。
- 新增两进程 CPU/Gloo 测试：full 与 diamond 均以同一批输入，对照单进程
  全局 batch，比较两次更新的 loss、所有可训练参数的梯度和 AdamW 参数。
  同类正样本跨 rank 分布，`find_unused_parameters=False`。
- FP32 分组计算存在舍入差异；对梯度使用逐参数向量误差界
  `2e-5 * reference_norm + 3e-5`，参数使用 `rtol=3e-5, atol=3e-6`。
  没有发现 world-size 梯度缩放错误。

## 与 AGENTS.md 或“严格复现”表述的区别

1. 论文（25）明确用固定 tau；可学习 CLIP logit_scale 是历史实现行为，
   不应作为当前论文分类头默认。当前新 paper 模型已使用固定 tau。
2. Algorithm 1 将类别文字作为输入，但没有完整说明未知标签推理如何获得
   类别描述。训练 ground_truth、验证 class_bank 不是已经核验的作者协议。
3. 论文没有完整披露 optimizer、weight decay、batch、warmup、冻结策略、
   输入归一化、短视频补齐和 temporal position encoding。现有工程默认值
   不能仅凭 AGENTS.md 写为“论文设置”。
4. （24）正文将 p 称为 ground truth；附录 B 又将 p 定义为 softmax 预测，
   符号存在冲突。实现选用稳定的 KL(target || prediction)，并对同类正样本
   归一化；不是逐字照抄矛盾公式。
5. L_MG 的视频级 MV pooling 未完整定义；saliency-weighted pooling 属于假设。
6. Table 4 明确列出 K400/SSV2 的 4×3；HMDB51 Table 3 未单列其视图数。
   HMDB51 4×3 是当前可运行的评估选择，不能称为已确认的完整作者协议。
7. 第 4.2 节取最后 P-frame；累积 residual 符合其累积运动描述。MV 的
   accumulate boolean 没有明示，当前 False 是已记录的选择，非论文原句。

## 仍然需要实测的问题

当前 HMDB51 的 train ground_truth / eval class_bank 会改变选帧；类别平均
可以抵消 saliency，GOP<K 时补齐不同最高分 GOP 会改变重复次数。这些是
已确认的机制风险，不等于已测出其精度损失。full 在 3570 个样本上训练
约 3.92 亿参数也可能过拟合。不能用简单改 batch/lr 或增加轮数冒称修复。

默认无 temporal position encoding，联合重排 I/R GOP 具有顺序不敏感性。
但论文没有给出时间位置编码的完整定义，不能为追求精度而静默添加后称为
原论文结构。

真实编码是否为固定 GOP=12、native CoViAR MV/R 范围、全部类别映射、列表
是否重叠，以及本次 checkpoint 的验证消融，都需要服务器数据。此机没有
这些数据和 native decoder，不伪造通过，也没有执行完整训练。

## checkpoint 与重新运行

本次修改前的 paper/legacy checkpoint 没有 `melsc_norm_order`，自动恢复为
`pre_and_post`，保持其原始计算和 optimizer 参数顺序。新 checkpoint 保存
明确的 norm 配置。旧权重的可评估性不代表它已享有新的结构修复。

要验证修复后的新模型，从原始 CLIP 开始，保持独立输出目录：

```bash
RESUME="" INIT_CHECKPOINT="" K400_CHECKPOINT="" \
bash scripts/train_emclip_hmdb51.sh --melsc-norm-order post_pool
```

正式验证仍禁止真实标签选帧。不能承诺新训练一定达到论文 HMDB51 的 78.9%。
