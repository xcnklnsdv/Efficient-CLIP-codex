codex exec \
  --sandbox workspace-write \


  - <<'CODEX_PROMPT'
你现在要在当前仓库中完整复现论文：

Efficient Motion-Centric CLIP for Compressed Video Action Recognition
简称：EM-CLIP

这不是只写设计文档的任务。你必须检查现有仓库，实际修改代码、补充配置、训练脚本、测试代码和复现文档，最终给出可以运行的实现。

============================================================
一、总目标
============================================================

基于当前仓库已有的：

1. CLIP ViT-B/16 视觉编码器和文本编码器；
2. 视频动作识别训练、验证和 DDP 框架；
3. CoViAR 压缩视频读取功能；
4. 现有数据集配置、类别文本、checkpoint、日志和模型保存逻辑；

实现论文中的完整 EM-CLIP，包括：

1. Motion-Guided Saliency Extraction，简称 MGSE；
2. Action Correlation Generator，简称 ACG；
3. Motion-Embedded Long-Term Spatiotemporal Correlation，简称 MELSC；
4. Intra and Inter Feature Embedding，简称 IFE；
5. Global Spatial Prompt Learning，简称 GSPL；
6. Long-Term Motion Prompt Learning，简称 LMPL；
7. Spatio-Temporal Aggregation Transformer，简称 SAG；
8. MGSE 双向运动—文本对齐损失 L_MG；
9. MELSC 视频—文本分类损失 L_ME；
10. 完整训练、验证、推理、checkpoint、DDP 和 AMP 支持；
11. EM-CLIP 和 EM-CLIP-diamond 两种论文变体；
12. HMDB51、UCF101、K400、SSV2 四个压缩域数据集的运行脚本。

不要只给出伪代码，不要只生成 TODO，不要只创建空壳类。

============================================================
二、开始修改前必须完成的检查
============================================================

首先检查整个仓库，至少检查：

- AGENTS.md；
- README；
- main.py 或其他训练入口；
- model、models、clip、models_adapter 等模型目录；
- dataset、datasets、dataloader 等数据目录；
- utils、loss、engine、config；
- 现有 CoViAR 读取代码；
- 现有 DDP、AMP、checkpoint 和评价代码；
- 现有文本 prompt、类别文件和类别映射；
- 当前 dataset 配置字典所在文件；
- 当前模型 forward 的输入输出接口；
- 当前训练 loss 的构成；
- 当前 Git 工作区中已有但未提交的修改。

先形成实现计划，再立即执行实现。

必须遵守：

1. 优先复用现有模块，避免建立第二套完全独立的训练框架。
2. 不得执行 git reset、git checkout、git clean、强制覆盖或删除用户修改。
3. 不得修改数据集原始文件。
4. 新功能必须通过参数控制，原有模型和训练命令不得失效。
5. 不允许吞掉异常；路径、shape、类别映射错误要给出明确报错。
6. 不进行完整训练，只运行静态检查、单元测试和小规模 smoke test。
7. 所有关键 tensor shape 都写注释和运行时断言。
8. 代码兼容当前仓库使用的 Python 和 PyTorch 版本，不要擅自升级环境。
9. 不要通过联网下载模型；优先使用仓库已有的 CLIP checkpoint 加载方式。
10. 新增代码风格应与当前项目保持一致。

============================================================
三、论文配置和模型变体
============================================================

实现以下模型变体：

A. emclip_b16_k8，默认完整模型

- backbone：CLIP ViT-B/16；
- 候选 GOP 数 T=16；
- MGSE 最终选择 K=8；
- 输入：I + MV + Residual；
- 启用 MGSE；
- 启用 MELSC；
- 图像分辨率：256；
- GOP_SIZE=12；
- temperature tau=0.01；
- Transformer 层数：12；
- hidden width：768；
- heads：12；
- CLIP 图文公共嵌入维度按照仓库 checkpoint 自动读取，通常为 512。

B. emclip_b16_k16

- T=32；
- K=16；
- 其他设置同上。

C. emclip_diamond_b16_k8

- 不使用 MGSE；
- 按 TSN 风格随机或分段采样 K=8 个 GOP；
- 输入仅 I + Residual；
- 启用 MELSC；
- 用于论文中的 EM-CLIP-diamond 消融和对比。

D. emclip_diamond_b16_k16

- K=16；
- 其他设置同 C。

所有数值都必须配置化，不能把 T、K、tau、GOP size、图像尺寸散落硬编码在模型中。

============================================================
四、数据集配置
============================================================

将下面配置合并到当前项目的数据集配置系统中。已有同名配置时进行兼容性修改，不得建立冲突的第二份来源。

DATASETS = {
    'ssv2': dict(
        TRAIN_ROOT='/mnt/data/sthv2/videos',
        VAL_ROOT='/mnt/data/sthv2/videos',
        TRAIN_LIST='/home/fuh/m2clip/configs/ssv2_train.txt',
        VAL_LIST='/home/fuh/m2clip/configs/ssv2_val.txt',
        NUM_CLASSES=174,
    ),

    'ssv2_mpeg4': dict(
        TRAIN_ROOT='/mnt/data/sthv2/mpeg4_video/20bn-something-something-v2',
        VAL_ROOT='/mnt/data/sthv2/mpeg4_video/20bn-something-something-v2',
        TRAIN_LIST='/home/fuh/CMPT/lists/sthv2/train_rgb.txt',
        VAL_LIST='/home/fuh/CMPT/lists/sthv2/val_rgb.txt',
        NUM_CLASSES=174,
        USE_COVIAR=True,
        RETURN_MV_RES=True,
        COMPRESSED_VIDEO_ROOT='/mnt/data/sthv2/mpeg4_video/20bn-something-something-v2',
        GOP_SIZE=12,
    ),
    
    'hmdb51_mpeg4': dict(
        TRAIN_ROOT='/mnt/data/hmdb51/mpeg4_videos',
        VAL_ROOT='/mnt/data/hmdb51/mpeg4_videos',
        TRAIN_LIST='/home/fuh/CMPT/lists/hmdb51/train_rgb_split_1.txt',
        VAL_LIST='/home/fuh/CMPT/lists/hmdb51/val_rgb_split_1.txt',
        NUM_CLASSES=51,
        USE_COVIAR=True,
        RETURN_MV_RES=True,
        COMPRESSED_VIDEO_ROOT='/mnt/data/hmdb51/mpeg4_videos',
        GOP_SIZE=12,
    ),
    
    'ucf101_mpeg4': dict(
        TRAIN_ROOT='/mnt/data/ucf101/Cucf101',
        VAL_ROOT='/mnt/data/ucf101/Cucf101',
        TRAIN_LIST='/home/fuh/CMPT/lists/ucf101/train_rgb_split_1.txt',
        VAL_LIST='/home/fuh/CMPT/lists/ucf101/val_rgb_split_1.txt',
        NUM_CLASSES=101,
        USE_COVIAR=True,
        RETURN_MV_RES=True,
        COMPRESSED_VIDEO_ROOT='/mnt/data/ucf101/Cucf101',
        GOP_SIZE=12,
    ),
    
    'k400': dict(
        TRAIN_ROOT='/mnt/data/CKinetics',
        VAL_ROOT='/mnt/data/CKinetics',
        TRAIN_LIST='/mnt/data/CKinetics/datalist/k400_train.txt',
        VAL_LIST='/mnt/data/CKinetics/datalist/k400_val.txt',
        NUM_CLASSES=400,
        USE_COVIAR=True,
        RETURN_MV_RES=True,
        COMPRESSED_VIDEO_ROOT='/mnt/data/CKinetics',
        GOP_SIZE=12,
    ),
}

数据集读取器必须自动兼容当前列表文件的实际格式。检查真实列表内容后再实现，不要仅凭文件名猜测格式。

至少支持：

- relative_path num_frames label
- video_id num_frames label
- relative_path label
- 带空格类别名映射的情况

需要完成以下验证：

1. label 必须落在 [0, NUM_CLASSES-1]；
2. 路径不存在时显示数据集名、原始列表行、解析后路径；
3. 类别数和类别文本数必须一致；
4. 视频后缀不得无条件重复拼接；
5. SSV2 的数字 ID 路径要按现有视频实际后缀解析；
6. K400 需要处理列表路径已经包含 train/val 子目录的情况；
7. 不能在 Dataset.__getitem__ 中静默返回其他样本掩盖坏数据。

============================================================
五、压缩视频采样和 CoViAR 读取
============================================================

论文使用 MPEG-4 GOP：

- 每个 GOP 共 12 帧；
- 第 0 帧为 I-frame；
- 后面最多 11 个 P-frame；
- MGSE 使用候选 GOP 最后一个有效 P-frame的 MV；
- MELSC 使用对应 GOP 的 I-frame和最后一个有效 P-frame的累积 Residual。

实现统一的压缩视频样本结构，例如：

CompressedVideoSample:
    video_path
    label
    candidate_gop_indices
    i_frames       [T, 3, H, W]
    motion_vectors [T, 2, H, W]
    residuals      [T, 3, H, W]
    valid_mask     [T]
    metadata

具体要求：

1. 将视频按照 GOP 数划分为 T 个非重叠时间段。
2. 训练时从每段随机选一个 GOP。
3. 验证时采用确定性的中心 GOP。
4. 支持多 temporal view，验证中不同 view 使用不同等距偏移。
5. GOP 数不足 T 时必须使用合理的重复采样或均匀索引，并返回 valid mask。
6. 最后一个不完整 GOP 使用该 GOP 最后一个有效 P-frame。
7. 没有有效 P-frame的 GOP要有明确 fallback，不得越界。
8. Residual 必须读取为相对于 GOP I-frame的累积残差。
9. MV 使用最后一个有效 P-frame的运动向量，并明确当前 coviar accumulate 参数的实际语义。
10. 在复现文档中记录 MV 是否使用 accumulate，以及选择依据。
11. 复用当前仓库已经验证过的 coviar.load 或 read_compressed 接口。
12. 不允许重新使用 OpenCV 全解码来伪造 MV 或 Residual。
13. Dataset 输出统一为：
    I  [T, 3, 256, 256]
    MV [T, 2, 256, 256]
    R  [T, 3, 256, 256]
14. batch 后为：
    I  [B, T, 3, 256, 256]
    MV [B, T, 2, 256, 256]
    R  [B, T, 3, 256, 256]

空间增强必须同步作用于 I、MV、R：

- crop 参数一致；
- resize 参数一致；
- horizontal flip 一致；
- MV 水平翻转后 x 分量取负；
- resize 后 MV 的 x、y 数值按宽高缩放比例同步修正；
- 不得把 MV 当成普通 RGB 图像直接增强。

沿用当前项目已验证的压缩域归一化策略，并将其集中配置化。默认候选值：

- MV clamp 到 [-20, 20] 后除以 20；
- Residual clamp 到 [-1, 1]；
- I-frame 使用 CLIP mean/std；
- Residual 是否使用 CLIP mean/std由现有实现和数据实际范围决定；
- 在文档中说明最终选择。

============================================================
六、MGSE：Motion-Guided Saliency Extraction
============================================================

新增独立、可测试的 MGSE 模块。

建议接口：

class MotionGuidedSaliencyExtraction(nn.Module):
    def forward(
        self,
        motion_vectors,       # [B,T,2,H,W]
        class_text_features,
        labels=None,
        valid_mask=None,
        training_mode=True,
    ):
        ...
        return {
            "selected_indices": ...,
            "saliency": ...,
            "motion_frame_features": ...,
            "motion_video_features": ...,
            "mg_logits_mv2text": ...,
            "mg_logits_text2mv": ...,
        }

------------------------------------------------------------
6.1 Motion Encoder
------------------------------------------------------------

1. 使用独立的预训练 CLIP ViT-B/16 视觉 Transformer作为 MV encoder。
2. patch size=16。
3. MV 每个 patch 输入维度为 16×16×2=512。
4. 输出 Transformer hidden width=768。
5. 使用 CLS token 和二维位置编码。
6. 将原始 CLIP 三通道 patch embedding修改为两通道。

两通道初始化策略：

rgb_weight = clip_conv1.weight
mv_weight = rgb_weight.mean(dim=1, keepdim=True).repeat(1, 2, 1, 1)

为保持量级，可根据当前初始化规范乘以 3/2。将该策略封装成函数并写单元测试。

7. 256×256 输入对应 16×16 patches。
8. 将 CLIP 原始位置编码通过 bicubic interpolation插值到 16×16。
9. 不得简单截断位置编码。
10. 最终取最后一层 CLS：
    f_t = z_mv_last[:, CLS]
11. 经 LayerNorm 和 projection映射到 CLIP 图文公共空间：
    v_t = projection(LN(f_t))
12. 输出：
    motion_frame_features [B,T,D_text]

------------------------------------------------------------
6.2 Text Encoder
------------------------------------------------------------

prompt 模板默认严格使用：

"a photo of a {label}"

同时允许配置多个模板进行平均，但默认必须保留论文模板。

需要支持两类文本表示：

1. class-level EOT feature：
   [C,D_text]
2. label word token features：
   每个类别保留有效 label word token，使用 mask 对齐；
   [C,S_max,D_text]

不得把 padding、SOT、EOT 当作普通类别词平均。

文本编码器可训练时，训练阶段不得长期缓存旧文本特征；验证阶段可以安全缓存。

------------------------------------------------------------
6.3 Action Correlation Generator
------------------------------------------------------------

按照论文公式实现：

1. 对每个 motion frame feature 做逐特征标准化或 LayerNorm；
2. 对文本 token feature做相同处理；
3. 再进行 L2 normalize；
4. 计算 cosine similarity；
5. 除以 tau，默认 tau=0.01；
6. softmax 必须沿候选时间维 T；
7. 对有效文本词维度取平均；
8. 使用 valid_mask 排除无效候选；
9. 得到 saliency [B,T]；
10. 使用 torch.topk 选 K 个候选；
11. 对 top-k 索引重新按时间升序排序；
12. 再从 I 和 Residual 中 gather对应帧；
13. gather 必须支持 batch，不能写错误的 Python for-loop索引；
14. 返回原始 saliency 和 selected_indices便于可视化。

为了避免 tau=0.01 在 AMP 下出现 NaN：

- cosine、temperature division、softmax、log_softmax 和 loss 均使用 float32；
- 使用稳定的 PyTorch softmax/log_softmax；
- 不得先 exp 再手动除法；
- 检查所有输出 isfinite；
- 不要通过把 NaN 替换为零来掩盖问题。

------------------------------------------------------------
6.4 论文中的标签使用歧义和泄漏保护
------------------------------------------------------------

论文的 MGSE 描述将类别 label text 作为输入，但验证阶段使用真实标签选帧会造成标签泄漏。

因此必须实现以下模式：

A. mgse_text_mode="ground_truth"

- 训练阶段使用该样本真实类别的 label word token；
- 仅用于论文公式诊断或消融；
- 默认禁止在 validation/test 使用；
- eval 时发现此模式必须抛出明确异常；
- 只有显式指定：
  --allow-mgse-label-leakage-for-diagnostic
  才允许运行，并在日志中打印醒目警告。

B. mgse_text_mode="class_bank"

- 验证和正式结果的默认模式；
- 使用全部类别文本，不读取样本真实标签；
- 对每个类别分别在 T 维做 temporal softmax；
- 再对类别或有效 label token维聚合；
- 默认聚合为 mean；
- 同时支持 max 和 logsumexp 配置；
- 正式推理默认使用该模式。

C. mgse_text_mode="predicted_class"

- 可选模式；
- 先根据 motion 全局特征预测类别，再使用预测类别文本计算 saliency；
- 预测过程中不得使用 ground truth。

README 和实现说明必须明确区分上述三种模式。

============================================================
七、L_MG：运动—文本双向对齐损失
============================================================

实现论文中的双向 KL 对齐损失：

L_MG = 0.5 * (
    KL(p_mv2w || q_mv2w) +
    KL(p_w2mv || q_w2mv)
)

要求：

1. 将 T 个 motion frame feature按 valid mask 聚合为视频级 motion feature。
2. 默认使用 saliency-weighted pooling。
3. 也支持 mean pooling用于消融。
4. motion feature和对应类别 EOT text feature计算 batch-wise similarity。
5. 实现 mv-to-text和text-to-mv两个方向。
6. tau 默认 0.01。
7. 同类别样本不能全部被当作负样本。
8. 构造 multi-positive target：
   target[i,j] = 1，当 labels[i] == labels[j]；
   每行归一化为概率分布。
9. 使用：
   F.kl_div(
       F.log_softmax(logits.float(), dim=-1),
       target.float(),
       reduction="batchmean"
   )
10. 不使用 reduction="mean"，避免 PyTorch KLDiv 语义警告。
11. 支持 DDP 全局负样本和全局同类正样本。
12. 优先使用带梯度的 all_gather。
13. labels 可以普通 all_gather。
14. 单卡模式必须正常运行。
15. batch size=1 时 loss 仍然有限，不产生 NaN。
16. 返回：
    loss_mg
    loss_mg_mv2text
    loss_mg_text2mv

============================================================
八、MELSC：长程时空相关模块
============================================================

新增 MotionEmbeddedLongTermSpatiotemporalCorrelation 模块。

输入：

I_selected [B,K,3,H,W]
R_selected [B,K,3,H,W]

输出：

video_features [B,D_text]
以及调试用中间结果。

------------------------------------------------------------
8.1 IFE：I-frame和Residual特征嵌入
------------------------------------------------------------

1. I 和 Residual 使用两个独立视觉分支。
2. 两个分支都初始化自同一个预训练 CLIP ViT-B/16 checkpoint。
3. 参数对象必须独立，不能无意共享同一个 nn.Module。
4. I patch embedding直接复制 CLIP RGB conv1。
5. Residual patch embedding也复制 CLIP RGB conv1作为初始化。
6. 各自拥有 CLS token、位置编码、pre-LN、Transformer。
7. 位置编码插值到 256×256对应的 16×16 patch grid。
8. 内部统一 tensor layout：
   z_i [B,K,N,D]
   z_r [B,K,N,D]
   其中 N=1+16×16=257，D=768。
9. 每层送入 CLIP block时 reshape为：
   [B*K,N,D]
10. 需要显式检查 reshape/permute 后 contiguous或使用 reshape安全处理。

------------------------------------------------------------
8.2 GSPL
------------------------------------------------------------

每个 Transformer layer l 都执行：

a_i = W_i(cls_i)
q_i = LN(a_i)

其中：

cls_i = z_i[:,:,0,:]  # [B,K,D]
a_i   = [B,K,D]

从 Residual 分支构造：

a_r = W_r(cls_r)
q_r = k_r = v_r = LN(a_r)

实现 Intra-frame Spatial Interaction Attention，论文记为 ISIA：

gs_hat = MultiHeadAttention(
    query=q_i,
    key=k_r,
    value=v_r
)

这里 attention 沿 K 个时间位置执行，因此：

q_i [B,K,D]
k_r [B,K,D]
v_r [B,K,D]

得到：

gs = a_i + dropout(gs_hat)
gs [B,K,D]

随后变成每个 frame的一个 prompt token：

gs_prompt [B,K,1,D]

使用 12 个 attention heads。

------------------------------------------------------------
8.3 LMPL
------------------------------------------------------------

使用 Residual CLS token：

a_r = W_r(cls_r)
q_r = k_r = v_r = LN(a_r)

执行 Inter-frame Motion Interaction Attention，论文记为 IMIA：

lm_hat = MultiHeadAttention(
    query=q_r,
    key=k_r,
    value=v_r
)

得到：

lm = a_r + dropout(lm_hat)
lm_prompt [B,K,1,D]

LMPL 必须在 K 个 frame之间建模，不能错误地只在单帧 patch维做 self-attention。

------------------------------------------------------------
8.4 SAG
------------------------------------------------------------

每一层将两个 prompt token加入 I-frame token序列：

i_with_prompt = concat(
    z_i,
    gs_prompt,
    lm_prompt,
    dim=token_dimension
)

shape：

[B,K,N+2,D]

然后 reshape为：

[B*K,N+2,D]

送入对应层的 CLIP MHSA + FFN。

每层结束后：

1. 保留更新后的原始 N 个 I tokens；
2. 本层 prompt token不直接累积到下一层；
3. 下一层根据新的 I/R CLS重新生成 gs和lm；
4. Residual 分支按照该层自己的 CLIP MHSA + FFN更新；
5. Residual 分支不拼接 gs/lm；
6. 每层 shape 都写断言。

使用 CLIP 原始 block权重时，需要确保额外 prompt token不会破坏 attention mask；视觉 Transformer应为全注意力，不得使用文本 causal mask。

------------------------------------------------------------
8.5 最终视频聚合
------------------------------------------------------------

最后一层得到 I-frame CLS：

frame_cls = z_i[:,:,0,:]  # [B,K,768]

按照论文公式实现视频级聚合：

1. 对 K 个 frame CLS执行 temporal MHSA；
2. 再经过 FFN；
3. 对时间维做 AvgPool；
4. 经 final LayerNorm；
5. 经 CLIP visual projection映射到 D_text；
6. 做 L2 normalize；
7. 输出 video_features [B,D_text]。

Temporal aggregator至少使用一个标准 pre-norm Transformer block，并把层数配置化，默认 1。

============================================================
九、L_ME和总损失
============================================================

类别文本特征：

class_text_features [C,D_text]

最终 logits：

logits = logit_scale.exp() * video_features @ class_text_features.T

其中：

- video_features已归一化；
- class_text_features已归一化；
- logit_scale沿用 CLIP 参数；
- 对 logit_scale进行与当前 CLIP实现一致的安全限制。

分类损失：

loss_me = F.cross_entropy(logits.float(), labels)

总损失：

loss = lambda_mg * loss_mg + lambda_me * loss_me

默认：

lambda_mg = 1.0
lambda_me = 1.0

emclip_diamond不使用 MGSE 时：

loss_mg = 0
loss = loss_me

日志中分别打印：

- loss
- loss_mg
- loss_mg_mv2text
- loss_mg_text2mv
- loss_me
- acc1
- acc5
- mgse_saliency_entropy
- selected GOP indices统计
- learning rate
- grad norm
- max memory

不能把分类 CE、CLIP CE 和其他已有 loss重复计算。检查当前训练代码，明确最终 loss组成。

============================================================
十、预训练参数、冻结和初始化
============================================================

支持以下训练策略：

--emclip-train-mode full
    端到端训练视觉、文本和新增模块。
    作为论文主复现默认模式。

--emclip-train-mode freeze_text
    冻结 CLIP text encoder，训练三个视觉分支及新增模块。

--emclip-train-mode freeze_clip
    冻结所有 CLIP原始参数，仅训练 MGSE projection、GSPL、LMPL、
    temporal aggregator等新增参数。

要求：

1. 参数冻结必须在 optimizer构建前完成。
2. optimizer只接收 requires_grad=True参数。
3. 日志打印总参数量和可训练参数量。
4. DDP 默认 find_unused_parameters=False。
5. 不同变体中未使用的模块必须冻结或根本不创建，不能依赖
   find_unused_parameters=True掩盖结构问题。
6. 新增 Linear、attention、MLP使用项目统一初始化；不存在时使用
   Xavier uniform和零 bias。
7. 所有论文没有说明的初始化选择记录在实现文档中。

============================================================
十一、训练配置
============================================================

论文明确配置：

- epochs=30
- initial lr=8e-6
- scheduler=cosine annealing
- image size=256
- tau=0.01
- attention dropout=0.0
- MLP dropout=0.0
- stochastic depth=0.0
- backbone=CLIP ViT-B/16
- GOP size=12
- 默认 T=16
- 默认 K=8

论文没有明确说明 optimizer、weight decay、batch size、warmup。
将下面内容作为“工程默认值”，但必须可配置并在文档标记为假设：

- optimizer=AdamW
- betas=(0.9,0.98)
- eps=1e-6
- weight_decay=0.2
- warmup_epochs=0
- batch_size_per_gpu=4
- grad_clip_norm=1.0
- seed=1024
- AMP enabled
- num_workers=8
- pin_memory=True

学习率默认直接使用 8e-6，不因 world size自动线性放大，除非显式传入：
--scale-lr-by-global-batch

训练代码必须支持：

- torchrun DDP；
- DistributedSampler；
- sampler.set_epoch(epoch)；
- AMP；
- GradScaler；
- 梯度裁剪；
- resume；
- save latest；
- save best Top-1；
- 自动恢复 optimizer、scheduler、scaler和epoch；
- rank0写日志和checkpoint；
- 所有 rank正确参与 validation统计；
- all_reduce正确汇总样本数和指标；
- 不得对各 rank百分比直接取平均。

============================================================
十二、训练和验证视图
============================================================

训练：

- 1 temporal view；
- 1 spatial crop；
- 分段随机候选 GOP；
- synchronized random crop和flip。

快速验证默认：

- 1 temporal view；
- 1 center crop。

论文对比设置：

- 支持 4 temporal views × 3 spatial crops；
- 将所有 view的 logits平均后计算最终预测；
- 不得把同一视频的不同 view当成独立样本计算准确率；
- 提供命令行参数：
  --test-num-temporal-views
  --test-num-spatial-crops

============================================================
十三、命令行参数
============================================================

在当前训练入口中集成参数，或在仓库架构明显不适合时新增
main_emclip.py，但必须复用原有 engine和工具。

至少支持：

--model emclip_b16
--dataset ssv2_mpeg4
--emclip-variant emclip
--candidate-frames 16
--selected-frames 8
--gop-size 12
--input-size 256
--mgse-temperature 0.01
--mgse-text-mode class_bank
--mgse-class-aggregation mean
--motion-pooling saliency
--lambda-mg 1.0
--lambda-me 1.0
--emclip-train-mode full
--temporal-aggregator-layers 1
--epochs 30
--lr 8e-6
--weight-decay 0.2
--batch-size 4
--num-workers 8
--amp
--resume
--output-dir
--eval
--test-num-temporal-views
--test-num-spatial-crops
--debug-shapes
--verify-compressed-inputs

参数名需遵循当前项目已有风格；若已有同义参数则复用，不得重复建立两套。

============================================================
十四、代码组织
============================================================

根据当前仓库实际结构决定最终路径。建议但不强制：

models/emclip.py
models/emclip_mgse.py
models/emclip_melsc.py
datasets/compressed_video_dataset.py
losses/emclip_loss.py
configs/emclip_datasets.py
engine_emclip.py
main_emclip.py
tests/test_emclip_shapes.py
tests/test_emclip_mgse.py
tests/test_emclip_dataset.py
tests/test_emclip_losses.py
scripts/train_emclip_ssv2.sh
scripts/train_emclip_hmdb51.sh
scripts/train_emclip_ucf101.sh
scripts/train_emclip_k400.sh
scripts/eval_emclip_ssv2.sh
scripts/eval_emclip_hmdb51.sh
scripts/eval_emclip_ucf101.sh
scripts/eval_emclip_k400.sh
scripts/smoke_emclip.sh
IMPLEMENTATION_NOTES_EMCLIP.md

仓库已有相应文件时，应在现有文件中集成，而不是机械创建上述所有文件。

============================================================
十五、运行脚本
============================================================

创建四个训练脚本，默认使用4张 GPU，但允许环境变量覆盖：

NPROC_PER_NODE=${NPROC_PER_NODE:-4}
BATCH_SIZE=${BATCH_SIZE:-4}
MASTER_PORT=${MASTER_PORT:-29501}
OUTPUT_ROOT=${OUTPUT_ROOT:-output_dir/emclip}

脚本必须使用：

torchrun \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --master_port="${MASTER_PORT}" \
  <实际训练入口> \
  ...

四个数据集分别使用：

1. ssv2_mpeg4
2. hmdb51_mpeg4
3. ucf101_mpeg4
4. k400

默认完整模型：

--emclip-variant emclip
--candidate-frames 16
--selected-frames 8
--mgse-text-mode class_bank
--epochs 30
--lr 8e-6
--input-size 256
--amp

每个脚本：

1. 使用 set -euo pipefail；
2. 创建输出目录；
3. 保存完整命令；
4. 将 stdout/stderr通过 tee写入带时间戳日志；
5. 支持 RESUME环境变量；
6. 支持 CUDA_VISIBLE_DEVICES由用户外部设置；
7. 不在脚本中强制固定具体 GPU编号；
8. 输出目录中包含数据集名、模型变体、T、K和时间戳。

同时创建 emclip_diamond的示例运行命令。

============================================================
十六、测试
============================================================

至少完成以下测试：

------------------------------------------------------------
16.1 Shape测试
------------------------------------------------------------

使用较小的 synthetic输入测试：

B=2
T=4
K=2
H=W=64

验证：

- MV encoder输出 [B,T,D_text]；
- saliency输出 [B,T]；
- top-k输出 [B,K]；
- selected I/R输出 [B,K,3,H,W]；
- GSPL输出 [B,K,D]；
- LMPL输出 [B,K,D]；
- MELSC输出 [B,D_text]；
- logits输出 [B,C]；
- loss为标量且 finite。

模型必须支持位置编码动态插值，因此测试分辨率可以小于256。

------------------------------------------------------------
16.2 Top-k和mask测试
------------------------------------------------------------

验证：

- 无效候选不会被选中；
- top-k结果最终按时间排序；
- 每个 batch样本独立 gather；
- K>T或有效候选少于K时给出清晰处理；
- 不发生跨 batch错误索引。

------------------------------------------------------------
16.3 数据增强测试
------------------------------------------------------------

验证：

- 水平翻转后 MV x分量变号；
- y分量不变号；
- resize后 MV x/y按比例变化；
- I/MV/R crop位置完全相同。

------------------------------------------------------------
16.4 Loss测试
------------------------------------------------------------

验证：

- 相同类别的两个样本构成 multi-positive target；
- reduction使用 batchmean；
- batch size=1有限；
- AMP环境下 tau=0.01不产生 NaN；
- DDP helper在未初始化分布式时正常退化为单卡。

------------------------------------------------------------
16.5 标签泄漏测试
------------------------------------------------------------

验证：

- eval + ground_truth模式默认抛出异常；
- class_bank模式完全不读取 labels来计算 selected_indices；
- 改变 labels不应改变 class_bank模式的 selected_indices。

------------------------------------------------------------
16.6 真实数据 smoke test
------------------------------------------------------------

在路径存在时：

- 每个数据集读取1个样本；
- 打印 path、label、GOP数、candidate indices；
- 打印 I/MV/R shape、dtype、min、max、mean；
- 运行一次 no_grad forward；
- 不进行完整 epoch训练。

数据路径不存在或 coviar不可用时，不要伪造成功；输出明确的跳过原因。

------------------------------------------------------------
16.7 工程检查
------------------------------------------------------------

运行适合当前仓库的：

- Python语法检查；
- import测试；
- 单元测试；
- 一个 synthetic forward/backward；
- git diff --check。

不要因为仓库原有无关测试失败而大范围重构；需要区分本次修改导致的错误和仓库原有错误。

============================================================
十七、验收标准
============================================================

最终代码必须满足：

1. 原有模型仍可运行。
2. EM-CLIP可通过配置创建。
3. forward中真实使用 I、MV、Residual。
4. MGSE真实影响 selected_indices。
5. selected_indices真实决定 MELSC输入。
6. GSPL和LMPL逐层执行。
7. GSPL使用 I query和Residual key/value。
8. LMPL沿 K 帧执行 temporal self-attention。
9. SAG把 gs和lm作为两个 prompt token拼入每帧 I token。
10. L_MG和L_ME分别计算。
11. total loss没有重复添加已有 CLIP loss。
12. DDP下无未使用可训练参数。
13. AMP下不出现 NaN。
14. validation默认不存在真实标签选帧泄漏。
15. 四个数据集都有训练和评估脚本。
16. checkpoint保存 latest和best。
17. 支持 resume。
18. README中给出完整运行命令。
19. 所有论文未明确的实现决定都有记录。
20. 不得声称一定复现论文精度。

============================================================
十八、复现说明文档
============================================================

创建 IMPLEMENTATION_NOTES_EMCLIP.md，至少包含：

1. 论文模块与代码文件对应关系；
2. 公式1至公式30的代码对应关系；
3. 每个关键 tensor shape；
4. CoViAR具体调用方式；
5. last P-frame和累积 Residual定义；
6. MV accumulate参数的选择；
7. 256输入的位置编码插值；
8. MV两通道 patch embedding初始化；
9. I、R、MV三个视觉分支是否独立；
10. GSPL、LMPL、SAG实现细节；
11. MGSE标签泄漏问题及三种 text mode；
12. multi-positive KL target；
13. DDP all-gather实现；
14. 论文明确超参数；
15. 论文未说明而采用的工程假设；
16. 与论文可能存在的差异；
17. 四个数据集训练命令；
18. 4×3 view评估命令；
19. smoke test命令；
20. 常见错误排查。

明确写出以下事实：

“由于论文未公开源码，并且没有完整披露优化器、batch size、
参数冻结策略、MGSE测试阶段类别文本来源等细节，本实现属于
基于论文公式和描述的工程复现，不能保证与作者私有实现逐行一致。”

============================================================
十九、最终输出
============================================================

完成修改和测试后，在最终回复中给出：

1. 实际修改和新增的文件列表；
2. 模型整体数据流；
3. 关键实现决定；
4. 论文未说明的假设；
5. 已执行测试及结果；
6. 未执行测试及原因；
7. 四个数据集的实际训练命令；
8. 一个单卡 smoke test命令；
9. 一个4卡正式训练命令；
10. 当前仍存在的风险和限制。

不要只回复“已完成”。
CODEX_PROMPT