# FSDETR 论文解读：频域-空间协同建模，让小目标检测不再"视而不见"

> **论文标题**：FSDETR: Frequency-Spatial Feature Enhancement for Small Object Detection  
> **作者团队**：Jianchao Huang, Fengming Zhang, Haibo Zhu, Tao Yan (江南大学)  
> **代码仓库**：[GitHub](https://github.com/YT3DVision/FSDETR)

---

## 一、研究背景与动机

小目标检测一直是计算机视觉领域的一块"硬骨头"。尤其在无人机航拍场景中，目标往往小于 (32 x  32) 像素，面临三重困境：

1. **深度下采样导致特征退化**：CNN 骨干每经过一层 stride=2 的下采样，小目标的像素信息就被压缩四倍。到深层特征图时，一个 (16 x 16 ) 的小人可能只剩下 1~2 个像素的响应。
2. **密集场景中的相互遮挡**：无人机视角下人群、车辆高度密集，目标之间严重重叠，传统检测器难以区分。
3. **复杂背景干扰**：城市航拍中纹理杂乱，小目标极易被背景噪声淹没。

![Fig. 1: Comparison on VisDrone2019](https://raw.githubusercontent.com/Liwx1014/PicBed/main/images/b0b081695ebcd22dfe6b60d92870e1dec602d9d9421158c8e5cfbf1fa8628e72.jpg)

*▲ 图1：VisDrone 2019 上各模型的 AP_S 对比。FSDETR 在同等参数规模下取得最高的 AP_S。*

论文还指出了一个关键空白：现有 DETR 系列检测器**几乎完全依赖空间域建模**，而频域学习方法（如 FFC、GFNet）虽已被证明能有效增强纹理和结构线索，却从未被整合到小目标检测器中。FSDETR 正是要填补这个空白——**将频域-空间协同建模引入 DETR 框架**。

---

## 二、核心创新点与整体架构

### 2.1 三大模块总览

FSDETR 以 RT-DETR 为基线，提出三个即插即用的改进模块：

| 模块 | 位置 | 解决的核心问题 | 技术手段 |
|:---|:---|:---|:---|
| **SHAB** | 骨干网络深层 | 长程依赖建模不足，小目标全局上下文缺失 | 在 C2f 结构中嵌入轻量自注意力 |
| **DA-AIFI** | 编码器 | 密集场景噪声干扰、全局注意力计算冗余 | 可变形注意力替代全局注意力 |
| **FSFPN + CFSB** | 特征金字塔 | 高频纹理在跨尺度融合中丢失 | 频域滤波 + 空间边缘提取的并行融合 |

其中 **FSFPN + CFSB** 是最大亮点——它贡献了消融实验中最高的单模块增益（+1.9 AP50）。

### 2.2 整体数据流

![Fig. 2: Overall Architecture](https://raw.githubusercontent.com/Liwx1014/PicBed/main/images/4d8c2cdf7bf4e1c78a7ef1f5b10347a3b86600e98c762084229fe277c4d17b15.jpg)

*▲ 图2：FSDETR 整体架构。骨干网络集成 SHAB 和 C2f 模块，高效混合编码器使用 DA-AIFI 进行帧内交互、FSFPN 进行跨尺度融合。*

```
输入图像 → 骨干(CSPNet + C2f + SHAB) → 高效混合编码器(DA-AIFI + FSFPN) → Decoder → 预测
```

骨干产生多尺度特征 \(\{P_2, P_3, P_4, P_5\}\)，SHAB 插入 P4/P5 深层增强长程建模。编码器中，DA-AIFI 先对高层特征做自适应稀疏交互，FSFPN 再通过 CFSB 融合频域和空间信息。最后送入标准 query selection 和 Transformer 解码器。

---

## 三、模块详解：从源码看设计哲学

### 3.1 SHAB：只对一半通道做注意力

![Fig. 3: SHAB Structure](https://raw.githubusercontent.com/Liwx1014/PicBed/main/images/bdefa5fba41b58a55211a4023c1a07f4ec753c6689df590bec6ceba598e1d8a6.jpg)

*▲ 图3：SHAB 结构。继承 C2f 的跨阶段部分连接拓扑，用 SHSA Block 替换标准 bottleneck。*

SHAB 继承自 C2f 的骨架，但把内部的 Bottleneck 替换成了 **SHSABlock**：

```python
# modules/custom_block.py
class SHAB(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(SHSABlock(self.c) for _ in range(n))
```

SHSABlock 由三个子模块级联而成——每个都带残差连接：

```python
class SHSABlock(torch.nn.Module):
    def __init__(self, dim, qk_dim=16, pdim=64):
        super().__init__()
        self.conv = Residual(Conv2d_BN(dim, dim, 3, 1, 1, groups=dim))  # ① DW Conv
        self.mixer = Residual(SHSA(dim, qk_dim, pdim))                   # ② 单头自注意力
        self.ffn = Residual(SHSABlock_FFN(dim, int(dim * 2)))            # ③ FFN

    def forward(self, x):
        return self.ffn(self.mixer(self.conv(x)))
```

> 输入特征先过一层**逐通道 3×3 卷积**增强局部空间信息，然后过一层**轻量自注意力**捕获全局上下文，最后过一个**两层 MLP**（expansion ratio=2）做非线性变换。每个步骤都有残差连接。

**"砍半"策略——SHAB 省参数的核心技巧**：

```python
class SHSA(torch.nn.Module):
    def forward(self, x):
        B, C, H, W = x.shape
        x1, x2 = torch.split(x, [self.pdim, self.dim - self.pdim], dim=1)  # ← 砍半！
        x1 = self.pre_norm(x1)       # 只有 x1 做 GroupNorm
        qkv = self.qkv(x1)           # 只有 x1 过 QKV 投影
        q, k, v = qkv.split(...)
        attn = (q.T @ k) * self.scale
        attn = attn.softmax(dim=-1)
        x1 = (v @ attn.T).reshape(B, self.pdim, H, W)  # x1 做注意力
        x = self.proj(torch.cat([x1, x2], dim=1))       # x2 原封不动 concat 回来
        return x
```

- `pdim` 默认 64，`dim` 默认 128（在骨干深层），所以**确实只有一半通道在做注意力**
- 另一半通道（x2）走恒等映射，保留原始空间纹理信息
- 最后用 1×1 卷积把两部分信息重新融合

**为什么这样设计？** 传统自注意力是 \(O(C^2 \cdot HW)\) 的，砍半后复杂度降到 \(O((C/2)^2 \cdot HW) = O(C^2/4 \cdot HW)\)，**省 75% 计算**。同时，被跳过的半通道恰好保留了 CNN 固有的局部归纳偏置，让 SHAB 既有 Transformer 的全局视野，又不丢失 CNN 的细节感觉。

### 3.2 DA-AIFI：把注意力花在刀刃上

标准的 AIFI（帧内特征交互）使用全局多头自注意力——但对小目标来说，**特征图上 95% 的位置都是背景**，全局注意力等于把海量计算浪费在无关区域。

DA-AIFI 的思路很直接：**不要看所有位置，只看到最值得看的那几个位置**。源码把自注意力换成了可变形注意力（`DAttention`，来自 CVPR 2022 的 ViT with Deformable Attention）：

```python
# modules/custom_transformer.py
class DA_AIFI(nn.Module):
    def __init__(self, c1, cm=2048, num_heads=8, ...):
        super().__init__()
        self.DAttention = DAttention(channel=c1, q_size=(20, 20))  # 取代全局自注意力
        self.fc1 = nn.Conv2d(c1, cm, 1)  # FFN 第一层（expansion=8x）
        self.fc2 = nn.Conv2d(cm, c1, 1)  # FFN 第二层
        self.norm1 = LayerNorm(c1)
        self.norm2 = LayerNorm(c1)

    def forward_post(self, src, ...):
        src2 = self.DAttention(src)       # 可变形注意力
        src = src + self.dropout1(src2)    # 残差
        src = self.norm1(src)
        src2 = self.fc2(self.act(self.fc1(src)))  # FFN
        src = src + self.dropout2(src2)
        return self.norm2(src)
```

整体结构就是标准的 Transformer Encoder Layer，**唯一变化是把 Attention 从"全局"换成了"可变形"**。

**DAttention 是怎么"只看到想看的位置"的？**

```python
# modules/attention.py
class DAttention(nn.Module):
    def forward(self, x):
        # 1. 用卷积网络从 query 特征中预测每个位置的采样偏移量
        offset = self.conv_offset(q_off)       # shape: (B*g, 2, Hg, Wg)
        offset = offset.tanh().mul(offset_range).mul(self.offset_range_factor)

        # 2. 参考点 + 偏移量 = 实际采样位置
        reference = self._get_ref_points(Hk, Wk, ...)  # 均匀网格参考点
        pos = offset + reference                        # 变形后的采样位置

        # 3. 在变形后的位置上做 grid_sample 获取 K/V
        x_sampled = F.grid_sample(x, grid=pos[...])  # 双线性插值采样

        # 4. 标准 QKV 注意力，但 K/V 只来自采样点
        q = self.proj_q(x)           # Q: 所有位置
        k = self.proj_k(x_sampled)   # K: 只来自 K 个采样点
        v = self.proj_v(x_sampled)   # V: 只来自 K 个采样点
        attn = einsum('b c m, b c n -> b m n', q, k) * scale
        out = einsum('b m n, b c n -> b c m', attn.softmax(-1), v)
```

> 对特征图上的每一个位置，模型自己学习"我应该往哪个方向看、看多远"。在 20×20 的特征图上，原本要计算 400×400=160,000 对位置关系，现在每个位置只看 \(K\) 个采样点（默认约 400），复杂度从 \(O(N^2)\) 降至 \(O(NK)\)。\(N^2=160,000\) vs \(NK=8,000\)，**省了 95% 的计算**。

而且这些采样点不是均匀分布的——通过可学习的偏移量，模型会自动把采样点"拽"到有目标的位置，**自适应聚焦于信息密集区域**。这就是为什么计算量大幅下降的同时，精度反而上升了。

### 3.3 FSFPN + CFSB：频域与空间的化学反应

传统 FPN 有两大致命伤：（1）自顶向下路径中的上采样会**模糊高频纹理**；（2）步长卷积**直接丢弃 75% 的像素**。对小目标来说，这两刀下去基本就不剩什么了。

FSFPN 从"信号恢复与增强"的视角重新设计了 FPN，将整个颈部分为**预处理层 + CFSB 融合层**。

![Fig. 4: CFSB Structure](https://raw.githubusercontent.com/Liwx1014/PicBed/main/images/081609685a3c67b308406cfdd9c1a71156d996c2d6115a74cb8fde5a5f436105.jpg)

*▲ 图4：CFSB 结构。空间分支用 Scharr 算子提取边缘，频域分支用 DFT + 可学习滤波掩码增强高频信号，最后逐元素融合。*

#### 3.3.1 预处理层：SPDConv 和 SNI

**SPDConv（Space-to-Depth Convolution）** 的核心操作就一行：

```python
class SPDConv(nn.Module):
    def forward(self, x):
        # 把 2×2 窗口的 4 个子像素拆开，沿通道维拼接：C→4C, H→H/2, W→W/2
        x = torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2],
                       x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1)
        x = self.conv(x)  # 4C → ouc
        return x
```

> 传统 stride=2 的 3×3 卷积，每 2×2 窗口只保留一个值（其余 3 个直接丢弃）。SPDConv 把 2×2 窗口的 4 个值全部保留，从"空间上的 2×2"变成"通道上的 4"。**没有丢弃任何像素**，空间信息无损地转移到了通道维度。

**SNI（软最近邻插值）** 比标准上采样更"软"：

```python
class SNI(nn.Module):
    def __init__(self, up_f=2):
        self.us = nn.Upsample(None, up_f, 'nearest')  # 最近邻上采样 2x
        self.alpha = 1 / (up_f ** 2)                     # alpha = 0.25

    def forward(self, x):
        return self.alpha * self.us(x)  # 缩放后输出
```

> 普通最近邻上采样把一个像素复制成 4 个（2×2 窗口），参与后续卷积时权重被放大 4 倍。SNI 上采样后乘以 0.25，把膨胀的激活值压回正常范围，避免高频纹理被过度抑制。

#### 3.3.2 CFSB：双域融合核心

CFSB 继承自 C2f 的骨架，但把内部 Bottleneck 替换成 `FreqSpatial` 模块：

```python
class CFSB(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(FreqSpatial(self.c) for _ in range(n))
```

`FreqSpatial` 是双域融合的核心，包含两个并行分支：

```python
class FreqSpatial(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.sed = ScharrConv(in_channels)       # 空间：Scharr 梯度算子
        self.spatial_conv1 = Conv(in_channels, in_channels)
        self.spatial_conv2 = Conv(in_channels, in_channels)

        self.fft_conv = Conv(in_channels * 2, in_channels * 2, 3)  # 频域：卷积滤波
        self.fft_conv2 = Conv(in_channels, in_channels, 3)
        self.final_conv = Conv(in_channels, in_channels, 1)

    def forward(self, x):
        # ── 空间分支 ──
        spatial_feat = self.sed(x)               # Scharr 边缘提取
        spatial_feat = self.spatial_conv1(spatial_feat)
        spatial_feat = self.spatial_conv2(spatial_feat + x)  # 残差连接

        # ── 频域分支 ──
        fft_feat = torch.fft.rfft2(x, norm='ortho')          # ① 实数 FFT
        x_fft_real = torch.real(fft_feat).unsqueeze(-1)      # ② 拆成实部/虚部
        x_fft_imag = torch.imag(fft_feat).unsqueeze(-1)
        fft_feat = torch.cat((x_fft_real, x_fft_imag), dim=-1)
        fft_feat = rearrange(fft_feat, 'b c h w d -> b (c d) h w')  # ③ 合并为通道

        fft_feat = self.fft_conv(fft_feat)           # ④ 在频域做卷积（可学习滤波）

        fft_feat = rearrange(fft_feat, 'b (c d) h w -> b c h w d', d=2)  # ⑤ 拆回实部/虚部
        fft_feat = torch.view_as_complex(fft_feat)                        # ⑥ 还原复数
        fft_feat = torch.fft.irfft2(fft_feat, s=(h, w), norm='ortho')    # ⑦ 逆 FFT

        fft_feat = self.fft_conv2(fft_feat)

        # ── 融合 ──
        out = spatial_feat + fft_feat     # 逐元素相加
        return self.final_conv(out)       # 跨通道交互
```

> **空间那路**：Scharr 算子扫一遍特征图 → 提取梯度响应 → 两个卷积 + 残差 → 输出"边界在哪"  
> **频域那路**：FFT 变成"频率分量" → 频谱上做可学习卷积（高频增强、低频抑制）→ 逆 FFT 回空间域 → 输出"纹理长什么样"  
> **融合**：两路输出逐元素相加 → 1×1 卷积做跨通道交互

#### 3.3.3 频域分支深度拆解

**为什么需要频域？**

想象你在听一首歌。想知道里面有没有小提琴的声音，你**不会去看波形图上的每个采样点**——密密麻麻的线条毫无意义。你会去看**频谱图**：低频是鼓和贝斯，中频是人声，高频是小提琴的泛音。

图像也是同样的道理。一张 640×640 的航拍图中，一个 8×8 的小人只占 64 个像素。空间域里，CNN 的 3×3 卷积一次只能看到 9 个像素——如果这 9 个像素恰好全是背景（概率极高），它就完全错过了目标。但转换到频域后：

- **大面积的天空、路面 → 低频分量**（颜色变化缓慢）
- **小目标的边缘、纹理 → 高频分量**（像素值瞬间跳变——从路面灰到小人白）
- **杂乱背景中的噪声 → 分散在各频率**

关键洞察：**在空间域，小目标被"位置"困住了；在频域，小目标的高频信号是全局可见的——整个频谱图上的高频区域都在"替它说话"。**

---

**七步逐行拆解频域分支：**

**第①步：实数 FFT — 把像素变成频率**

```python
fft_feat = torch.fft.rfft2(x, norm='ortho')  # [B, C, H, W] → [B, C, H, W//2+1]（复数）
```

傅里叶变换的核心定理：**任何一张图像都可以分解成无数个不同频率、不同方向的正弦波的叠加**。

- 纯蓝色的天空 → FFT 后只有低频分量（颜色几乎不变 → 频率≈0）
- 白色小点在灰色路面上 → FFT 后高频分量遍布全图（灰到白是瞬间跳变 → 需要无穷多高频正弦波才能拟合）

`rfft2` 中的 `r`（real）表示"输入是实数"。实数的 FFT 结果是对称的（共轭对称性），只需保留一半频率分量，另一半可推出来——**省一半计算**。

输出是**复数**（实部 + 虚部 × i），实部和虚部共同编码每个频率分量的**幅度**（有多强）和**相位**（在哪个位置）。

**第②③步：拆开实部虚部，喂给卷积**

```python
x_fft_real = torch.real(fft_feat).unsqueeze(-1)   # [B,C,H,W//2+1] → [B,C,H,W//2+1,1]
x_fft_imag = torch.imag(fft_feat).unsqueeze(-1)   # 同上
fft_feat = torch.cat((x_fft_real, x_fft_imag), dim=-1)   # [B,C,H,W//2+1,2]
fft_feat = rearrange(fft_feat, 'b c h w d -> b (c d) h w')  # [B,2C,H,W//2+1]
```

**因为 PyTorch 的卷积只能处理实数，不能直接处理复数。** 做法是把复数的实部和虚部拆开，当成两个独立通道叠在一起（`C → 2C`），就能喂给普通卷积了。

**第④步：频域卷积 — 整个模块的灵魂**

```python
fft_feat = self.fft_conv(fft_feat)  # Conv2d(2C, 2C, kernel=3)
```

`fft_conv` 只是普通的 3×3 卷积，但输入是**频谱图**。这层卷积在做的，本质上是：

> **在频率空间里，对每个位置的每个频率分量，学习该"放大"还是"抑制"。**

打个比方：某个卷积权重学到了"高频区域 × 2.0"，另一个学到了"低频区域 × 0.1"。训练中，梯度告诉它们：**"你把对应小目标边缘的高频分量放大后，mAP 涨了 0.5 个点——继续保持！"**

**为什么频域 3×3 卷积的感受野比空间域大得多？** 在空间域，3×3 卷积每个位置只能看 9 个像素邻居。但在频域：

- 频域中的每个"像素"是一个**频率分量**，对应该频率在整个空间上的统计模式
- 频域中的 3×3 卷积在相邻频率间交互——相邻频率往往对应相似的空间纹理
- 每个频率分量本身已经是"全图统计"了，所以频域卷积天然具有**准全局感受野**

计算复杂度只有 \(O(C)\)，远小于自注意力的 \(O(N^2)\)。

**第⑤⑥⑦步：拼回复数，逆 FFT 回到空间域**

```python
fft_feat = rearrange(fft_feat, 'b (c d) h w -> b c h w d', d=2)  # 拆回 [B,C,H,W//2+1,2]
fft_feat = torch.view_as_complex(fft_feat)                        # 实部+虚部 → 复数
fft_feat = torch.fft.irfft2(fft_feat, s=(h, w), norm='ortho')    # 逆 FFT → [B,C,H,W]
```

这三步是第②③步的逆操作：滤波后的实部/虚部拼回复数 → 逆 FFT → 回到空间域。`irfft2` 的 `s=(h, w)` 确保输出尺寸一致。此时特征图已"脱胎换骨"——高频纹理被调亮，低频背景噪声被压暗。最后再过 `fft_conv2`（普通空间卷积）做微调。

**一句话总结：频域分支本质上是一个"可学习的智能锐化滤镜"——它不是手动调的，而是通过反向传播自己学会该增强哪些频率、抑制哪些频率。**

---

**空间+频域：为什么 1+1 > 2？**

两个分支是**互补**的：

| 维度 | Scharr 空间分支 | 频域分支 |
|:---|:---|:---|
| **告诉你什么** | "边缘在**哪里**"（像素级位置） | "边缘**有多锐**"（全局频率统计） |
| **擅长的** | 空间定位——坐标级别的精度 | 模式识别——"目标纹理 vs 背景噪声" |
| **局限** | 不知道梯度在全局尺度上是否重要 | 丢弃了部分精确位置信息（FFT 的固有属性） |
| **计算代价** | 固定 3×3 卷积 | \(O(C \log N)\)（FFT） |

单独用频域分支：知道"这图里有很多高频纹理"，但不知道具体在哪。单独用空间分支：知道每个像素的梯度方向，但不知道这些梯度在目标尺度和背景噪声尺度上哪个更重要。

**两者相加：空间分支定位，频域分支定调。互补得天衣无缝。**

#### 3.3.4 Scharr 算子细节

```python
class ScharrConv(nn.Module):
    def __init__(self, channel):
        super().__init__()
        # Scharr 3×3 核（比 Sobel 旋转对称性更好）
        scharr_kernel_x = [[3, 0, -3],   scharr_kernel_y = [[3, 10, 3],
                           [10, 0, -10],                     [0, 0, 0],
                           [3, 0, -3]]                      [-3, -10, -3]]
        # 固定权重，不参与训练
        self.scharr_kernel_x_conv.weight.data = scharr_kernel_x
        self.scharr_kernel_y_conv.weight.data = scharr_kernel_y
        self.scharr_kernel_x_conv.requires_grad = False  # 冻结！

    def forward(self, x):
        grad_x = self.scharr_kernel_x_conv(x)  # 水平梯度
        grad_y = self.scharr_kernel_y_conv(x)  # 垂直梯度
        return grad_x * 0.5 + grad_y * 0.5      # 等权融合
```

Scharr 算子的权重是**冻结的**（`requires_grad = False`），它只是一个固定的特征提取器——把"边缘在哪"的信息提炼出来，交给后续的可学习卷积去编码。这种 **"固定算子提取 + 可学习卷积编码"** 的设计，既保证了边缘检测的准确性（Scharr 有严格的数学定义），又保留了特征表达的灵活性。

### 3.4 损失函数：为什么选这三个？

\[\mathcal{L} = 2 \cdot \mathcal{L}_{VFL} + 5 \cdot \mathcal{L}_{L1} + 2 \cdot \mathcal{L}_{Focaler-EIoU}\]

三个损失的分工非常清晰：

- **Varifocal Loss**：RT-DETR 原生的分类损失。对高 IoU 的正样本给更高权重（IoU-aware weighting），对密集场景中低质量的"勉强算正样本"降权——密集场景里同一个 GT 周围可能有多个 anchor，VFL 自动筛选质量最高的。
- **Focaler-EIoU**：标准 IoU Loss 对低重叠样本的梯度很小（IoU 接近 0 时变化平坦），导致极小人难以收敛。Focaler-EIoU 通过**重构梯度函数**，让低重叠区间的梯度"变陡"——相当于给"差生"更多辅导。
- **L1 Loss**：直接回归 bbox 坐标，提供稳定的梯度信号。与 IoU Loss 互补——IoU 管"框的整体重叠度"，L1 管"每个坐标的精确偏移"。

---

## 四、实验结果

### 4.1 VisDrone 2019（无人机密集小目标）

| 模型 | 参数量 | AP50 | AP_S |
|:---|:---|:---|:---|
| RT-DETR-R18 (基线) | 20.0M | 36.3 | 11.3 |
| YOLOv11m | 20.0M | 35.0 | 9.8 |
| D-Fine-M | 19.2M | 40.7 | 13.0 |
| **FSDETR** | **14.7M** | **40.5** | **13.9** |

- 参数量比基线减少 **26.5%**（14.7M vs 20.0M），\(AP_S\) 提升 **+2.6 个点**
- 在所有对比模型中取得最高的 \(AP_S\)
- 全面超越同参数量级的 YOLO 系列（YOLOv8m/v10m/v11m/v12m）

### 4.2 TinyPerson（极小人检测，< 20×20 像素）

| 模型 | \(AP_{50}^{tiny}\) | \(AP_{50}^{tiny1}\) |
|:---|:---|:---|
| RT-DETR-R18 (基线) | 42.44 | 24.56 |
| D-Fine-M | 47.28 | 26.92 |
| RFLA（专门小目标方法） | 45.31 | 29.70 |
| **FSDETR** | **48.95** | **31.85** |

- \(AP_{50}^{tiny}\) 达到 48.95%，比第二名高出 1.67 个点
- 在 < 20×20 像素的极端尺度下，验证了频域-空间协同建模的有效性

### 4.3 消融实验（VisDrone 2019）

| 配置 | AP50 | AP_S | 
|:---|:---|:---|
| 基线 (RT-DETR-R18) | 36.3 | 11.3 |
| + SHAB | 37.5 (+1.2) | 11.9 (+0.6) |
| + DA-AIFI | 37.1 (+0.8) | 11.6 (+0.3) |
| + FSFPN (CFSB) | 38.2 (+1.9) | 12.5 (+1.2) |
| SHAB + DA-AIFI | 38.9 (+2.6) | 12.3 (+1.0) |
| **全部** | **40.5 (+4.2)** | **13.9 (+2.6)** |

关键发现：

- **FSFPN + CFSB 贡献了最大的单模块增益**：+1.9 AP50，+1.2 AP_S
- 三个模块的组合增益（+4.2）**大于各自增益之和**（1.2 + 0.8 + 1.9 = 3.9），说明三者间存在正向协同效应
- 频域信息与空间信息确实是互补的——CFSB 的频域分支捕获了空间分支无法感知的全局纹理模式

### 4.4 可视化分析

![Fig. 5: VisDrone Error Analysis](https://raw.githubusercontent.com/Liwx1014/PicBed/main/images/e2b855d961317181572196a43f7a80290a3b42ca4c4d05a18ab9f0178d603de7.jpg)

![Fig. 5: YOLO11m Comparison](https://raw.githubusercontent.com/Liwx1014/PicBed/main/images/dd2e2b278cece0291fe737e9d167e08522910c818664f34a8e7babd7dd71b91a.jpg)

![Fig. 5: FSDETR Result](https://raw.githubusercontent.com/Liwx1014/PicBed/main/images/bc03dc8b818e7c625fa611e0e8328412cf81ff3fa8d43ca90142f0b4271697f2.jpg)

*▲ 图5：VisDrone 误差分析。绿=TP，红=FP，蓝=FN。FSDETR 在高密度区域显著减少了蓝色漏检框。*

![Fig. 6: TinyPerson Baseline](https://raw.githubusercontent.com/Liwx1014/PicBed/main/images/90e8ea280619f8a01db385dc956f162d98a0eeeff95cf9cad8155b01ae0f68a9.jpg)

![Fig. 6: YOLO11m Result](https://raw.githubusercontent.com/Liwx1014/PicBed/main/images/59ba639e5c2bede7ec9d8c7957a9958c566aef04874c6669e8f5651d31fd2601.jpg)

![Fig. 6: FSDETR Result](https://raw.githubusercontent.com/Liwx1014/PicBed/main/images/9c9ee2cfefda9a432d930a04d0ab7ee5cf8dd288b499845908f11db3bd0453c5.jpg)

*▲ 图6：TinyPerson 误差分析。FSDETR 在低对比度海边场景中误检（红色框）明显更少。*

可视化直观展示了 FSDETR 的两个优势：

- **漏检更少**：高密度区域中蓝色 FN 框显著减少，归功于 SHAB 的长程依赖建模和 FSFPN 的高频信号保留
- **误检更少**：低对比度场景中红色 FP 框更少，频域分支的自适应滤波有效抑制了背景噪声

---

## 五、启示与展望

### 启示一：频域是空间特征的"免费午餐"

CFSB 的实验结果（+1.2 AP_S）证明了：在 FPN 中引入**可学习的频域滤波**（2D DFT + 卷积掩码 + IDFT），能以极小的计算代价显著增强小目标的纹理表征。DFT 是 \(O(N \log N)\) 的，而同等感受野的自注意力是 \(O(N^2)\) 的。任何检测器的颈部网络（FPN / PANet / BiFPN）理论上都可以嵌入类似的频域增强模块。

### 启示二：注意力需要"按需分配"

DA-AIFI 把全局自注意力替换为可变形注意力，复杂度从 \(O(N^2)\) 降到 \(O(NK)\)，精度反而提升。对于小目标检测，**稀疏、自适应的特征交互比密集的全局注意力更有效**——因为小目标只占图像面积的极小比例，绝大多数计算应该聚焦在目标区域而非背景。

### 启示三：轻量化 ≠ 牺牲性能

FSDETR 比 RT-DETR-R18 少 26.5% 参数，却在两个小目标基准上全面超越。秘诀在于每个模块都做了"减法"：

| 模块 | 减法策略 |
|:---|:---|
| SHAB | 只对一半通道做注意力，另一半恒等映射 |
| DA-AIFI | K 个稀疏采样点替代全局注意力 |
| CFSB | 轻量 DFT + 单层可学习掩码，而非重型的频域自注意力 |

这为边缘端/移动端的小目标检测部署提供了有价值的参考范式。

### 启示四：频域-空间协同是通用增强策略

虽然 FSDETR 基于 RT-DETR 构建，但 SHAB / DA-AIFI / CFSB 三个模块都是**即插即用**的。理论上任何检测器（YOLO、D-FINE、RF-DETR 等）都可以引入类似的频域-空间协同模块来增强小目标性能。未来值得探索的方向：

- 将 CFSB 嵌入 YOLO 的 PANet 颈部
- 用频域增强替代或补充 Transformer 中的位置编码
- 在开放词汇检测器中引入频域分支改善细粒度类别的判别能力

---

*📌 论文链接：[FSDETR (arXiv:2604.14884)](https://arxiv.org/abs/2604.14884) | 代码：[github.com/YT3DVision/FSDETR](https://github.com/YT3DVision/FSDETR)*

