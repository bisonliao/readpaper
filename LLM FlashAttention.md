# FlashAttention 

## 1. 要解决的核心问题

Transformer 模型在处理长序列时面临两大瓶颈：

- **计算与内存复杂度高**：自注意力机制的时间和内存复杂度均为序列长度 N 的二次方（O(N^2)）。当序列变长时，计算和内存开销急剧增长。

- **被忽视的 IO 瓶颈**：现有近似注意力方法主要关注减少 FLOPs（浮点运算次数），但忽略了内存访问开销（IO）。在现代 GPU 上，计算速度已远超内存带宽（如 A100 的片上 SRAM 带宽约 19TB/s，而高带宽内存 HBM 仅 1.5TB/s），导致大多数操作受内存访问限制而非计算限制。

- **标准实现的缺陷**：PyTorch/TensorFlow 等框架的标准注意力实现会将巨大的 N x N 注意力矩阵反复写入/读出相对慢速的 GPU HBM，造成严重 IO 瓶颈。

论文核心观点：注意力算法需要是 **IO 感知的（IO-aware）** ——即需精细控制不同层级内存（快速片上 SRAM 与慢速 HBM）之间的数据读写。

## 2. 解决方案：FlashAttention 算法

### 2.1 三项核心技术

FlashAttention 通过以下技术实现精确注意力的高效计算：

- **分块（Tiling）**  
  将 Q, K, V 矩阵分成小块，仅将当前计算所需块加载到快速片上 SRAM；利用 softmax 的可分解性（通过维护归一化统计量 m, l）增量计算注意力输出，避免将完整的 N x N 注意力矩阵写入 HBM。

- **重计算（Recomputation）**  
  前向传播中不存储 N x N 注意力矩阵，仅保存输出 O 和 softmax 统计量；反向传播时**在 SRAM 内重计算所需注意力值**。虽然增加了 FLOPs，但因避免了从 HBM 读取大矩阵，整体速度更快。

- **核融合（Kernel Fusion）**  
  将矩阵乘、softmax、masking、dropout 等操作融合为单一 CUDA kernel，输入仅从 HBM 读取一次，输出仅写回一次，大幅减少数据移动。

### 2.2 理论分析

- **IO 复杂度**：证明 FlashAttention 的 HBM 访问次数为 Theta(N^2 * d^2 / M)，其中 d 为注意力头维度，M 为 SRAM 大小；而标准注意力为 Theta(N*d + N^2)。

- **实际优势**：当 d=64、M 约 100KB 时，d^2 远小于 M，FlashAttention 的 HBM 访问次数可减少高达 9 倍。

- **最优性**：证明在所有 M 属于 [d, N*d] 范围内，不存在能渐近改进 FlashAttention HBM 访问次数的精确注意力算法。

### 2.3 扩展：Block-Sparse FlashAttention

- 将 FlashAttention 与块稀疏模式结合，仅计算非零块。
- **IO 复杂度**：Theta(N*d + N^2 * d^2 * s / M)，其中 s 为非零块比例。
- 适用于超长序列（如 64K），在保持近似质量的同时实现近线性扩展。

## 3. 实验效果

### 3.1 模型训练加速

| 模型/任务        | 序列长度 | 加速效果 | 对比基线                     |
| ---------------- | -------- | -------- | ---------------------------- |
| BERT-large       | 512      | 15% 更快 | 超越 MLPerf 1.1 训练速度记录 |
| GPT-2 (small)    | 1K       | 3.5 倍   | vs HuggingFace               |
| GPT-2 (medium)   | 1K       | 3.0 倍   | vs HuggingFace               |
| Long-Range Arena | 1K-4K    | 2.4 倍   | vs 标准注意力                |

### 3.2 模型质量提升（得益于更长上下文）

- **语言建模**：GPT-2 使用 4K 上下文（比标准 1K 长 4 倍）仍比 Megatron-LM 快 30%，困惑度提升 0.7。

- **长文档分类**：
  - MIMIC-III：16K 序列长度比 512 提升 4.3 个 F1 分数点
  - ECtHR：8K 序列长度比 512 提升 8.5 个 F1 分数点

- **突破性能力**：
  - 首次在 Path-X（序列长度 16K）上实现超越随机性能（61.4% 准确率）
  - Block-sparse FlashAttention 首次在 Path-256（序列长度 64K）上实现超越随机性能（63.1% 准确率）

### 3.3 注意力计算基准测试

- **运行时加速**：序列长度 ≤512 时，比 PyTorch 实现快 3 倍；GPT-2 上注意力计算快 7.6 倍。

- **内存效率**：内存占用线性于序列长度；序列长度 512 时比标准注意力节省 20 倍内存；支持序列长度达 64K（其他方法大多在 8K–16K 时内存溢出）。

- **关键洞察**：HBM 访问次数是运行时的主要决定因素。例如 GPT-2 上，FlashAttention 因重计算增加了 FLOPs（75.2 GFLOPs vs 标准 66.6），但 HBM 访问大幅减少（4.4GB vs 40.3GB），实际运行更快（7.3ms vs 41.7ms）。

### 3.4 与近似方法对比

- 序列长度 ≤512 时，FlashAttention 比所有近似/稀疏注意力方法更快。
- 序列长度 >1K 时，部分近似方法（如 Linformer）开始在速度上接近，但精度有损。
- Block-sparse FlashAttention 在所有测试序列长度上均快于现有近似方法，且精度损失极小。

## 4. 总结

FlashAttention 通过 **IO 感知设计**（分块 + 重计算 + 核融合），在**不牺牲模型精度**的前提下，同时实现了：

1. 更快的训练速度（最高 3.5 倍加速）
2. 更低的内存占用（线性于序列长度，最高节省 20 倍内存）
3. 更长的上下文建模能力（支持 64K 序列），带来模型质量提升与新能力

该工作揭示了深度学习优化中被忽视的 **IO 瓶颈**，为高效 Transformer 设计提供了新范式。代码已开源：https://github.com/HazyResearch/flash-attention



## 5 疑问

算法复杂度分析中的一个关键概念：**"HBM 访问次数" 指的是访问的元素总数（字节数），而非“读写操作的次数”**。这是理解 FlashAttention 核心贡献的关键。

### 为什么是 Ω(Nd + N²) 而非常数？

从 Algorithm 0 看，确实只有 3 个主要步骤（对应您说的“8 次读写”），但每一步涉及的**数据量**随序列长度变化：

```
Algorithm 0 Standard Attention Implementation
1: Load Q, K by blocks from HBM, compute S = QK^T, write S to HBM.
   → 读 Q (N×d) + 读 K (N×d) + 写 S (N×N) = 2Nd + N² 个元素

2: Read S from HBM, compute P = softmax(S), write P to HBM.
   → 读 S (N×N) + 写 P (N×N) = 2N² 个元素

3: Load P and V by blocks from HBM, compute O = PV, write O to HBM.
   → 读 P (N×N) + 读 V (N×d) + 写 O (N×d) = N² + 2Nd 个元素
```

**总计 HBM 访问元素数**：
```
(2Nd + N²) + 2N² + (N² + 2Nd) = 4Nd + 4N² = Θ(Nd + N²)
```

### 关键点澄清

| 概念             | 说明                                                         |
| ---------------- | ------------------------------------------------------------ |
| **操作次数**     | 固定（如 3 步计算），与 N 无关                               |
| **访问元素总数** | 随 N 增长：Q/K/V 各 N×d，S/P 各 N×N → 总计 Θ(Nd + N²)        |
| **主导项**       | 长序列场景下 N >> d，N² 项主导（如 N=4096, d=64 时，N²=16M, Nd=256K） |

### 实证验证（论文 Figure 2 左图）

GPT-2 (N=1024, d=64) 上的实际测量：
- **标准注意力**：HBM 读写总量 **40.3 GB**
- **FlashAttention**：HBM 读写总量 **4.4 GB**（减少约 9 倍）

这与理论分析一致：标准实现需反复搬运 N×N 的注意力矩阵（~1M 元素），而 FlashAttention 通过分块（tiling）将其保留在快速的片上 SRAM 中，仅搬运 Q/K/V/O（各 ~64K 元素）。

### 为什么这很重要？

现代 GPU 的瓶颈是 **内存带宽** 而非计算能力：
- A100 的 HBM 带宽：~1.5 TB/s
- A100 的片上 SRAM 带宽：~19 TB/s（快 10+ 倍）

即使 FlashAttention 因重计算增加了 FLOPs（75.2 GFLOPs vs 66.6），但因**大幅减少 HBM 访问**（4.4GB vs 40.3GB），实际运行更快（7.3ms vs 41.7ms）。

> 💡 简言之：Ω(Nd + N²) 描述的是**数据搬运量**的渐近复杂度，而非操作步骤数。FlashAttention 的核心创新正是通过 IO 感知设计，将二次复杂度的数据搬运降至近线性。