## Bidirectional Mamba Fusion Concept: HZip-Mamba

With this method, we can fuse the bidirectional Mamba scans (Y_f) and (Y_b), and use local temporal context and sequence rate-of-change to produce a high-quality (Y) result for each position.

### Step 1: Global Bidirectional Scanning (O(N) in parallel)

* Action: Run two global Mamba scanners in parallel (one forward, one backward).
* Output: Obtain two complete, unmodified “one-direction suggestions”:

  * (Y_f) (forward output sequence from (t=1) to (N), shape (L \times D))
  * (Y_b) (backward output sequence from (t=N) to (1), shape (L \times D))

### Step 2: Broaden the View and Generate Local Context (emphasising the 2 × k kernel concept)

* Action:

  1. Concatenate (Y_f) and (Y_b) along the feature dimension (D) to obtain an enhanced feature sequence (Y_in) (shape (L \times (2D))).

  2. Apply a 1D convolution (Conv1D) to (Y_in), set (\text{kernel_size} = k) (e.g., (k=3) or (5)), and the input channels to (2D).

  * Although implemented as a 1D convolution, its width (k) slides over timesteps (e.g., (t-1, t, t+1)).
  * Since the input channels are the concatenated features of (Y_f) and (Y_b) (2D channels), this 1D kernel logically acts on the feature spaces of both (Y_f) and (Y_b) simultaneously.
  * The kernel can be viewed as extracting and fusing information jointly along the two directional dimensions (Y_f) and (Y_b) and across the local (k) timesteps, forming an effective **2 × k** local information extractor.
* Output: Produce a new “wide-view summary” sequence, (Y_local_context).
* Role: (Y_local_context[t]) fuses the interactive information between (Y_f) and (Y_b) at time (t) and its neighbors ((t \pm (k-1)/2)).

### Step 3: Compute Intelligent Gating (incorporating gradient information)

#### 3.1. Compute sequence rate of change (gradient magnitude)

* Action: Compute the local rate-of-change (gradient magnitude) of (Y_f) and (Y_b) at the current timestep (t).
* Formulas:

  * Forward gradient (G_f[t]): measures the instantaneous change of (Y_f).
    (G_f[t] = \text{Abs}(Y_f[t] - Y_f[t-1]))
  * Backward gradient (G_b[t]): measures the instantaneous change of (Y_b).
    (G_b[t] = \text{Abs}(Y_b[t] - Y_b[t+1]))
* Output: Feature vectors of sequence rate-of-change (G_f) and (G_b).
* The gradient computation window can be enlarged, e.g., from (t-5) to (t+5).
* For unstable data, increase the gradient window and apply smoothing/denoising to the gradients.

#### 3.2. Feature fusion and gating input

* Action: Concatenate the (Y_local_context) from Step 2 with the computed (G_f) and (G_b) as the input to the gating networks.
  ( \text{Gating_Input} = \text{Concat}(Y_local_context, G_f, G_b) )

#### 3.3. Compute gating vectors

* Action: Feed (\text{Gating_Input}) into two independent gating networks (MLPs).
* Output: Two dimension-wise gating vectors (g_f) and (g_b).
  ( g_f = \text{Sigmoid}(\text{MLP_f}(\text{Gating_Input})) )
  ( g_b = \text{Sigmoid}(\text{MLP_g}(\text{Gating_Input})) )

### Step 4: Dimension-wise fusion

* Action: Use (g_f) and (g_b) to perform element-wise weighted fusion on the original (Y_f) and (Y_b).
* Final output ((Y_t)):
  $$Y_t = (g_f[t] * Y_f[t]) + (g_b[t] * Y_b[t])$$

### Extensions:

1. **Kernel-based blocking**: Use convolutional kernels to merge segments of similar properties into a block; then incorporate the structural parameters from the kernel outputs to more precisely control the gating (g_f) and (g_b). (The block data can reuse the block maps from Flexible Block Mamba.)
2. **Dynamic adaptive kernels**: Use a Receptive Field Controller (RFC) to predict kernel size and configuration based on the inputs and the bidirectional outputs. This further enhances model flexibility, allowing it to adaptively adjust the context window size according to the task or data characteristics.


## 双向mamba融合设想： HZip-Mamba

通过此方法，可融合双向 Mamba 扫描出的 Yf 和 Yb，并利用局部时序上下文和序列变化率为每个点位生成优质的 Y 结果。

### 步骤一：全局双向扫描 (O(N) 并行)

* 动作：并行运行两个全局的 Mamba 扫描器（一个正向，一个反向）。
* 输出：得到两个完整的、未经修改的“单向建议”序列：
    * Y\_f (从 t=1 到 N 的正向输出序列, 形状为 L x D)
    * Y\_b (从 t=N 到 1 的反向输出序列, 形状为 L x D)

### 步骤二：拓宽视野与局部上下文生成（重点增强 2 x k 卷积核概念）

* 动作：
    1. 将 Y\_f 和 Y\_b 沿着特征维度（D）拼接 (Concat) 起来，得到一个增强特征序列 Y\_in (形状为 L x (2D))。
    2. 我们对 Y\_in 使用一个一维卷积 (Conv1D)，设置 kernel\_size = k (例如 k=3 或 5)，以及 2D 的输入通道数。

    * 尽管在框架中是 1D 卷积，但其宽度 k 在时间步上滑动（例如 t-1, t, t+1）。
    * 由于输入通道是 Y\_f 和 Y\_b 的特征拼接 (2D 通道)，因此这个 1D 卷积核在逻辑上同时作用于 Y\_f 和 Y\_b 的特征空间。
    * 卷积核的形状可以被理解为同时在 Y\_f 和 Y\_b 这两个方向维度上，以及局部 k 个时间步上提取和融合信息，形成一个有效的 2 x k 局部信息提取器。
* 输出：得到一个新的“宽视野摘要”序列，Y\_local\_context。
* 作用：Y\_local\_context[t] 融合了 t 时刻及其邻居（t±(k-1)/2）上 Y\_f 和 Y\_b 的交互信息。

### 步骤三：计算智能门控 (融入梯度信息)

#### 3.1. 计算序列变化率（梯度大小）

* 动作：计算 Y\_f 和 Y\_b 在当前时间步 t 上的序列局部变化率（梯度大小）。
* 计算公式：
    * 向前梯度 G\_f[t]：衡量 Y\_f 的瞬时变化。
        G\_f[t] = Abs( Y\_f[t] - Y\_f[t-1] )
    * 向后梯度 G\_b[t]：衡量 Y\_b 的瞬时变化。
        G\_b[t] = Abs( Y\_b[t] - Y\_b[t+1] )
* 输出：序列变化率特征向量 G\_f 和 G\_b。
* 梯度计算范围可设定为更大的窗口，如t-5到t+5。
* 针对不稳定的数据，梯度可进行增大梯度计算窗口并平滑降噪处理。

#### 3.2. 特征融合与门控输入

* 动作：将步骤 2 的 Y\_local\_context 与计算出的序列变化率 G\_f 和 G\_b 拼接起来，作为门控网络的输入。
    Gating\_Input = Concat(Y\_local\_context, G\_f, G\_b)

#### 3.3. 计算门控向量

* 动作：将 Gating\_Input 喂给两个独立的门控网络 (MLP)。
* 输出：两个维度级的门控向量 g\_f 和 g\_b。
    g\_f = Sigmoid(MLP\_f(Gating\_Input))
    g\_b = Sigmoid(MLP\_g(Gating\_Input))

### 步骤四：维度级融合

* 动作：使用 g\_f 和 g\_b，对原始 Y\_f 和 Y\_b 进行逐元素的加权融合。
* 最终输出 (Y\_t)：
$$Y_t = (g_f[t] * Y_f[t]) + (g_b[t] * Y_b[t])$$
### 拓展：
1. 卷积核分块：可以用卷积核将性质相同的序列合并成一个块，这样加入卷积核输出的结构参数能够更精确地调控gf和gb的门控。（分块的数据可复用Flexible block mamba里的分块地图）
2. 动态自适应卷积核：通过感受野控制器 (RFC) 根据输入和双向输出预测卷积核的尺寸和配置。这种方法进一步提升了模型的灵活性，使其能够根据任务或数据的特性自适应地调整上下文窗口大小。
