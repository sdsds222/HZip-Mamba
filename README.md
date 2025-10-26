---

## Bidirectional Mamba Fusion Proposal: HZip-Mamba

This method aims to fuse the forward ($Y\_f$) and backward ($Y\_b$) outputs from a Bidirectional Mamba scan, leveraging local temporal context and sequence volatility (gradient information) to generate a high-quality resulting sequence ($Y$).

### Step 1: Global Bidirectional Scan (O(N) Parallel)

* **Action:** Simultaneously run two global Mamba Scanners (one forward, one backward) in parallel.
* **Output:** Two complete, unmodified "unidirectional suggested" sequences:
    * Y\_f (Forward output sequence from t=1 to N, shape L x D)
    * Y\_b (Backward output sequence from t=N to 1, shape L x D)

### Step 2: Context Widening and Local Context Generation (Focus on the 2 x k Convolution Concept)

* **Action:**
    1.  Concatenate $Y\_f$ and $Y\_b$ along the feature dimension (D) to obtain an enhanced feature sequence $Y\_in$ (shape L x (2D)).
    2.  Apply a one-dimensional convolution (Conv1D) to $Y\_in$, setting the kernel\_size = k (e.g., k=3 or 5), with an input channel count of 2D.
* **Core Understanding – 2 x k Convolution Kernel Operation:**
    * Although implemented as a 1D convolution in the framework, its width k slides across the temporal steps (e.g., t-1, t, t+1).
    * Since the input channels are the concatenated features of $Y\_f$ and $Y\_b$ (2D channels), this 1D convolution kernel **logically operates simultaneously on the feature space of both $Y\_f$ and $Y\_b$**.
    * The kernel's shape can be interpreted as extracting and fusing information simultaneously across the two directional dimensions ($Y\_f$ and $Y\_b$) and across k local temporal steps, forming an **effective 2 x k local information extractor**.
* **Output:** A new "wide-field summary" sequence, Y\_local\_context.
* **Purpose:** Y\_local\_context[t] fuses the interactive information between $Y\_f$ and $Y\_b$ at time t and its neighbors (t±(k-1)/2).

### Step 3: Calculating Intelligent Gating (Incorporating Gradient Information)

#### 3.1. Calculating Sequence Volatility (Gradient Magnitude)

* **Action:** Calculate the sequence local rate of change (gradient magnitude) for $Y\_f$ and $Y\_b$ at the current time step t.
* **Calculation Formulas:**
    * Forward Gradient G\_f[t]: Measures the instantaneous change in $Y\_f$.
        G\_f[t] = Abs( Y\_f[t] - Y\_f[t-1] )
    * Backward Gradient G\_b[t]: Measures the instantaneous change in $Y\_b$.
        G\_b[t] = Abs( Y\_b[t] - Y\_b[t+1] )
* **Output:** Sequence volatility feature vectors G\_f and G\_b.
* **Note:** The gradient calculation window can be set larger, such as t-5 to t+5. For noisy data, gradients can be calculated over a wider window and smoothed for noise reduction.

#### 3.2. Feature Fusion and Gating Input

* **Action:** Concatenate the wide-field summary $Y\_local\_context$ from Step 2 with the calculated sequence volatility $G\_f$ and $G\_b$ to form the input for the gating network.
    Gating\_Input = Concat(Y\_local\_context, G\_f, G\_b)

#### 3.3. Calculating Gating Vectors

* **Action:** Feed the Gating\_Input into two independent Gating Networks (MLP).
* **Output:** Two dimension-wise gating vectors $g\_f$ and $g\_b$.
    g\_f = Sigmoid(MLP\_f(Gating\_Input))
    g\_b = Sigmoid(MLP\_g(Gating\_Input))

### Step 4: Dimension-wise Fusion

* **Action:** Use $g\_f$ and $g\_b$ to perform element-wise weighted fusion on the original $Y\_f$ and $Y\_b$.
* **Final Output (Y\_t):**
    $$Y_t = (g_f[t] * Y_f[t]) + (g_b[t] * Y_b[t])$$

### Extensions:

1.  **Convolutional Kernel Blocking (Chunking):** The convolutional kernel can be used to merge sequences with similar properties into a block. This allows the structural parameters output by the convolution to regulate the $g\_f$ and $g\_b$ gates more precisely. (The data chunking can reuse the block map from Flexible Block Mamba).
2.  **Dynamic Adaptive Convolutional Kernel:** Use a Receptive Field Controller (RFC) to predict the kernel size and configuration based on the input and the bidirectional outputs. This further enhances the model's flexibility, enabling it to adapt the context window size based on the task or data characteristics.
---


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
