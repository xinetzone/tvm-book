# 深度神经网络模型量化方法综述

参考：
- [深度神经网络模型量化方法综述](https://cje.ustb.edu.cn/cn/article/doi/10.13374/j.issn2095-9389.2022.12.27.004?viewType=HTML)

## 基本概念

模型量化技术可以通过减少深度神经网络模型参数的位宽和中间过程特征图的位宽，从而达到压缩加速深度神经网络的目的，使量化后的网络能够部署在资源有限的边缘设备上。

### 问题定义

神经网络模型的量化问题可以被定义为以下的优化问题：

对于神经网络模型 $\mathcal{G}$，假设其内部包含 $t$ 个需要被量化的参数层，所有参数层中的权值组成集合 $\mathcal{W} = \{\mathbf{w}_1, \cdots, \mathbf{w}_t\}$；同时，其内部包含 $m$ 个需要被量化的激活值层，所有激活值层中的值组成集合 $\mathcal{A} = \{\mathbf{a}_1, \cdots, \mathbf{a}_m\}$。给出所有权值和激活值对应的量化映射集合 $\mathcal{F} = \{q_1(\mathbf{w}_1; b_1), \cdots, q_{t+m}(\mathbf{a}_m; b_{t+m})\}$，其中 $q_i$ 表示第 $i$ 个参数层或激活值层的量化映射，$b$ 为量化映射后低精度数值的位宽。将此量化映射插入到神经网络模型 $\mathcal{G}$ 中的特定位置，形成量化网络模型 $\mathcal{Q}$。通过优化量化映射集合 $\mathcal{F}$，使得量化网络模型 $\mathcal{Q}$ 的损失函数 $L(\mathcal{Q})$ 最小，即：

$$
\min_{\mathcal{F,W}} |L(\mathcal{Q}) - L(\mathcal{G})|
$$

### 量化计算原理

以线性非对称量化为例，浮点数量化为有符号定点数的计算原理如下：

$$
\mathbf{x}_{\text{int}} = \operatorname{clip}(\lfloor \cfrac{\mathbf{x}}{s} \rceil + z; -2^{b-1}, 2^{b-1}-1)
$$

其中，$\mathbf{x}$ 为输入的浮点数值，$\mathbf{x}_{\text{int}}$ 为量化后的有符号定点数值，$s$ 为量化比例因子，$z$ 为量化零点，$b$ 为量化后的定点数的位宽，如 INT8 数据类型中 $b$ 为 8。$\operatorname{clip}$ 为截断函数，定义如下

$$
\operatorname{clip}(x; a, c) = \begin{cases}
a, & x < a \\
x, & a \leq x \leq c \\
c, & x > c
\end{cases}
$$

从定点数转换为浮点数称为反量化过程，具体定义如下：

$$
\mathbf{x} \approx \hat{\mathbf{x}} = s({\mathbf{x}_{\text{int}} - z})
$$

设置量化范围为 $(q_{\min},q_{\max})$，截断范围为 $(c_{\min},c_{\max})$，量化参数 $s$ 和 $z$ 的计算公式如下：

$$
\begin{aligned}
s &= \cfrac{q_{\max} - q_{\min}}{c_{\max} - c_{\min}} = \cfrac{q_{\max} - q_{\min}}{2^{b} - 1} \\
z &= c_{\max} - \lfloor \cfrac{q_{\max}}{s} \rceil \text{ 或 } z = c_{\min} - \lfloor \cfrac{q_{\min}}{s} \rceil 
\end{aligned}
$$

其中截断范围是根据量化的数据类型决定的，如 INT8 的截断范围为 $[-128, 127]$；量化范围根据不同的量化算法确定。

### 量化误差

量化误差来源于舍入误差和截断误差，即  $\lfloor \cdot \rceil$ 和 clip 运算。四舍五入的计算方式会产生舍入误差，误差范围为$[-\cfrac{s}{2}, \cfrac{s}{2}]$。当浮点数 $x$ 过大，比例因子 $s$ 过小时，容易导致量化定点数超出截断范围，产生截断误差。理论上，比例因子 $s$ 的增大可以减小截断误差，但会造成舍入误差的增大。因此为了权衡两种误差，需要设计合适的比例因子和零点，来减小量化误差。

### 线性对称量化和线性非对称量化

线性量化中定点数之间的间隔是均匀的，例如 INT8 线性量化将量化范围均匀等分为 256 个数。线性对称量化中零点是根据量化数据类型确定并且零点 $z$ 位于量化定点数范围上的中心对称点，例如 INT8 中零点为 0。线性非对称量化中零点 $z$ 一般不在量化定点数范围上的中心对称点。

对称量化是非对称量化的简化版本，理论上非对称量化能够更好的处理数据分布不均匀的情况，因此实践中大多采用非对称量化方案。

```{figure} images/quant.png
:align: center
:width: 70%

线性对称量化和线性非对称量化
```

### Per-Layer 量化和 Per-Channel 量化

- Per-Layer 量化将网络层的所有通道作为整体进行量化，所有通道共享相同的量化参数。
- Per-Channel 量化将网络层的各个通道独立进行量化，每个通道有自己的量化参数。Per-Channel 量化更好的保留各通道的信息，能够更好的适应不同通道之间的差异，提供更好的量化效果。

```{figure} images/per-quant.png
:align: center
:width: 70%

Per-Layer 量化和 Per-Channel 量化
```

```{admonition} 注意
:class: attention

Per-Channel 量化中只针对权重进行 Per-Channel 量化，激活值和中间值仍为 Per-Layer 量化。
```

### 量化算法

量化比例因子 $s$ 和零点 $z$ 是影响量化误差的关键参数，而量化范围的求解对量化参数起到决定性作用。本章节介绍三种关于量化范围求解的算法：Max，KL-Divergence 和 MSE, Percentile。

Max 量化算法是通过计算浮点数中的最大值和最小值直接确定量化范围的最大值和最小值。可知，Max 量化算法不会产生截断误差，但对异常值很敏感，因为大异常值可能会导致舍入误差过大。

$$
q_{\min} = \min \mathsf{V} \\
q_{\max} = \max \mathsf{V} \\
$$

其中 $\mathsf{V}$ 表示浮点数 Tensor。

KL-Divergence 量化算法计算浮点数和定点数的分布，通过调整不同的阈值来更新浮点数和定点数的分布，并根据 KL 散度最小化两个分布的相似性来确定量化范围的最大值和最小值。KL-Divergence 量化算法通过最小化浮点数和定点数之间的分布差异，能够更好地适应非均匀的数据分布并缓解少数异常值的影响。

$$
{\arg\min}_{q_{\min}, q_{\max}} H(\Psi(\mathsf{V}), \Psi(\mathbf{V}_{int}))
$$

其中$H(\cdot, \cdot)$ 表示 KL 散度。$\Psi(\cdot)$ 为分布函数，将对应数据计算为离散分布，$\mathbf{V}_{int}$ 为量化定点数 Tensor。

MSE 量化算法通过最小化浮点数与量化反量化后浮点数的均方误差损失，确定量化范围的最大值和最小值，在一定程度上缓解大异常值带来的量化精度丢失问题。由于MSE 量化算法的具体实现是采用暴力迭代搜索近似解，速度较慢，内存开销较大，但通常会比 Normal 量化算法具有更高的量化精度。

$$
{\arg\min}_{q_{\min}, q_{\max}} ||\mathsf{V} - \hat{\mathbf{V}}(q_{\min}, q_{\max})||_F^2
$$

其中 $\hat{\mathbf{V}}(q_{\min}, q_{\max})$ 为 $\mathbf{V}$ 的量化、反量化形式，$||\cdot||_F$ 为 F 范数。

Percentile 量化算法通过统计学方法计算浮点数的百分位数来确定量化范围，能有效排除异常值干扰。具体实现步骤如下：

1. 设定上下百分位阈值（如 p_low=1%, p_high=99%）
2. 计算量化范围边界值：
$$
q_{\min} = \text{percentile}(\mathsf{V}, p_{\text{low}}) \\
q_{\max} = \text{percentile}(\mathsf{V}, p_{\text{high}})
$$
3. 根据公式计算比例因子 $s$ 和零点 $z$

该方法的优势在于：
- 通过截断极端值提高量化鲁棒性
- 参数调节灵活（可通过调整百分位数平衡精度与异常值容忍度）
- 计算复杂度低于 MSE 算法

实际应用中建议：
- 初始值可设为 [1%, 99%] 的保守范围
- 对于重尾分布数据可适当扩大百分位范围（如 [0.5%, 99.5%]）
- 配合校准数据集进行阈值微调

```{note}
当 p_low=0% 且 p_high=100% 时，Percentile 算法退化为 Max 算法。与 MSE 算法相比，Percentile 在保持较好量化精度的同时，显著降低了计算开销，更适合部署在资源受限的边缘设备。
```