# DnCNN 概述

DnCNN（去噪卷积神经网络 {cite:p}`DnCNN`）采用残差学习策略，隐式去除隐藏层中潜在的干净图像。这个特性促使训练 DnCNN 模型可以处理几种通用的图像去噪任务，如高斯去噪、单幅图像超分辨率（super-resolution）和 JPEG 图像去噪（JPEG image deblocking）。

图像去噪是底层视觉领域经典而活跃的课题，在许多实际应用中都是不可或缺的一步。

```{admonition} 目标
:class: tip

从有噪声的观测 $\mathbf{y}$ 中恢复干净的图像 $\mathbf{x}$，它遵循图像退化模型 $\mathbf{y} = \mathbf{x} + \mathbf{v}$。其中 $\mathbf{x}$、$\mathbf{y}$ 和 $\mathbf{v}$ 分别代表给定的干净图像、噪声图像和标准差 $\sigma$ 的加性高斯噪声（additive Gaussian noise，简称 AWGN）。
```

从贝叶斯的观点来看，当 似然（likelihood） 已知时，图像先验建模将在图像去噪中发挥核心作用。

DnCNN 不学习带有显式图像先验的判别模型，而是将图像去噪视为简单的判别学习问题，即通过前馈 CNN 将噪声从噪声图像中分离出来。

DnCNN 不是直接输出去噪后的图像 $\hat{\mathbf{x}}$，而是用来预测残差图像 $\hat{\mathbf{v}}$，即噪声观测图像和潜在的干净图像之间的差值。也就是说，DnCNN 通过隐藏层中的运算隐式去除了潜在的干净图像。进一步引入 BN 技术来稳定和提高 DnCNN 的训练性能。结果表明，残差学习和 BN 可以相互促进，两者的融合可以有效地加快训练速度，提高去噪性能。

```{note}
DnCNN 的目的是设计更有效的高斯去噪器，观察到，当 $\mathbf{v}$ 为高分辨率图像的 ground truth 与低分辨率图像的双三次上采样的差值时，高斯去噪的图像退化模型可以转化为单幅图像的超分辨率（single image
super-resolution，简称 SISR）问题；类似地，JPEG 图像去块问题可以用相同的图像退化模型来建模，取 $\mathbf{v}$ 为原始图像与压缩图像的差值。从这个意义上讲，SISR 和 JPEG 图像去块可以被视为一般图像去噪问题的两种特殊情况，尽管在 SISR 和 JPEG 去块中，噪声与 AWGN 有很大不同。
```

```{admonition} 亮点
1. 提出了端到端可训练的深度 CNN 高斯去噪算法。与现有的基于深度神经网络的直接估计潜在干净图像的方法相比，该网络采用残差学习策略从噪声观测中去除潜在干净图像。

2. 经实验发现：残差学习和 BN，不仅可以加快训练速度，而且可以提高去噪性能。

3. DnCNN 可以很容易地扩展到处理一般的图像去噪任务。可以训练单一的 DnCNN 模型进行高斯盲去噪，并且比针对特定噪声水平训练的其他方法获得更好的性能。此外，有希望使用单个 DnCNN 模型解决三个一般的图像去噪任务，即盲高斯去噪，SISR 和 JPEG 去块。
```

与使用许多残差单元（即，identity shortcuts）的残差网络不同，DnCNN 使用单个残差单元来预测残差图像。

## 残差学习

DnCNN 尝试学习残差映射 $\mathcal{R}(\mathbf{y}) \to \mathbf{v}$，便有 $\mathbf{x} = \mathbf{y} - \mathcal{R}(\mathbf{y})$。形式上，是期望残差图像与噪声输入估计残差图像之间的平均均方误差

$$
\ell(\varTheta) = \frac{1}{2N} \sum_{i=1}^N || \mathcal{R}(\mathbf{y}_i; \varTheta) - (\mathbf{y}_i - \mathbf{x}_i) ||_F^2
$$ (DnCNN)

可以作为损失函数学习 DnCNN 中的可训练参数 $\varTheta$。这里 $\{(\mathbf{y}_i, \mathbf{x}_i)\}_{i=1}^N$ 表示 $N$ 个噪声-干净训练图像（patch）对。