# 张量分块

从低维到高维延申。

有一维列向量：

$$ 
\boldsymbol{x} = (x_1, \cdots, x_{w})^T \in \mathbb{R}^{w}
$$

折叠为二维矩阵：

$$
\boldsymbol{X} = (\boldsymbol{x}_1^T, \cdots, \boldsymbol{x}_h^T)^T \in \mathbb{R}^{h \times w}
$$

折叠为四维张量：

$$
\mathsf{X} = \begin{pmatrix}
\boldsymbol{X}_{11}^T & \cdots & \boldsymbol{X}_{1c}^T\\
\vdots & \ddots & \vdots \\
\boldsymbol{X}_{b1}^T & \cdots & \boldsymbol{X}_{bc}^T\\
\end{pmatrix} \in \mathbb{R}^{b \times c \times w \times h}
$$

延展为六维张量：

$$
\mathsf{Y} = \begin{pmatrix}
\mathsf{X}_{11}^T & \cdots & \mathsf{X}_{1c_0}^T\\
\vdots & \ddots & \vdots \\
\mathsf{X}_{b_01}^T & \cdots & \mathsf{X}_{b_0c_0}^T\\
\end{pmatrix} \in \mathbb{R}^{b_0 \times c_0 \times h \times w \times c \times b}
$$
