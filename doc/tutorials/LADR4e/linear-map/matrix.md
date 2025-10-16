# 矩阵

```{admonition} 定义：矩阵（matrix）、$\mathbf{A}_{j,k}$
假设 $m$ 和 $n$ 是非负整数．$m \times n$ 矩阵 $\mathbf{A}$ 是由 $\mathbb{F}$ 中元素构成的 $m$ 行 $n$ 列的矩形阵列：

$$
\mathbf{A} = \begin{bmatrix}
A_{1,1} & \cdots & A_{1,n} \\
\vdots &\ddots & \vdots \\
A_{m,1} & \cdots & A_{m,n}
\end{bmatrix}
$$

记号：
- $A_{j,k}$ 表示矩阵 $\mathbf{A}$ 的第 $j$ 行第 $k$ 列的元素．
- 当 $m = n$ 时，$\mathbf{A}$ 是一个 $n \times n$ 矩阵，称为**方阵**（square matrix）．
```

```{admonition} 定义：线性映射的矩阵（matrix of a linear map）、$\mathscr{M}(\mathcal{T})$
假设 $\mathcal{T}\in \mathscr{L}(\mathbb{V}, \mathbb{W})$，$v_1, \cdots, v_n$ 是 $\mathbb{V}$ 的基，$w_1, \cdots, w_m$ 是 $\mathbb{W}$ 的基，$\mathcal{T}$ 是从 $\mathbb{V}$ 关于这些基的矩阵是 $m\times n$ 矩阵 $\mathscr{M}(\mathcal{T})$，其中各元素 $A_{j,k}$ 由下式定义：

$$
\mathcal{T} v_k = A_{1,k} w_1 + \cdots + A_{m,k} w_m
$$

如果从上下文无法明确得知基 $v_1, \cdots, v_n$ 和 $w_1, \cdots, w_m$ 取什么，那么就用 $\mathscr{M}(\mathcal{T}, (v_1, \cdots, v_n), (w_1, \cdots, w_m))$ 这个记号。
```

为了记住 $\mathscr{M}(\mathcal{T})$ 是如何由 $\mathcal{T}$ 构造出来的，你可以在矩阵的上方横着标上定义空间的基向量 $v_1, \cdots, v_n$，并在矩阵的左侧竖着列出  $\mathcal{T}$ 映射到的向量空间的基向量 $w_1, \cdots, w_m$，就像下面这样：

$$
\mathscr{M}(\mathcal{T}) = \begin{matrix}
\begin{matrix}
&&\begin{matrix} v_1  \cdots  v_k \cdots  v_n \end{matrix} \end{matrix} \\
\begin{matrix}
\begin{matrix} w_1 \\ \vdots \\ w_m \end{matrix}
\begin{bmatrix}
&& \mathbf{A}_{1,k} && \\
&&\vdots &&  \\
&& \mathbf{A}_{m,k} 
\end{bmatrix}
\end{matrix}
\end{matrix}
$$

```{admonition} 定义：矩阵加法（matrix addition）
两个相同大小的矩阵之和，是将两矩阵对应位置上的元素相加所得的矩阵
```

```{admonition} 线性映射之和的矩阵
假设 $\mathcal{T}, \mathcal{U} \in \mathscr{L}(\mathbb{V}, \mathbb{W})$．那么 $\mathscr{M}(\mathcal{T} + \mathcal{U}) = \mathscr{M}(\mathcal{T}) + \mathscr{M}(\mathcal{U})$．
```

```{admonition}  定义：矩阵的标量乘法（scalar multiplication of a matrix）
一个标量和一个矩阵的乘积，是将该矩阵的各元素都乘以该标量所得的矩阵：

$$
\lambda \mathbf{A} = \begin{bmatrix}
\lambda A_{1,1} & \cdots & \lambda A_{1,n} \\
\vdots &\ddots & \vdots \\
\lambda A_{m,1} & \cdots & \lambda A_{m,n}
\end{bmatrix}
$$

```{admonition} 标量与线性映射之积的矩阵
假设 $\lambda \in \mathbb{F}$ 且 $\mathcal{T} \in \mathscr{L}(\mathbb{V}, \mathbb{W})$，那么 $\mathscr{M}(\lambda \mathcal{T}) = \lambda \mathscr{M}(\mathcal{T})$
```

```{admonition} 记号：$\mathbb{F}^{m,n}$
对于正整数 $m$ 和 $n$，各元素均属于 $\mathbb{F}$ 的所有 $m \times n$ 矩阵构成的集合记作 $\mathbb{F}^{m,n}$．
```

```{admonition} $\dim \mathbb{F}^{m,n}=mn$
假设 $m$ 和 $n$ 为正整数．按上面定义的加法和标量乘法，$\mathbb{F}^{m,n}$ 是维数为 $mn$ 的向量空间．
```

```{admonition} 定义：矩阵乘法（matrix multiplication）
假设 $\mathbf{A}$ 是 $m \times n$ 矩阵且  $\mathbf{B}$ 是 $n \times p$ 矩阵．那么 $\mathbf{AB}$ 定义为 $m \times p$ 矩阵，其中第 $j$ 行第 $k$ 列的元素由下式给出：

$$
(\mathbf{AB})_{j,k} = \sum_{r=1}^n A_{j,r} B_{r,k}
$$

于是，取 $\mathbf{A}$ 的第 $j$ 行和 $\mathbf{B}$ 的第 $k$ 列，将它们对应位置上的元素分别相乘再相加，就得
到了 $\mathbf{AB}$ 第 $j$ 行第 $k$ 列的元素．
```

```{admonition} 线性映射之积的矩阵
假设 $\mathcal{T} \in \mathscr{L}(\mathbb{U}, \mathbb{V})$ 且 $\mathcal{S} \in \mathscr{L}(\mathbb{V}, \mathbb{W})$．那么 $\mathscr{M}(\mathcal{S} \mathcal{T}) = \mathscr{M}(\mathcal{S}) \mathscr{M}(\mathcal{T})$．
```

```{admonition} 记号：$\mathbf{A}_{j,.}$、$\mathbf{A}_{.,k}$
假设 $\mathbf{A}$ 是 $m \times n$ 矩阵，
- 如果 $1 \leq j \leq m$，那么 $\mathbf{A}_{j,.}$ 表示由 $\mathbf{A}$ 的第 $j$ 行构成的 $1 \times n$ 的矩阵。
- 如果 $1 \leq k \leq n$，那么 $\mathbf{A}_{.,k}$ 表示由 $\mathbf{A}$ 的第 $k$ 列构成的 $m \times 1$ 的矩阵。
```

```{admonition} 矩阵之积的元素等于行乘以列
假设 $\mathbf{A}$ 是 $m \times n$ 矩阵，$\mathbf{B}$ 是 $n \times p$ 矩阵，那么 $\mathbf{AB}$ 第 $j$ 行第 $k$ 列的元素等于 $\mathbf{A}_{j,.}$ 与 $\mathbf{B}_{.,k}$ 的点积，即：

$$
(\mathbf{AB})_{j,k} = \mathbf{A}_{j,.} \cdot \mathbf{B}_{.,k}
$$
```

```{admonition} 矩阵之积的列等于矩阵与列之积
假设 $\mathbf{A}$ 是 $m \times n$ 矩阵，$\mathbf{B}$ 是 $n \times p$ 矩阵，那么 $\mathbf{AB}$ 的第 $k$ 列等于 $\mathbf{A}$ 与 $\mathbf{B}_{.,k}$ 的点积，即：

$$
(\mathbf{AB})_{.,k} = \mathbf{A} \mathbf{B}_{.,k}
$$
```

```{admonition} 列的线性组合
假设 $\mathbf{A}$ 是 $m \times n$ 矩阵，$\mathbf{x}=\begin{bmatrix} x_1 \\ \vdots \\ x_n \end{bmatrix}$ 是 $n \times 1$ 列向量，那么 $\mathbf{Ax}$ 是 $\mathbf{A}$ 中各列的线性组合，而与这些列相乘的标量来自 $\mathbf{x}$，即：

$$
\mathbf{Ax} = x_1 \mathbf{A}_{.,1} + \cdots + x_n \mathbf{A}_{.,n}
$$
```

```{admonition} 将矩阵乘法视为列或行的线性组合
假设 $\mathbf{C}$ 是 $m \times c$ 矩阵，$\mathbf{R}$ 是 $c \times n$ 矩阵.

1. 若 $k \in \{1, \cdots, c\}$，那么 $\mathbf{CR}$ 的第 $k$ 列是 $\mathbf{C}$ 各列的线性组合，其中各系数来自 $\mathbf{R}$ 的第 $k$ 列。
2. 若 $j \in \{1, \cdots, m\}$，那么 $\mathbf{CR}$ 的第 $j$ 行是 $\mathbf{R}$ 的各行的线性组合，其中各系数来自 $\mathbf{C}$ 的第 $j$ 行．
```

```{admonition} 定义：列秩（column rank）、行秩（row rank）
假设 $\mathbf{A}$ 是 $m\times n$ 矩阵，其各元素属于 $\mathbb{F}$．
- $\mathbf{A}$ 的**列秩**是 $\mathbf{A}$ 的各列在 $\mathbb{F}^{m,1}$ 中的张成空间的维数．
- $\mathbf{A}$ 的**行秩**是 $\mathbf{A}$ 的各行在 $\mathbb{F}^{1,n}$ 中的张成空间的维数．
```

```{admonition} 定义：转置（transpose）、$\mathbf{A}^t$
矩阵 $\mathbf{A}$ 的转置记为 $\mathbf{A}^t$，是互换 $\mathbf{A}$ 的行和列所得的矩阵．具体地说，如果 $\mathbf{A}$ 是 $m \times n$ 矩阵，那么 $\mathbf{A}^t$ 是 $n \times m$ 矩阵，其中各元素由下面等式给出：

$$
(\mathbf{A}^t)_{k,j} = A_{j,k}
$$
```

```{admonition} 行列分解（column-row factorization）
假设 $\mathbf{A}$ 是 $m \times n$ 矩阵，其中各元素均属于 $\mathbb{F}$ 中且列秩 $c\ge 1$. 那么存在各元素均属于  $\mathbb{F}$ 的 $m \times c$ 矩阵 $\mathbf{C}$ 和 $c \times n$ 矩阵 $\mathbf{R}$，使得 $\mathbf{A} = \mathbf{C}\mathbf{R}$ 成立。
```

```{admonition} 列秩等于行秩
假设 $\mathbf{A} \in \mathbb{F}^{m,n}$，那么 $\mathbf{A}$ 的列秩等于 $\mathbf{A}$ 的行秩．
```

```{admonition} 定义：秩（rank）
矩阵 $\mathbf{A} \in \mathbb{F}^{m,n}$ 的秩是 $\mathbf{A}$ 的列秩．
```

```{admonition} 定义：可逆的（invertible）、逆（inverse）
- 对于线性映射 $\mathcal{T} \in \mathscr{L}(\mathbb{V}, \mathbb{W})$，如果存在线性映射 $\mathcal{S} \in \mathscr{L}(\mathbb{W}, \mathbb{V})$，使得 $\mathcal{ST}$ 等于 $\mathbb{𝑉}$ 上的恒等算子且 $\mathcal{TS}$ 等于 $\mathbb{W}$  上的恒等算子，则称 $\mathcal{T}$ 是**可逆的**．
- 满足 $\mathcal{ST=I}$ 及 $\mathcal{TS=I}$ 的线性映射 $\mathcal{S} \in \mathscr{L}(\mathbb{W}, \mathbb{V})$ 被称为 $\mathcal{T}$ 的一个**逆**．（注意，第一个 $\mathcal{I}$ 是 $\mathbb{𝑉}$ 上的恒等算子，第二个 $\mathcal{I}$ 是 $\mathbb{W}$ 上的恒等算子）
```

```{admonition} 逆是唯一的
可逆的线性映射具有唯一的逆．
```

```{admonition} 记号：$\mathcal{T}^{-1}$
如果 $\mathcal{T}$ 是可逆的，那么它的逆记作 $\mathcal{T}^{-1}$．换言之，如果 $\mathcal{T} \in \mathscr{L}(\mathbb{V}, \mathbb{W})$ 是可逆的，那么 $\mathcal{T}^{-1}$$\mathcal{T}^{-1}$ 是  $\mathscr{L}(\mathbb{W}, \mathbb{V})$ 中唯一使得 $\mathcal{T}^{-1}\mathcal{T}=\mathcal{I}$ 和 $\mathcal{T}\mathcal{T}^{-1}=\mathcal{I}$ 成立的元素．
```

```{admonition} 可逆性 $\iff$ 单射性和满射性
线性映射是可逆的，当且仅当它既是单射又是满射．
```

```{admonition} 若 $\dim \mathbb{V} = \dim \mathbb{W} < \infty $，则单射性与满射性等价
假设 $\mathbb{V}$ 和 $\mathbb{W}$ 都是有限维向量空间，$\dim \mathbb{V} = \dim \mathbb{W}$，且 $\mathcal{T} \in \mathscr{L}(\mathbb{V}, \mathbb{W})$．那么
$\mathcal{T}$ 可逆 $\iff$ $\mathcal{T}$ 是单射 $\iff$ $\mathcal{T}$ 是满射．
```
 
```{admonition} $\mathcal{ST=I} \iff \mathcal{TS=I}$ ($\mathcal{T}$ 和 $\mathcal{S}$ 作用于维数相同的向量空间)
假设 $\mathbb{V}$ 和 $\mathbb{W}$ 是维数相同的有限维向量空间，$\mathcal{S} \in \mathscr{L}(\mathbb{W}, \mathbb{V})$ 且 $\mathcal{T} \in \mathscr{L}(\mathbb{V}, \mathbb{W})$．那么 $\mathcal{ST=I} \iff \mathcal{TS=I}$ 成立．
```

```{admonition} 定义：同构（isomorphism）、同构的（isomorphic）
- 同构就是可逆线性映射．
- 对于两个向量空间，若存在将其中一个向量空间映成另一个向量空间的同构，则称它们是同构的．
```

```{admonition} 维数表明了向量空间是否同构
对于 $\mathbb{F}$ 上的两个有限维向量空间，当且仅当它们的维数相同时，它们才是同构的．
```

```{admonition} $\mathscr{L}(\mathbb{V}, \mathbb{W})$ 与 $\mathbb{F}^{m\times n}$ 是同构的
设 $v_1, \cdots, v_n$ 是 $\mathbb{V}$ 上的一个基，$w_1, \cdots, w_m$ 是 $\mathbb{W}$ 上的一个基．那么 $\mathscr{M}$ 是 $\mathscr{L}(\mathbb{V}, \mathbb{W})$ 与 $\mathbb{F}^{m\times n}$ 是间的同构.
```

```{admonition} $\dim \mathscr{L}(\mathbb{V}, \mathbb{W})=(\dim \mathbb{V}) (\dim \mathbb{W})$
假设 $\mathbb{V}$ 和 $\mathbb{W}$ 都是有限维向量空间，那么 $\mathscr{L}(\mathbb{V}, \mathbb{W})$ 是有限维的，且 $\dim \mathscr{L}(\mathbb{V}, \mathbb{W})=(\dim \mathbb{V}) (\dim \mathbb{W})$ 成立．
```

```{admonition} 定义：向量的矩阵（matrix of a vector）$\mathscr{M}(\mathbf{v})$
假设 $\mathbf{v} \in \mathbb{V}$ 且 $v_1, \cdots, v_n$ 是 $\mathbb{V}$ 上的一个基. $\mathbf{v}$ 关于该基的矩阵是 $n\times 1$ 矩阵

$$
\mathscr{M}(\mathbf{v}) = \begin{bmatrix}
    b_1 \\
    \vdots \\
    b_n
\end{bmatrix},
$$

其中 $b_1, \cdots, b_n$ 是使得下式成立的标量：

$$
\mathbf{v} = b_1 v_1 + \cdots + b_n v_n.
$$
```

```{admonition} $\mathscr{M}(\mathcal{T})_{.,k}=\mathscr{M}(\mathcal{T} \mathbf{v}_k)$ 
设 $\mathcal{T} \in \mathscr{L}(\mathbb{V}, \mathbb{W})$ 且 $v_1, \cdots, v_n$ 是 $\mathbb{V}$ 上的一个基，$w_1, \cdots, w_m$ 是 $\mathbb{W}$ 上的一个基．令 $1\le k \le n$. 那么 $\mathscr{M}(\mathcal{T})$ 的第 $k$ 列，记作 $\mathscr{M}(\mathcal{T})_{.,k}$ 就等于 $\mathscr{M}(\mathcal{T} \mathbf{v}_k)$
```

```{admonition} 线性映射的作用就像矩阵乘法
假设 $\mathcal{T} \in \mathscr{L}(\mathbb{V}, \mathbb{W})$ 且 $v \in \mathbb{V}$．假设 $v_1, \cdots, v_n$ 是 $\mathbb{V}$ 上的一个基且 $w_1, \cdots, w_m$ 是 $\mathbb{W}$ 上的一个基．那么

$$
\mathscr{M}(\mathcal{T} \mathbf{v}) = \mathscr{M}(\mathcal{T}) \mathscr{M}(\mathbf{v})
$$
```

```{admonition} $\operatorname{range}\mathcal{T}$ 的维数等于 $\mathscr{M}(\mathcal{T})$ 的列秩
假设 $\mathbb{V}$ 和 $\mathbb{W}$ 都是有限维向量空间，$\mathcal{T} \in \mathscr{L}(\mathbb{V}, \mathbb{W})$，那么 $\dim \operatorname{range}\mathcal{T}$ 等于 $\mathscr{M}(\mathcal{T})$ 的列秩．
```

```{admonition} $\mathscr{M}(\mathcal{T}, (v_1, \cdots, v_n))$
如果 $\mathcal{T}\in \mathscr{L}(\mathbb{V})$ 且 $v_1, \cdots, v_n$ 是 $\mathbb{V}$ 上的一个基，那么记号 

$$
\mathscr{M}(\mathcal{T}, (v_1, \cdots, v_n)) = \mathscr{M}(\mathcal{T}, (v_1, \cdots, v_n), (v_1, \cdots, v_n))
$$
```

```{admonition} 定义：恒等矩阵（identity matrix） $\mathbf{I}$
设 $n$ 为正整数．仅对角线上（即那些行号和列号相等的位置）的元素为 $1$ 而其他各元素均为 $0$ 的 $n \times n$ 矩阵

$$
\begin{bmatrix}
    1 & & 0 \\
    &\ddots& \\
    0 & & 0
\end{bmatrix}
$$
就称为恒等矩阵，记作 $\mathbf{I}$．
```

```{admonition} 定义：可逆的（invertible），逆（inverse）、$\mathbf{A}^{-1}$
设 $\mathbf{A}$ 是一个 $n \times n$ 矩阵．如果存在一个 $n \times n$ 矩阵 $\mathbf{B}$，使得

$$
\mathbf{AB} = \mathbf{BA} = \mathbf{I}
$$

那么就称 $\mathbf{A}$ 是可逆的，且 $\mathbf{B}$ 是 $\mathbf{A}$ 的逆矩阵，记作 $\mathbf{A}^{-1}$．
```

```{note} 有些数学家使用术语“非奇异”（nonsingular）和“奇异”（singular），它们分别与“可
逆”和“不可逆”同义．
```

```{admonition} 线性映射之积的矩阵
设 $\mathcal{T} \in \mathscr{L}(\mathbb{U}, \mathbb{V})$ 和 $\mathcal{S} \in \mathscr{V}(\mathbb{V}, \mathbb{W})$．若 $u_1, \cdots, u_m$ 是 $\mathbb{U}$ 上的一个基，$v_1, \cdots, v_n$ 是 $\mathbb{V}$ 上的一个基，$w_1, \cdots, w_p$ 是 $\mathbb{W}$ 上的一个基，那么

$$
\mathscr{M}(\mathcal{ST}, (u_1, \cdots, u_m), (w_1, \cdots, w_p)) = \mathscr{M}(\mathcal{S}, (v_1, \cdots, v_n), (w_1, \cdots, w_p)) \mathscr{M}(\mathcal{T}, (u_1, \cdots, u_m), (v_1, \cdots, v_n))
$$
```

```{admonition} 恒等算子关于两个基的矩阵
假设 $v_1, \cdots, v_n$ 和 $v_1, \cdots, v_n$ 是 $\mathbb{V}$ 上的两个基，那么矩阵

$\mathscr{M}(\mathcal{I}, (u_1, \cdots, u_n), (v_1, \cdots, v_n)) = \mathbf{I}$ 和 $\mathscr{M}(\mathcal{I}, (v_1, \cdots, v_n), (u_1, \cdots, u_n)) = \mathbf{I}$

都是可逆的，且互为对方的逆．
```

```{admonition} 换基公式（change-of-basis formula）
设 $\mathcal{T} \in \mathscr{L}(\mathbb{V})$，假设 $u_1, \cdots, u_n$ 和 $v_1, \cdots, v_n$ 是 $\mathbb{V}$ 上的一个基. 令

$$
\mathbf{A} = \mathscr{M}(\mathcal{T}, (u_1, \cdots, u_n)) \text{且} \mathbf{B}=\mathscr{M}(\mathcal{T}, (v_1, \cdots, v_n))
$$

且 $\mathbf{C} = \mathscr{M}(\mathcal{I}, (u_1, \cdots, u_n), (v_1, \cdots, v_n))$，那么

$$
\mathbf{A} = \mathbf{C}^{-1} \mathbf{B} \mathbf{C}
$$
```

```{admonition} 逆的矩阵等于矩阵的逆
设 $v_1, \cdots, v_n$ 是 $\mathbb{V}$ 的基且 $\mathcal{T} \in \mathscr{L}(\mathbb{V})$ 是可逆的．那么 $\mathscr{M}(\mathcal{T}^{-1}) = (\mathscr{M}(\mathcal{T}))^{-1}$，式中两个矩阵均是关于基
$v_1, \cdots, v_n$ 的。
```