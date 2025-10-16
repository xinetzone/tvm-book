# 对偶

```{admonition} 定义：线性泛函（linear functional）
$\mathbb{V}$ 上的**线性泛函**是 $\mathbb{V}$到 $\mathbb{F}$ 的线性映射。换言之，线性泛函是 $\mathscr{L}(\mathbb{V}, \mathbb{F})$ 的元素。
```

```{admonition} 定义：对偶空间（dual space）$\mathbb{V}^{\prime}$
$\mathbb{V}$ 的对偶空间，记作 $\mathbb{V}^{\prime}$，是 $\mathbb{V}$ 上全体线性泛函所构成的向量空间。换言之，$\mathbb{V}^{\prime} = \mathscr{L}(\mathbb{V},\mathbb{F})$
```

```{admonition} $\dim \mathbb{V}^{\prime} = \dim \mathbb{V}$
假设 $\mathbb{V}$ 是有限维向量空间，那么 $\mathbb{V}^{\prime}$ 也有限维，且 $\dim \mathbb{V}^{\prime} = \dim \mathbb{V}$。
```

```{admonition} 定义：对偶基（dual basis）
若 $\mathbf{v}_1, \ldots, \mathbf{v}_n$ 是 $\mathbb{V}$ 的一个基，那么它的对偶基是 $\mathbb{V}^{\prime}$ 中的元素 $\varphi_1, \ldots, \varphi_n$ 所构成的组，其中各 $\varphi_i$ 满足下式的线性泛函：
$$
\varphi_j(\mathbf{v}_k) = \begin{cases}
1 & \text{若 } j = k \\
0 & \text{若 } j \neq k
\end{cases}
$$
```

```{admonition} 对偶基给出了线性组合的系数
假设 $v_1,\cdots,v_n$ 是 𝑉 的基，且 $\varphi_1, \ldots, \varphi_n$ 是其对偶基．那么对每个 $v\in \mathbb{V}$，有
$$
v = \sum_{j=1}^n \varphi_j(v)v_j
$$
```

```{admonition} 对偶基是对偶空间的基
假设 $\mathbb{V}$ 是有限维的．那么 $\mathbb{V}$ 的基的对偶基是 $\mathbb{V}^{\prime}$ 的基．
```

```{admonition} 定义：对偶映射（dual map）$\mathcal{T}^{\prime}$
若 $\mathcal{T} \in \mathscr{L}(\mathbb{V}, \mathbb{W})$. $\mathcal{T}$ 的对偶映射是由下式定义的线性映射 $\mathcal{T}^{\prime} \in \mathscr{L}(\mathbb{W}^{\prime}, \mathbb{V}^{\prime})$：对于每个 $\varphi \in \mathbb{W}^{\prime}$，有
$$
\mathcal{T}^{\prime}(\varphi) = \varphi \circ \mathcal{T}
$$
```

```{admonition} 对偶映射的代数性质
设 $\mathcal{T} \in \mathscr{L}(\mathbb{V}, \mathbb{W})$，那么
1. 对于所有 $\mathcal{S} \in \mathscr{L}(\mathbb{V}, \mathbb{W})$，有 $\mathcal{(S+T)}^{\prime} = \mathcal{S}^{\prime} + \mathcal{T}^{\prime}$;
2. 对于所有 $\lambda \in \mathbb{F}$，有 $\mathcal{(\lambda T)}^{\prime} = \lambda \mathcal{T}^{\prime}$;
3. 对于所有 $\mathcal{S} \in \mathscr{L}(\mathbb{V}, \mathbb{W})$，有 $\mathcal{(ST)}^{\prime} = \mathcal{T}^{\prime} \mathcal{S}^{\prime}$;
```

