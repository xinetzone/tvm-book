# 向量空间的积和商

```{admonition} 定义：向量空间的积（product of vector spaces）
设 $\mathbb{V}_1, \cdots, \mathbb{V}_m$ 都是 $\mathbb{F}$ 上的向量空间．

- 乘积 $\mathbb{V}_1 \times \cdots \times \mathbb{V}_m$ 定义为
$$
\mathbb{V}_1 \times \cdots \times \mathbb{V}_m = \left\{ (v_1, \cdots, v_m) : v_1 \in \mathbb{V}_1,\cdots, v_m \in \mathbb{V}_m \right\}
$$
- $\mathbb{V}_1 \times \cdots \times \mathbb{V}_m$ 上的加法定义为
$$
(v_1, \cdots, v_m) + (w_1, \cdots, w_m) = (v_1 + w_1, \cdots, v_m + w_m)
$$
- $\mathbb{V}_1 \times \cdots \times \mathbb{V}_m$ 上的标量乘法定义为
$$
\lambda(v_1, \cdots, v_m) = (\lambda v_1, \cdots, \lambda v_m)
$$
```

```{admonition} 向量空间的积是向量空间
设 $\mathbb{V}_1, \cdots, \mathbb{V}_m$ 都是 $\mathbb{F}$ 上的向量空间．那么 $\mathbb{V}_1 \times \cdots \times \mathbb{V}_m$ 是 $\mathbb{F}$ 上的向量空间．
```

```{admonition} 向量空间之积的维数是各向量空间维数之和
设 $\mathbb{V}_1, \cdots, \mathbb{V}_m$ 都是 $\mathbb{F}$ 上的向量空间．那么 $\mathbb{V}_1 \times \cdots \times \mathbb{V}_m$ 是有限维的且

$$\dim (\mathbb{V}_1 \times \cdots \times \mathbb{V}_m) = \dim \mathbb{V}_1 + \cdots + \dim \mathbb{V}_m$$
```

```{admonition} 积与直和
设 $\mathbb{V}_1, \cdots, \mathbb{V}_m$ 是 $\mathbb{V}$ 的子空间．由下式定义线性映射 $\Gamma : \mathbb{V}_1 \times \cdots \times \mathbb{V}_m \to \mathbb{V}_1 + \cdots + \mathbb{V}_m$：

$$
\Gamma(v_1, \cdots, v_m) = v_1 + \cdots + v_m
$$

那么 $\mathbb{V}_1 + \cdots + \mathbb{V}_m$ 是直和，当且仅当 $\Gamma$ 是单射．
```

```{admonition} 向量空间的和是直和，当且仅当该和的维数等于各求和项维数之和
设 $\mathbb{V}_1, \cdots, \mathbb{V}_m$ 是有限维的 $\mathbb{V}$ 的子空间．那么 $\mathbb{V}_1 + \cdots + \mathbb{V}_m$ 是直和，当且仅

$$\dim (\mathbb{V}_1 + \cdots + \mathbb{V}_m) = \dim \mathbb{V}_1 + \cdots + \dim \mathbb{V}_m$$
```

## 商空间（quotient space）

```{admonition} 记号：$\mathbf{v} + \mathbb{U}$
设 $\mathbf{v} \in \mathbb{V}$ 且 $\mathbb{U} \subseteq \mathbb{V}$，那么 $\mathbf{v} + \mathbb{U}$ 是 $\mathbb{V}$ 的一个由下式定义的子集：
$$
\mathbf{v} + \mathbb{U} = \left\{ \mathbf{v} + \mathbf{u} : \mathbf{u} \in \mathbb{U} \right\}
$$
```

```{admonition} 定义：平移（translate）
对于 $\mathbf{v} \in \mathbb{V}$ 且 $\mathbb{U} \subseteq \mathbb{V}$，称集合 $\mathbf{v}+\mathbb{U}$ 是 $\mathbb{U}$ 的一个平移。
```

```{admonition} 定义：商空间（quotient space）$\mathbb{V} / \mathbb{U}$
设 $\mathbb{U}$ 是 $\mathbb{V}$ 的子空间．那么商空间 $\mathbb{V} / \mathbb{U}$ 是由 $\mathbb{U}$ 的所有平移构成的集合．从而：
$$
\mathbb{V} / \mathbb{U} = \left\{ \mathbf{v} + \mathbb{U} : \mathbf{v} \in \mathbb{V} \right\}
$$
```

```{admonition} 子空间的两个平移要么相等要么不相交
设 $\mathbb{U}$ 是 $\mathbb{V}$ 的子空间且 $\mathbf{v,w} \in \mathbb{V}$．那么 

$$
\mathbf{v-w} \in \mathbb{U} \iff \mathbf{v} + \mathbb{U}= \mathbf{w} + \mathbb{U}\iff (\mathbf{v} + \mathbb{U}) \cap (\mathbf{w} + \mathbb{U}) \neq \emptyset
$$
```

```{admonition} 定义：$\mathbb{V} / \mathbb{U}$ 上的加法和标量乘法（addition and scalar multiplication on $\mathbb{V} / \mathbb{U}$）
设 $\mathbb{U}$ 是 $\mathbb{V}$ 的子空间．那么 $\mathbb{V} / \mathbb{U}$ 上的加法和标量乘法分别由下面两式定义：对所有 $\mathbf{v,w} \in \mathbb{V}$ 和所有 $\lambda \in \mathbb{F}$

$$
\begin{aligned}
(\mathbf{v} + \mathbb{U}) + (\mathbf{w} + \mathbb{U}) &= (\mathbf{v} + \mathbf{w}) + \mathbb{U} \\
\lambda (\mathbf{v} + \mathbb{U}) &= \lambda \mathbf{v} + \mathbb{U}
\end{aligned}
$$
```

```{admonition} 商空间是向量空间
假设 $\mathbb{U}$ 是 $\mathbb{V}$ 的子空间．那么带有定义如上的加法和标量乘法的 $\mathbb{V} / \mathbb{U}$ 就是向量空间．
```

```{admonition} 定义：商映射（quotient map）$\pi$
设 $\mathbb{U}$ 是 $\mathbb{V}$ 的子空间．那么商映射 $\pi : \mathbb{V} \to \mathbb{V} / \mathbb{U}$ 是由下式定义的线性映射：对所有 $\mathbf{v} \in \mathbb{V}$，
$$
\pi(\mathbf{v}) = \mathbf{v} + \mathbb{U}
$$
```

```{admonition} 商空间的维数
设 $\mathbb{V}$ 是有限维的，$\mathbb{U}$ 是 $\mathbb{V}$ 的子空间．那么

$$
\dim \mathbb{V} / \mathbb{U} = \dim \mathbb{V} - \dim \mathbb{U}
$$
```

```{admonition} 记号：$\widetilde{\mathcal{T}}$
设 $\mathcal{T} \in \mathscr{L}(\mathbb{V}, \mathbb{W})$. $\widetilde{\mathcal{T}}: \mathbb{V}/ \operatorname{null}\mathcal{T} \to \mathbb{W}$ 由下式定义：

$$
\widetilde{\mathcal{T}}(\mathbf{v} +  \operatorname{null} \mathcal{T}) = \mathcal{T}\mathbf{v}
$$
```

```{admonition} $\widetilde{\mathcal{T}}$ 的零空间和值域
设 $\mathcal{T} \in \mathscr{L}(\mathbb{V}, \mathbb{W})$．那么
1. $\widetilde{\mathcal{T}} \circ \pi = \mathcal{T}$，其中 $\pi$ 是将 $\mathbb{V}$ 映射到 $\mathbb{V} / (\operatorname{null} \mathcal{T})$ 的商映射；
2. $\widetilde{\mathcal{T}}$ 是单射；
3. $\operatorname{range}\widetilde{\mathcal{T}} = \operatorname{range}\mathcal{T}$
4. $\mathbb{V} / (\operatorname{null} \mathcal{T})$ 和 $\operatorname{range}\mathcal{T}$ 是同构的向量空间。
```
