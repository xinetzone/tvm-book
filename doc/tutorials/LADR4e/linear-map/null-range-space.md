# 零空间和值域

```{admonition} 定义：零空间（null space）、$\operatorname{null} \mathcal{T}$
对于 $\mathcal{T} \in \mathcal{L}(\mathbb{V}, \mathbb{W})$，$\mathcal{T}$ 的零空间记为 $\operatorname{null} \mathcal{T}$，是 $\mathbb{V}$ 的子集，其由被 $\mathcal{T}$ 映射到 $0$ 的所有向量构成：

$$
\operatorname{null} \mathcal{T} = \{ v \in \mathbb{V} : \mathcal{T}v = 0 \}.
$$
```

```{admonition} 零空间是子空间（null space is a subspace）
假设 $\mathcal{T} \in \mathcal{L}(\mathbb{V}, \mathbb{W})$ 是由 $\mathbb{V}$ 到 $\mathbb{W}$ 的线性映射．那么 $\operatorname{null} \mathcal{T}$ 是 $\mathbb{V}$ 的子空间．
```

```{admonition} 定义：单射（injective）
对于函数 $\mathcal{T}: \mathbb{V} \to \mathbb{W}$，若 $\mathcal{Tu = Tv} \implies u = v$，那么 $\mathcal{T}$ 被称为单射．
```

```{note}
逆否命题：如果 $u \ne v$ 蕴涵 $\mathcal{T}u \ne \mathcal{T}v$，那么 $\mathcal{T}$ 是单射．

术语“一对一的”（one-to-one）和单射的意思一样．
```

```{admonition} 单射性 $\iff$ 零空间等于 $\{0\}$
令 $\mathcal{T} \in \mathcal{L}(\mathbb{V}, \mathbb{W})$．那么 $\mathcal{T}$ 是单射当且仅当 $\operatorname{null}\mathcal{T} = \{0\}$．
```

```{admonition} 定义：值域（range）
对于 $\mathcal{T} \in \mathcal{L}(\mathbb{V}, \mathbb{W})$，$\mathcal{T}$ 的值域是 $\mathbb{W}$ 的子集，由所有等于 $\mathcal{T}v$（其中 $v\in \mathbb{V}$）的向量构成：

$$
\operatorname{range} \mathcal{T} = \{ \mathcal{T}v : v \in \mathbb{V} \}.
$$ 
```

```{admonition} 值域是子空间
假设 $\mathcal{T} \in \mathcal{L}(\mathbb{V}, \mathbb{W})$ 是由 $\mathbb{V}$ 到 $\mathbb{W}$ 的线性映射．那么 $\operatorname{range} \mathcal{T}$ 是 $\mathbb{W}$ 的子空间．
```

```{admonition} 定义：满射（surjective）
如果函数 $\mathcal{T}: \mathbb{V} \to \mathbb{W}$ 的值域等于 $\mathbb{W}$，则称该函数为满射．
```

```{admonition} 线性映射基本定理（fundamental theorem of linear maps）
假设 $\mathbb{V}$ 是有限维的且 $\mathcal{T} \in \mathcal{L}(\mathbb{V}, \mathbb{W})$，那么 $\operatorname{range}\mathcal{T}$ 是有限维的，且

$$
\dim \mathbb{V} = \dim \operatorname{null} \mathcal{T} + \dim \operatorname{range} \mathcal{T}．
$$
```

```{admonition} 映到更低维空间上的线性映射不是单射
假设 $\mathbb{V}$ 和 $\mathbb{W}$ 是有限维向量空间且满足 $\dim\mathbb{V} \gt \dim \mathbb{W}$．那么从 $\mathbb{V}$ 到 $\mathbb{W}$ 的线性映射一定不是单射．
```

```{admonition} 映到更高维空间上的线性映射不是满射
假设 $\mathbb{V}$ 和 $\mathbb{W}$ 是有限维向量空间且满足  $\dim\mathbb{V} \lt \dim \mathbb{W}$．那么从 $\mathbb{V}$ 到 $\mathbb{W}$ 的线性映射一定不是满射．
```

```{admonition} 齐次线性方程组
未知数个数多于方程个数的齐次线性方程组具有非零解。

注：齐次的（homogeneous），在这个语境里，意为每个方程右端的常数项都是 $0$．
```

```{admonition} 方程个数多于未知数个数的线性方程组
方程个数多于未知数个数的线性方程组当常数项取某些值时无解．
```

