# 有限维向量空间

```{admonition} 线性组合（linear combination）
$\mathbb{V}$ 中向量组 $v_1, \cdots, v_m$ 的线性组合是形如 $a_1v_1 + \cdots + a_mv_m$ 的向量，其中 $a_1, \cdots, a_m \in \mathbb{F}$．
```

```{admonition} 张成空间（span）
$\mathbb{V}$ 中向量组 $v_1, \cdots, v_m$ 的所有线性组合所构成的集合称为 $v_1, \cdots, v_m$ 的张成空间，记作 $\operatorname{span}(v_1, \cdots , v_m)$．换言之，

$$
\operatorname{span}(v_1, \cdots , v_m) = \{a_1v_1 + \cdots + a_mv_m : a_1, \cdots, a_m \in \mathbb{F}\}
$$

定义空向量组 $( )$ 的张成空间为 $\{0\}$．
```

```{admonition} 向量组的张成空间是最小的包含组中所有向量的子空间
$\mathbb{V}$ 中向量组的张成空间是最小的包含此向量组中所有向量的 $\mathbb{V}$ 的子空间．
```

```{admonition} 张成（spans）
如果 $\operatorname{span}(v_1, \cdots , v_m) = \mathbb{V}$，就说 $v_1, \cdots, v_m$ 张成 $\mathbb{V}$．
```

```{admonition} 有限维向量空间（finite-dimensional vector space）
如果向量空间 $\mathbb{V}$ 可由其中某个向量组张成，则称该向量空间是有限维的．
```

## 多项式

```{admonition} 多项式（polynomial）、$\mathcal{P}_F$
对于函数 $p : \mathbb{F} \to \mathbb{F}$，如果存在 $a_0, \cdots, a_m \in \mathbb{F}$ 使得对所有 $z \in \mathbb{F}$ 都有

$$
p(z) = a_0 + a_1z + a_2z^2 + \cdots + a_m z^m
$$

则称 $p$ 为系数在 $\mathbb{F}$ 中的多项式．

$\mathcal{P}_\mathbb{F}$ 是系数在 $\mathbb{F}$ 中的全体多项式所构成的集合．

$\mathcal{P}_\mathbb{F}$ 是 $\mathbb{F^F}$（全体由 $\mathbb{F}$ 到 $\mathbb{F}$ 的函数所构成的向量空间）的子空间．
```

```{admonition} 多项式的次数（degree of a polynomial）、$\deg p$
对于多项式 $p \in \mathcal{P}_\mathbb{F}$，如果存在 $a_0, \cdots, a_m \in \mathbb{F}$ 且 $a_m \neq 0$ 使得对每个 $z \in \mathbb{F}$，都有

$$
p(z) = a_0 + a_1z + a_2z^2 + \cdots + a_m z^m
$$

那么就说 $p$ 的次数是 $m$．

规定恒等于 0 的多项式的次数为 $-\infty$．

多项式 $p$ 的次数记为 $\deg p$．
```

```{admonition} 记号 $\mathcal{P}_m(\mathbb{F})$
对于非负整数 $m$，$\mathcal{P}_m(\mathbb{F})$ 表示系数在 $\mathbb{F}$ 中且次数不高于 $m$ 的所有多项式所构成的集合．
```

```{admonition} 定义：无限维向量空间（infinite-dimensional vector space）
如果一个向量空间不是有限维的，就称它是无限维的．
```

```{admonition} 定义：线性无关（linearly independent）
对于 $\mathbb{V}$ 中的向量组 $v_1, \cdots, v_m$，如果使得 $a_1v_1 + \cdots + a_mv_m = 0$ 成立的 $a_1, \cdots, a_m \in \mathbb{F}$ 的唯一选取方式是 $a_1 = \cdots = a_m = 0$，那么称该向量组为线性无关的．

规定空向量组 $( )$ 也是线性无关的．
```

```{admonition} 定义：线性相关（linearly dependent）
如果 $\mathbb{V}$ 中的一个向量组不是线性无关的，就称它是线性相关的．换言之，对于 $\mathbb{V}$ 中的向量组 $v_1, \cdots, v_m$，如果存在不全为 $0$ 的 $a_1, \cdots, a_m \in \mathbb{F}$ 使得 $a_1v_1 + \cdots + a_mv_m = 0$，那么该向量组是线性相关的．
```

```{admonition} 线性相关性引理（linear dependence lemma）
设 $\mathbb{V}$ 中的向量组 $v_1, \cdots, v_m$ 是线性相关的．那么存在 $k \in \{1, 2, \cdots, m\}$ 满足 $v_k \in \operatorname{span}(v_1, \cdots, v_{k-1})$．进而，如果 $k$ 满足上述条件且从 $v_1, \cdots, v_m$ 中移除第 $k$ 项，那么剩余向量组成的向量组的张成空间仍等于 $\operatorname{span}(v_1, \cdots, v_m)$．
```

```{admonition} 线性无关组的长度 $\le$ 张成组的长度
在有限维向量空间中，每个线性无关向量组的长度小于或等于每个张成向量组的长度．
```

```{admonition} 有限维的子空间
有限维向量空间的每个子空间都是有限维的．
```

## 基（basis）

```{admonition} 定义：基（basis）
$\mathbb{V}$ 中线性无关且张成 $\mathbb{V}$ 的向量组称为 $\mathbb{V}$ 的基．
```

```{admonition} 基的判定准则
$\mathbb{V}$ 中向量组  $v_1, \cdots, v_m$ 是 $\mathbb{V}$ 的基，当且仅当每个 𝑣 ∈ $\mathbb{V}$ 都可以被唯一地写成这样的形式：

$$
v = a_1v_1 + \cdots + a_mv_m
$$

其中 $a_1, \cdots, a_m \in \mathbb{F}$．
```

```{admonition} 每个张成组都包含基
向量空间中的每个张成组都能被削减成该向量空间的基．
```

```{admonition} 有限维向量空间的基
每个有限维向量空间都有基．
```

```{admonition} 每个线性无关组都可被扩充成基
有限维向量空间中每个线性无关向量组都可被扩充成该向量空间的基．
```

```{admonition} $\mathbb{V}$ 的每个子空间都是等于 $\mathbb{V}$ 的直和的组成部分
假设 $\mathbb{V}$ 是有限维的，$\mathbb{U}$ 是 $\mathbb{V}$ 的子空间．那么存在 $\mathbb{V}$ 的子空间 $\mathbb{W}$，使得 $\mathbb{V} = \mathbb{U} \oplus \mathbb{W}$．
```

```{admonition} 复化（complexification）
设 $\mathbb{V}$ 是实向量空间．$\mathbb{V}$ 的复化（complexification）记为 $\mathbb{V_C}$，等于 $\mathbb{V} \times \mathbb{V}$．

$\mathbb{V_C}$ 中的元素为有序对 $(u, v)$，其中 $u, v \in \mathbb{V}$，不过我们把它写作 $u + i v$．$\mathbb{V_C}$ 上的加法定义为 

$$
(u_1 + i v_1) + (u_2 + i v_2) = (u_1 + u_2) + i(v_1 + v_2)
$$

对所有 $u_1, v_1, u_2, v_2 \in \mathbb{V}$ 成立．

$\mathbb{V_C}$ 上的复标量乘法定义为

$$
(a + bi)(u + i v) = (a u − b v) + i(a v + b u)
$$ 

对所有 $a, b \in \mathbb{R}$ 和所有 $u, v \in \mathbb{V}$ 成立．

证明：具有如上加法和标量乘法定义的 $\mathbb{V_C}$ 是复向量空间．

注：将 $u \in \mathbb{V}$ 等同于 $u + i0$ 从而将 $\mathbb{V}$ 视为 $\mathbb{V_C}$ 的子集．这样一来，由 $\mathbb{V}$ 构造 $\mathbb{V_C}$ 就可以视作由 $\mathbb{R}^n$ 构造 $\mathbb{C}^n$ 的推广．

证明：如果 $v_1, \cdots, v_n$ 是实向量空间 $\mathbb{V}$ 的基，那么 $v_1, \cdots, v_n$ 也是其复化 $\mathbb{V}_\mathbb{C}$（视为复向量空间）的基．
```

```{admonition} 基的长度不依赖于基的选取
有限维向量空间的任意两个基都有相同的长度．
```

```{admonition} 定义：维数（dimension）、$\dim \mathbb{V}$
有限维向量空间的维数是这个向量空间中任意一个基的长度．
有限维向量空间 $\mathbb{V}$ 的维数记作 $\dim \mathbb{V}$．
```

```{admonition} 子空间的维数
如果 $\mathbb{V}$ 是有限维的且 $\mathbb{U}$ 是 $\mathbb{V}$ 的子空间，那么 $\dim \mathbb{U} \le \dim \mathbb{V}$．
```

```{note}
实向量空间 $\mathbb{R}^2$ 的维数是 $2$；复向量空间 $\mathbb{C}$ 的维数是 $1$．作为集合，$\mathbb{R}^2$ 可以被认为与 $\mathbb{C}$ 等同（并且，在两个空间上，加法是相同的，如果标量取自实数域，那么标量乘法也相同）．于是，当我们讨论向量空间的维数时，不可忽视 $\mathbb{F}$ 的选取带来的影响．
```

```{admonition} 长度恰当的线性无关组是基
假设 $\mathbb{V}$ 是有限维的．那么 $\mathbb{V}$ 中每个长度为 $\dim \mathbb{V}$ 的线性无关向量组都是 $\mathbb{V}$ 的基．
```

```{admonition} 某空间中与之维数相同的子空间等于这整个空间
假设 $\mathbb{V}$ 是有限维的，$\mathbb{U}$ 是 $\mathbb{V}$ 的子空间且满足 $\dim \mathbb{U} = \dim \mathbb{V}$．那么 $\mathbb{U} = \mathbb{V}$．
```

```{admonition} 长度恰当的张成组是基
假设 $\mathbb{V}$ 是有限维的．那么 $\mathbb{V}$ 中每个长度为 $\dim \mathbb{V}$ 的张成组都是 $\mathbb{V}$ 的基．
```

```{admonition} 子空间之和的维数
如果 $\mathbb{V}$ 是有限维的，$\mathbb{U}$ 和 $\mathbb{W}$ 是 $\mathbb{V}$ 的子空间，那么

$$
\dim(\mathbb{U} + \mathbb{W}) = \dim \mathbb{U} + \dim \mathbb{W} - \dim(\mathbb{U} \cap \mathbb{W})．
$$
```
