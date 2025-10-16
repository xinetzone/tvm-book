# 向量空间

```{admonition} 向量空间(vector space)
向量空间 $\mathbb{V}$ 是一个集合 $\mathcal{V}$，且满足以下性质：
- **可交换性**（commutativity）
    对任意 $u, v \in \mathcal{V}$，都有 $u+v = v+u$。
- **可结合性**（associativity）
    对于所有 $u, v, w \in \mathcal{V}$ 以及所有 $a, b \in \mathbb{F}$，都有 $(u+v)+w = u+(v+w)$ 以及 $(ab)v = a(bv)$。  
- **加法恒等元**（additive identity）
    对于所有 $v \in \mathcal{V}$，都存在 $0 \in \mathcal{V}$ 使得 $v + 0 = v$。
- **加法逆元**（additive inverse）
    对于每个 $v \in \mathcal{V}$，都存在 $-v \in \mathcal{V}$ 使得 $v + (-v) = 0$。
- **乘法恒等元**（multiplicative identity）
    对于所有 $v \in \mathcal{V}$，都有 $1v = v$。
- **分配性质**（distributive properties）
    对于所有 $u, v \in \mathcal{V}$ 以及所有 $a, b \in \mathbb{F}$，都有 $a(u+v) = au + av$ 以及 $(a+b)v = av + bv$。
```

```{admonition} 定义
向量空间的元素被称作**向量**（vector）或**点**（point）.
```

向量空间上的标量乘法依赖于 $\mathbb{F}$ 的选取. 由此，当需要描述得更确切时，会说 $\mathbb{V}$ 是 $\mathbb{F}$ 上的向量空间（vector space over $\mathbb{F}$），而不是仅仅说 $\mathbb{V}$ 是向量空间. 例如，$\mathbb{R}^n$ 是 $\mathbb{R}$ 上的向量空间，而 $\mathbb{C}^n$ 是 $\mathbb{C}$ 上的向量空间.

```{admonition} 记号 $\mathbb{F^S}$
- 若 $\mathbb{S}$ 是集合，则 $\mathbb{F^S}$ 表示从 $\mathbb{S}$ 到 $\mathbb{F}$ 的所有函数构成的集合。
- 对于 $f,g \in \mathbb{F^S}$，和 $f+g \in \mathbb{F^S}$ 是由下式定义的函数：对于所有 $x \in \mathbb{S}$，$(f+g)(x) = f(x) + g(x)$
- 对于 $\lambda \in \mathbb{F}$ 和 $f \in \mathbb{F^S}$，乘积 $\lambda f \in \mathbb{F}$ 是由下式定义的函数：对于所有 $x \in \mathbb{S}$，$(\lambda f)(x) = \lambda f(x)$.
```

举个上述记号的具体例子：如果 $\mathbb{S}$ 是区间 $[0, 1]$ 且 $\mathbb{F}$ = $\mathbb{R}$，那么 $\mathbb{R^{[0,1]}}$ 是全体定义在区间 $[0, 1]$ 上的实值函数所构成的集合．

向量空间 $\mathbb{F}^n$ 是向量空间 $\mathbb{F^S}$ 上的特例，因为每个 $(x_1, \cdots, x_n) \in \mathbb{F}^n$ 都可被视作从集合 $\{1, 2, \cdots, n\}$ 到 $\mathbb{F}$  的函数 $x$，只要将 $(x_1, \cdots, x_n)$ 的第 $k$ 个坐标写成 $x(k)$ 而不是 $x_k$
就可看出．换句话说，可以将 $\mathbb{F}^n$ 看成 $\mathbb{F}^{\{1,2,\cdots,n\}}$．类似地，可以将 $\mathbb{F}^{\infty}$ 看成 $\mathbb{F}^{\{1,2,\cdots\}}$．

```{admonition} 加法恒等元唯一
向量空间有唯一的加法恒等元 $\mathbf{0}$．
```

```{admonition} 加法逆元唯一
向量空间 $\mathbb{V}$ 里的每个元素 $\mathbf{v} \in \mathbb{V} $ 都有唯一的加法逆元 $-\mathbf{v} \in \mathbb{V}$．

$\mathbf{w - v}$ 定义为 $\mathbf{w + (−v)}$．
```

```{admonition} 子空间（subspace）
如果 $\mathbb{V}$ 的子集 $\mathbb{U}$ 是与 $\mathbb{V}$ 具有相同的加法恒等元、加法和标量乘法运算的向量空间，那么 $\mathbb{U}$ 就称为 $\mathbb{V}$ 的子空间．
```

```{admonition} 子空间的条件
当且仅当 $\mathbb{V}$ 的子集 $\mathbb{U}$ 满足以下三个条件时，$\mathbb{U}$ 是 $\mathbb{V}$ 的子空间．
- 加法恒等元（additive identity）：
    $\mathbf{0} \in \mathbb{U}$
- 对于加法封闭（closed under addition）：
    $\mathbf{u}, \mathbf{w} \in \mathbb{U}$ 意味着 $\mathbf{u} + \mathbf{w} \in \mathbb{U}$
- 对于标量乘法封闭（closed under scalar multiplication）：
    $\lambda \in \mathbb{F}$ 且 $\mathbf{u} \in \mathbb{U}$ 意味着 $\lambda \mathbf{u} \in \mathbb{U}$
```

```{admonition} 子空间的和（sum of subspaces）
假设 $\mathbb{V}_1, \cdots, \mathbb{V}_m$ 是 $\mathbb{V}$ 的子空间．$\mathbb{V}_1 + \cdots + \mathbb{V}_m$ 是由 $\mathbb{V}_1, \cdots, \mathbb{V}_m$ 中元素所有可能的和所构成的集合，记作 $\mathbb{V}_1 + \cdots + \mathbb{V}_m$．更确切地说，$\mathbb{V}_1 + \cdots + \mathbb{V}_m = \{v_1 + \cdots + v_m : v_1 \in \mathbb{V}_1, \cdots, v_m \in \mathbb{V}_m\}$．
```

```{admonition} 子空间的和是包含这些子空间的最小子空间
假设 $\mathbb{V}_1, \cdots, \mathbb{V}_m$ 是 $\mathbb{V}$ 的子空间，那么 $\mathbb{V}_1 + \cdots + \mathbb{V}_m$ 是最小的包含 $\mathbb{V}_1, \cdots, \mathbb{V}_m$ 的子空间．
```

```{admonition} 直和（direct sum）
设 $\mathbb{V}_1, \cdots, \mathbb{V}_m$ 是 $\mathbb{V}$ 的子空间．
- 如果 $\mathbb{V}_1 + \cdots + \mathbb{V}_m$ 中的每个元素都能用 $v_1 + \cdots + v_m$（其中各 $v_k \in \mathbb{V}_k$）这种形式唯一地表示出来，则称子空间之和 $\mathbb{V}_1 + \cdots + \mathbb{V}_m$ 为直和．
- 如果 $\mathbb{V}_1 + \cdots + \mathbb{V}_m$ 是直和，那么用 $\mathbb{V}_1 \oplus \cdots \oplus \mathbb{V}_m$ 来表示 $\mathbb{V}_1 + \cdots + \mathbb{V}_m$，其中记号 $\oplus$ 表示此处的和是直和．
```

```{admonition} 直和的条件
假定 $\mathbb{V}_1, \cdots, \mathbb{V}_m$ 是 $\mathbb{V}$．那么 $\mathbb{V}_1 + \cdots + \mathbb{V}_m$ 是直和，当且仅当用 $v_1 + \cdots + v_m$（其中各 $v_k \in \mathbb{V}_k$）表示 $\mathbf{0}$ 的唯一方式是将每个 $v_k$ 都取 $0$.
```

```{admonition} 两个子空间的直和
假定 $\mathbb{U}$ 和 $\mathbb{W}$ 是 $\mathbb{V}$ 的子空间．那么 $\mathbb{U} + \mathbb{W}$ 是直和 $\iff$ $\mathbb{U} \cap \mathbb{W} = \{\mathbf{0}\}$．
```

```{admonition} 偶函数
函数 $f : \mathbb{R} \to \mathbb{R}$ 被称为**偶的**（even），若 $f(-x) = f(x)$ 对所有 $x \in \mathbb{R}$ 成立．
```
```{admonition} 奇函数
函数 $f : \mathbb{R} \to \mathbb{R}$ 被称为**奇的**（odd），若 $f(-x) = -f(x)$ 对所有 $x \in \mathbb{R}$ 成立．
```

令 $\mathbb{V}_e$ 代表 $\mathbb{R}$ 上的实值偶函数构成的集合，令 $\mathbb{V}_o$ 代表 $\mathbb{R}$ 上的实值奇函数构成的集合，可证明：$\mathbb{R} = \mathbb{V}_e \oplus \mathbb{V}_o$．
