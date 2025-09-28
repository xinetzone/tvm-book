# 线性映射定义

```{admonition} 定义：线性映射（linear map）
从 $\mathbb{V}$ 到 $\mathbb{W}$ 的线性映射是满足下列性质的函数 $\mathcal{T} : \mathbb{V} \to \mathbb{W}$．

- 可加性（additivity）：对于所有 $u, v \in \mathbb{V}$，均有 $\mathcal{T}(u + v) = \mathcal{T}(u) + \mathcal{T}(v)$．
- 齐次性（homogeneity）：对于所有 $\lambda \in \mathbb{F}$ 和所有 $v \in \mathbb{V}$，均有 $\mathcal{T}(\lambda v) = \lambda \mathcal{T}(v)$．

有些数学家使用术语“线性变换”（linear transformation），这和线性映射同义．
```

```{admonition} 记号：$\mathcal{L}(\mathbb{V}, \mathbb{W})$、$\mathcal{L}(\mathbb{V})$
从 $\mathbb{V}$ 到 $\mathbb{W}$ 的全体线性映射构成的集合记作 $\mathcal{L}(\mathbb{V}, \mathbb{W})$．

从 $\mathbb{V}$ 到 $\mathbb{V}$ 的全体线性映射构成的集合记作 $\mathcal{L}(\mathbb{V})$．换言之，$\mathcal{L} (\mathbb{V}) = \mathcal{L} (\mathbb{V}, \mathbb{V})$．
```

```{admonition} 线性映射引理（linear map lemma）
假定 $v_1, \cdots, v_n$ 是 $\mathbb{V}$ 的基且 $w_1, \cdots , w_n \in \mathbb{W}$．那么存在唯一的线性映射 $\mathcal{T}: \mathbb{V} \to \mathbb{W}$ 使得对每个 $k = 1, \cdots , n$ 都有 $\mathcal{T}(v_k) = w_k$．
```

```{admonition} 定义：$\mathcal{L} (\mathbb{V}, \mathbb{W})$ 上的加法和标量乘法(addition and scalar multiplication on $\mathcal{L} (\mathbb{V}, \mathbb{W})$)
假设 $\mathcal{S}, \mathcal{T} \in \mathcal{L} (\mathbb{V}, \mathbb{W})$ 且 $\lambda \in \mathbb{F}$．和 $\mathcal{S+T}$ 与积 $\lambda \mathcal{T}$ 都是从 $\mathbb{V}$  到 $\mathbb{W}$ 的线性映射，分别定义如下：对于所有 $v \in \mathbb{V}$，

$$
(\mathcal{S+T})(v) = \mathcal{S}v + \mathcal{T}v, \quad (\lambda \mathcal{T})(v) = \lambda (\mathcal{T}v).
$$
```

```{admonition} \mathcal{L}(\mathbb{V}, \mathbb{W}) 是向量空间
有了上面定义的加法和标量乘法，$\mathcal{L}(\mathbb{V}, \mathbb{W})$ 就是向量空间．
```

```{admonition} 定义：线性映射的乘积（product of linear maps）
如果 $\mathcal{T} \in \mathcal{L}(\mathbb{V}, \mathbb{W})$ 且 $\mathcal{S} \in \mathcal{L}(\mathbb{V}, \mathbb{W})$，那么乘积 $\mathcal{ST} \in \mathcal{L}(\mathbb{V}, \mathbb{W})$ 就定义为：对于所有 $u \in \mathbb{U}$，

$$
(\mathcal{ST})(u) = \mathcal{S}(\mathcal{T}u).
$$
```

```{admonition} 线性映射乘积的代数性质
- **可结合性**（associativity）：对于任意使乘积有意义的线性映射 $\mathcal{T}_1,\mathcal{T}_2,\mathcal{T}_3$（意即 $\mathcal{T}_3$ 映射到 $\mathcal{T}_2$ 的定义空间中，$\mathcal{T}_2$ 映射到 $\mathcal{T}_1$ 的定义空间中），有 $\mathcal{(T_1T_2)T_3} = \mathcal{T_1}(\mathcal{T_2T_3})$．
- **恒等元**（identity）：对于任意 $\mathcal{T} \in \mathcal{L}(\mathbb{V}, \mathbb{W})$，有 $\mathcal{TI} = \mathcal{IT} = \mathcal{T}$．这里第一个 $\mathcal{I}$ 是 $\mathbb{V}$ 上的恒等算子，而第二个 $\mathcal{I}$ 是 $\mathbb{W}$ 上的恒等算子．
- **分配性质**（distributive properties）：
对于任意 $\mathcal{T, T_1, T_2} \in \mathcal{L}(\mathbb{U}, \mathbb{V})$ 和 $\mathcal{S, S_1, S_2} \in \mathcal{L}(\mathbb{V}, \mathbb{W})$，有 $\mathcal{(S_1 + S_2)T} = \mathcal{S_1T} + \mathcal{S_2T}$ 且 $\mathcal{S(T_1+T2)=ST_1 + ST_2}$．
```

```{admonition} 线性映射将 $0$ 映射为 $0$
假设 $\mathcal{T} \in \mathcal{L}(\mathbb{V}, \mathbb{W})$ 是由 𝑉 到 𝑊 的线性映射．那么 $\mathcal{T}(0) = 0$
```
