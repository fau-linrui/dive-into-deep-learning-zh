# 符号
:label:`chap_notation`

在整本书中，我们都遵循以下符号约定。请注意，其中一些符号是占位符，而另一些则指代特定的对象。作为一条通用的经验法则，不定冠词“a”通常表示该符号是一个占位符，格式相似的符号可以表示同类型的其他对象。例如，“$x$：标量”表示小写字母一般代表标量值，而“$\mathbb{Z}$：整数集”则特指符号$\mathbb{Z}$。



## 数值对象

* $x$：标量
* $\mathbf{x}$：向量
* $\mathbf{X}$：矩阵
* $\mathsf{X}$：一般张量
* $\mathbf{I}$：单位矩阵（某个给定维度），即所有对角元素为$1$、所有非对角元素为$0$的方阵
* $x_i$、$[\mathbf{x}]_i$：向量$\mathbf{x}$的第$i^\textrm{th}$个元素
* $x_{ij}$、$x_{i,j}$、$[\mathbf{X}]_{ij}$、$[\mathbf{X}]_{i,j}$：矩阵$\mathbf{X}$中第$i$行第$j$列的元素。



## 集合论


* $\mathcal{X}$：集合
* $\mathbb{Z}$：整数集
* $\mathbb{Z}^+$：正整数集
* $\mathbb{R}$：实数集
* $\mathbb{R}^n$：$n$维实数向量的集合
* $\mathbb{R}^{a\times b}$：具有$a$行$b$列的实数矩阵的集合
* $|\mathcal{X}|$：集合$\mathcal{X}$的基数（元素个数）
* $\mathcal{A}\cup\mathcal{B}$：集合$\mathcal{A}$与$\mathcal{B}$的并集
* $\mathcal{A}\cap\mathcal{B}$：集合$\mathcal{A}$与$\mathcal{B}$的交集
* $\mathcal{A}\setminus\mathcal{B}$：从$\mathcal{A}$中减去$\mathcal{B}$的集合差（只包含$\mathcal{A}$中不属于$\mathcal{B}$的那些元素）



## 函数与运算符


* $f(\cdot)$：函数
* $\log(\cdot)$：自然对数（底为$e$）
* $\log_2(\cdot)$：以$2$为底的对数
* $\exp(\cdot)$：指数函数
* $\mathbf{1}(\cdot)$：指示函数；若布尔参数为真则取值为$1$，否则为$0$
* $\mathbf{1}_{\mathcal{X}}(z)$：集合隶属指示函数；若元素$z$属于集合$\mathcal{X}$则取值为$1$，否则为$0$
* $\mathbf{(\cdot)}^\top$：向量或矩阵的转置
* $\mathbf{X}^{-1}$：矩阵$\mathbf{X}$的逆
* $\odot$：哈达玛积（Hadamard product，即按元素乘积）
* $[\cdot, \cdot]$：连结
* $\|\cdot\|_p$：$\ell_p$范数
* $\|\cdot\|$：$\ell_2$范数
* $\langle \mathbf{x}, \mathbf{y} \rangle$：向量$\mathbf{x}$与$\mathbf{y}$的内（点）积
* $\sum$：对一组元素求和
* $\prod$：对一组元素求积
* $\stackrel{\textrm{def}}{=}$：定义为左侧符号的等式



## 微积分

* $\frac{dy}{dx}$：$y$关于$x$的导数
* $\frac{\partial y}{\partial x}$：$y$关于$x$的偏导数
* $\nabla_{\mathbf{x}} y$：$y$关于$\mathbf{x}$的梯度
* $\int_a^b f(x) \;dx$：$f$关于$x$从$a$到$b$的定积分
* $\int f(x) \;dx$：$f$关于$x$的不定积分



## 概率与信息论

* $X$：随机变量
* $P$：概率分布
* $X \sim P$：随机变量$X$服从分布$P$
* $P(X=x)$：随机变量$X$取值$x$这一事件的概率
* $P(X \mid Y)$：给定$Y$时$X$的条件概率分布
* $p(\cdot)$：与分布$P$相关的概率密度函数（PDF）
* ${E}[X]$：随机变量$X$的期望
* $X \perp Y$：随机变量$X$与$Y$相互独立
* $X \perp Y \mid Z$：给定$Z$时随机变量$X$与$Y$条件独立
* $\sigma_X$：随机变量$X$的标准差
* $\textrm{Var}(X)$：随机变量$X$的方差，等于$\sigma^2_X$
* $\textrm{Cov}(X, Y)$：随机变量$X$与$Y$的协方差
* $\rho(X, Y)$：$X$与$Y$之间的皮尔逊相关系数，等于$\frac{\textrm{Cov}(X, Y)}{\sigma_X \sigma_Y}$
* $H(X)$：随机变量$X$的熵
* $D_{\textrm{KL}}(P\|Q)$：从分布$Q$到分布$P$的KL散度（或相对熵）



[讨论](https://discuss.d2l.ai/t/25)
