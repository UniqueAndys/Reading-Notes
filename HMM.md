# 隐马尔科夫模型

## 隐马尔科夫模型的基本概念
1. （隐马尔科夫模型） 隐马尔科夫模型是关于时序的概率模型，描述由一个隐藏的马尔科夫链随机生成不可观测的状态随机序列，再由各个状态生成一个观测而产生观测随机序列的过程。隐藏的马尔科夫链随机生成的状态的序列，称为状态序列（state sequence）；每个状态生成一个观测，而由此产生的观测的随机序列，称为观测序列（observation sequence）。


2. 隐马尔科夫模型由初始概率分布$\pi$、状态转移概率分布$A$ 以及观测概率分布$B$ 确定，定义如下：

- 设 $ Q = {q_1, q_2, \cdots , q_N} $ 是所有可能的状态的集合，$ V = {v_1, v_2, \cdots, v_M}$ 是所有可能的观测的集合。其中，$N$ 是可能的状态数，$M$ 是可能的观测数。


- $ I = (i_1, i_2, \cdots, i_T) $ 是长度为 $T$ 的状态序列，$ O = (o_1, o_2, \cdots, o_T) $ 是对应的观测序列。


- $ A = [a_{ij}]{}_{N*N} $ 是状态转移概率矩阵，其中，$a_{ij} = P(i_{t+1}=q_j|i_t=q_i), i=1,2,\cdots,N; j=1,2,\cdots,N$ 是在时刻 $t$ 处于状态 $q_i$ 的条件下在时刻 $t+1$ 转移到状态 $q_j$ 的概率。


- $B$ 是观测概率矩阵，$B=[b_j(k)]{}_{N*M}$，其中，$b_j(k)=P(o_t=v_k|i_t=q_j), k=1,2,\cdots,M;j=1,2,\cdots,N$ 是在时刻 $t$ 处于状态 $q_j$ 的条件下生成观测 $v_k$ 的概率。


- $\pi = (\pi_i) $ 是初始状态概率向量，其中，$\pi_i=P(i_1=q_i), i=1,2,\cdots,N$是时刻 $t = 1$ 处于状态 $q_i$ 的概率。


隐马尔科夫模型$\lambda$ 可以用三元符号表示，即 $\lambda=(A,B,\pi)$。$A,B,\pi$称为隐马尔科夫模型的三要素。状态转移概率矩阵$A$ 与初始状态概率向量$\pi$ 确定了隐藏的马尔科夫链，生成不可观测的状态序列。观测概率矩阵$B$ 确定了如何从状态生成观测，与状态序列综合确定了如何产生观测序列。


3. 从定义可知，隐马尔科夫模型作了两个基本假设：
- 齐次马尔可夫性假设，即假设隐藏的马尔科夫链在任意时刻 $t$ 的状态只依赖于其前一时刻的状态，与其他时刻的状态及观测无关，也与时刻 $t$ 无关。
$$P(i_t|i_{t-1},o_{t-1},\cdots,i_1,o_1)=P(i_t|i_{t-1}), t=1,2,\cdots,T$$

- 观测独立性假设，即假设任意时刻的观测只依赖于该时刻的马尔科夫链的状态，与其他观测及状态无关。
$$P(o_t|i_T,o_T,i_{T-1},o_{T-1},\cdots,i_{t+1},o_{t+1},i_t,i_{t-1},o_{t-1},\cdots,i_1,o_1)=P(o_t|i_t)$$


4. （观测序列的生成）
输入：隐马尔科夫模型 $\lambda=(A,B,\pi)$，观测序列长度 $T$ ;
输出：观测序列 $O=(o_1,o_2,\cdots,o_T)$.
(1) 按照初始状态分布 $\pi$ 产生状态 $i_1$
(2) 令 $t=1$
(3) 按照状态 $i_t$ 的观测概率分布 $b_{i_t}(k)$ 生成 $o_t$，如果 $t=T$，则终止
(4) 按照状态 $i_t$ 的状态转移概率分布 ${a_{i_ti_{t+1}}}$ 产生状态 $i_{t+1}$,  $i_{t+1}=1,2,\cdots,N$
(5) 令 $t=t+1$，转步(3)


5. 隐马尔科夫模型的3个基本问题
- 概率计算问题。给定模型 $\lambda=(A,B,\pi)$ 和观测序列 $O=(o_1,o_2,\cdots,o_T)$，计算在模型 $\lambda$ 下观测序列 $O$ 出现的概率 $P(O|\lambda)$。
- 学习问题。已知观测序列 $O=(o_1,o_2,\cdots,o_T)$，估计模型 $\lambda=(A,B,\pi)$参数，使得在该模型下观测序列概率 $P(O|\lambda)$ 最大，即用极大似然估计的方法估计参数。
- 预测问题，也称为解码（decoding）问题。已知模型 $\lambda=(A,B,\pi)$ 和观测序列 $O=(o_1,o_2,\cdots,o_T)$，求对给定观测序列条件概率 $P(I|O)$ 最大的状态序列 $I=(i_1,i_2,\cdots,i_T)$。即给定观测序列，求最有可能的对应的状态序列。


## 概率计算方法
给定模型 $\lambda=(A,B,\pi)$ 和观测序列 $O=(o_1,o_2,\cdots,o_T)$，计算在模型 $\lambda$ 下观测序列 $O$ 出现的概率 $P(O|\lambda)$。

1. 直接计算法，计算量很大，是 $O(TN^T)$ 阶的。


2. 前向算法

（前向概率）给定隐马尔科夫模型 $\lambda$，定义到时刻 $t$ 部分观测序列为 $o_1,o_2,\cdots,o_t$且状态为 $q_i$ 的概率为前向概率，记作
$$\alpha_t(i)=P(o_1,o_2,\cdots,o_t,i_t=q_i|\lambda)$$ 可以递推地求得前向概率 $\alpha_t(i)$ 及观测序列概率 $P(O|\lambda)$。

（观测序列概率的前向算法）
输入：隐马尔科夫模型 $\lambda$，观测序列 $O$；
输出：观测序列概率 $P(O|\lambda)$。
(1) 初值 $$\alpha_1(i)=\pi_ib_i(o_1), i=1,2,\cdots,N$$
(2) 递推 对 $t=1,2,\cdots,T-1$，$$\alpha_{t+1}(i)=\bigg[\sum_{j=1}^N\alpha_t(j)a_{ji}\bigg]b_i(o_{t+1}), i=1,2,\cdots,N$$
(3) 终止 $$P(O|\lambda)=\sum_{i=1}^N\alpha_T(i)$$

3. 后向算法

（后向概率）给定隐马尔科夫模型 $\lambda$，定义在时刻 $t$ 状态为 $q_i$ 的条件下，从 $t+1$ 到 $T$ 的部分观测序列为 $o_{t+1},o_{t+2},\cdots,o_T$ 的概率为后向概率，记作 $$\beta_t(i)=P(o_{t+1},o_{t+2},\cdots,o_T|i_t=q_i,\lambda)$$ 可以用递推的方法求得后向概率 $\beta_t(i)$ 及观测序列概率 $P(O|\lambda)$。

（观测序列概率的后向算法）
输入：隐马尔科夫模型 $\lambda$，观测序列 $O$；
输出：观测序列概率 $P(O|\lambda)$。
(1) $$\beta_T(i)=1, i=1,2,\cdots,N$$
(2) 对 $t=T-1,T-2,\cdots,1$ $$\beta_t(i)=\sum_{j=1}^Na_{ij}b_j(o_{t+1})\beta_{t+1}(j), i=1,2,\cdots,N$$
(3) $$P(O|\lambda)=\sum_{i=1}^N\pi_ib_i(o_1)\beta_1(i)$$

4. 一些概率与期望值的计算
- 给定模型 $\lambda$ 和观测 $O$，在时刻 $t$ 处于状态 $q_i$ 的概率。记 $$\gamma_t(i)=P(i_t=q_i|O,\lambda)=\frac{P(i_t=q_i,O|\lambda)}{P(O|\lambda)}=\frac{\alpha_t(i)\beta_t(i)}{\sum_{j=1}^N\alpha_t(j)\beta_t(j)}$$

- 给定模型 $\lambda$ 和观测 $O$，在时刻 $t$ 处于状态 $q_i$ 且在时刻 $t+1$ 处于状态 $q_j$ 的概率。记 $$\zeta_t(i,j)=P(i_t=q_i,i_{t+1}=q_j|O,\lambda)=\frac{P(i_t=q_i,i_{t+1}=q_j,O|\lambda)}{P(O|\lambda)}$$ 而 $$P(i_t=q_i,i_{t+1}=q_j,O|\lambda)=\alpha_t(i)a_{ij}b_j(o_{t+1})\beta_{t+1}(j)$$ 所以 $$\zeta_t(i,j)=\frac{\alpha_t(i)a_{ij}b_j(o_{t+1})\beta_{t+1}(j)}{\sum_{i=1}^N\sum_{j=1}^N\alpha_t(i)a_{ij}b_j(o_{t+1})\beta_{t+1}(j)}$$


## 学习算法
已知观测序列 $O=(o_1,o_2,\cdots,o_T)$，估计模型 $\lambda=(A,B,\pi)$参数，使得在该模型下观测序列概率 $P(O|\lambda)$ 最大，即用极大似然估计的方法估计参数。

1. 监督学习算法
假设已给训练数据包含 $S$ 个长度相同的观测序列和对应的状态序列 $\{(O_1,I_1),(O_2,I_2),\cdots,(O_s,I_s)\}$，那么可以利用极大似然估计法来估计隐马尔科夫模型的参数。
- 转移概率 $a_{ij}$ 的估计
设样本中时刻 $t$ 处于状态 $i$ 时刻 $t+1$ 转移到状态 $j$ 的频数为 $A_{ij}$，那么状态转移概率 $a_{ij}$ 的估计是 $$\hat{a}_{ij}=\frac{A_{ij}}{\sum_{j=1}^NA_{ij}}, i=1,2,\cdots,N; j=1,2,\cdots,N$$
- 观测概率 $b_j(k)$ 的估计
$$\hat{b}_j(k)=\frac{B_{jk}}{\sum_{k=1}^MB_{jk}}, j=1,2,\cdots,N; k=1,2,\cdots,M$$
- 初始状态概率 $\pi_i$ 的估计 $\hat{\pi}_i$ 为S个样本中初始状态为 $q_i$ 的频率

2. 非监督学习算法

（Baum-Welch算法）
输入：观测数据 $O=(o_1,o_2,\cdots,o_T)$；
输出：隐马尔科夫模型参数。
(1) 初始化
对 $n=0$，选取 $a_{ij}^{(0)}, b_j(k)^{(0)}, \pi_i^{(0)}$，得到模型 $\lambda^{(0)}=(A^{(0)},B^{(0)},\pi^{(0)})$
(2) 递推。对 $n=0,1,2,\cdots,$ $$a_{ij}^{(n+1)}=\frac{\sum_{t=1}^{T-1}\zeta_t(i,j)}{\sum_{t=1}^{T-1}\gamma_t(i)}$$ $$b_j(k)^{(n+1)}=\frac{\sum_{t=1,o_t=v_k}^T\gamma_t(j)}{\sum_{t=1}^T\gamma_t(j)}$$ $$\pi_i^{(n+1)}=\gamma_1(i)$$ 右端各值按观测 $O=(o_1,o_2,\cdots,o_T)$ 和模型 $\lambda^{(n)}=(A^{(n)},B^{(n)},\pi^{(n)})$ 计算。
(3) 终止。得到模型参数 $\lambda^{(n+1)}=(A^{(n+1)},B^{(n+1)},\pi^{(n+1)})$。


## 预测算法
已知模型 $\lambda=(A,B,\pi)$ 和观测序列 $O=(o_1,o_2,\cdots,o_T)$，求对给定观测序列条件概率 $P(I|O)$ 最大的状态序列 $I=(i_1,i_2,\cdots,i_T)$。即给定观测序列，求最有可能的对应的状态序列。

1. 近似算法
近似算法的想法是，在每个时刻 $t$ 选择在该时刻最有可能出现的状态 $i_t^\ast$，从而得到一个状态序列 $I^\ast=(i_1^\ast,i_2^\ast,\cdots,i_T^\ast)$，将它作为预测的结果：$$i_t^\ast=\arg\max_{1\le i\le N}[\gamma_t(i)], t=1,2,\cdots,T$$


2. 维特比算法
维特比算法实际是用动态规划解隐马尔科夫模型预测问题，即用动态规划求概率最大路径（最优路径）。
定义在时刻 $t$ 状态为 $i$ 的所有单个路径 $(i_1,i_2,\cdots,i_t)$ 中概率最大值为 $$\delta_t(i)=\max_{i_1,i_2,\cdots,i_{t-1}}P(i_t=i,i_{t-1},\cdots,i_1,o_t,\cdots,o_1|\lambda), i=1,1,\cdots,N$$
由定义可得变量 $$\delta_{t+1}(i)=\max_{i_1,i_2,\cdots,i_t}P(i_{t+1}=i,i_t,\cdots,i_1,o_{t+1},\cdots,o_1|\lambda)$$ $$=\max_{1 \le j \le N}[\delta_t(j)a_{ji}]b_i(o_{t+1}), i=1,2,\cdots,N$$
定义在时刻 $t$ 状态为 $i$ 的所有单个路径 $(i_1,i_2,\cdots,i_{t-1},i)$ 中概率最大的路径的第 $t-1$ 个结点为 $$\psi_t(i)=\arg\max_{1 \le j \le N}[\delta_{t-1}(j)a_{ji}], i=1,2,\cdots,N$$
（维特比算法）
输入：模型 $\lambda=(A,B,\pi)$ 和观测 $O=(o_1,o_2,\cdots,o_T)$；
输出：最优路径 $I^\ast=(i_1^\ast,i_2^\ast,\cdots,i_T^\ast)$。
(1) 初始化
$$\delta_1(i)=\pi_ib_i(o_1), i=1,2,\cdots,N$$ $$\psi_1(i)=0, i=1,2,\cdots,N$$
(2) 递推。对 $t=2,3,\cdots,T$
$$\delta_t(i)=\max_{1\le j\le N}[\delta_{t-1}(j)a_{ji}]b_i(o_t), i=1,2,\cdots, N$$ $$\psi_t(i)=\arg\max_{1\le j\le N}[\delta_{t-1}(j)a_{ji}], i=1,2,,\cdots,N$$
(3) 终止
$$P^\ast=\max_{1\le i\le N}\delta_T(i)$$ $$i_T^\ast=\arg\max_{1\le i\le N}[\delta_T(i)]$$
(4) 最优路径回溯。对 $t=T-1,T-2,\cdots,1$
$$i_t^\ast=\psi_{t+1}(i_{t+1}^\ast)$$ 求得最优路径 $I^\ast=(i_1^\ast,i_2^\ast,\cdots,i_T^\ast)$。
