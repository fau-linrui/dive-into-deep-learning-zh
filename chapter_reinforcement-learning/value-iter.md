```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(["pytorch"])
#required_libs("setuptools==66", "wheel==0.38.4", "gym==0.21.0")
```

# 值迭代
:label:`sec_valueiter`

本节将讨论机器人在每个状态如何选择最佳动作，以最大化轨迹的*回报*。我们将描述一个称为值迭代（value iteration）的算法，并为一个在冰湖（frozen lake）上行走的模拟机器人实现该算法。

## 随机策略

记作 $\pi(a \mid s)$ 的随机策略（stochastic policy，简称策略）是给定状态 $s \in \mathcal{S}$ 时关于动作 $a \in \mathcal{A}$ 的条件分布，$\pi(a \mid s) \equiv P(a \mid s)$。例如，假设机器人有四个动作 $\mathcal{A}=$ {向左、向下、向右、向上}。对于这样的动作集合 $\mathcal{A}$，在状态 $s \in \mathcal{S}$ 处的策略是一个分类分布，其中四个动作的概率可以是 $[0.4, 0.2, 0.1, 0.3]$；在另一个状态 $s' \in \mathcal{S}$ 处，同样四个动作的概率 $\pi(a \mid s')$ 可以是 $[0.1, 0.1, 0.2, 0.6]$。请注意，对任意状态 $s$ 都应有 $\sum_a \pi(a \mid s) = 1$。确定性策略是随机策略的特例，即分布 $\pi(a \mid s)$ 只对某一个特定动作给出非零概率，例如在我们这个四动作的例子中为 $[1, 0, 0, 0]$。

为使记号不那么繁琐，我们常常用 $\pi(s)$ 来表示这个条件分布，而不写 $\pi(a \mid s)$。

## 价值函数

现在设想机器人从状态 $s_0$ 出发，在每个时刻先从策略中采样一个动作 $a_t \sim \pi(s_t)$，再执行该动作到达下一个状态 $s_{t+1}$。轨迹 $\tau = (s_0, a_0, r_0, s_1, a_1, r_1, \ldots)$ 可能不同，取决于在中间各个时刻采样到哪个具体的动作 $a_t$。我们定义所有这类轨迹的平均*回报* $R(\tau) = \sum_{t=0}^\infty \gamma^t r(s_t, a_t)$
$$V^\pi(s_0) = E_{a_t \sim \pi(s_t)} \Big[ R(\tau) \Big] = E_{a_t \sim \pi(s_t)} \Big[ \sum_{t=0}^\infty \gamma^t r(s_t, a_t) \Big],$$

其中 $s_{t+1} \sim P(s_{t+1} \mid s_t, a_t)$ 是机器人的下一个状态，$r(s_t, a_t)$ 是在时刻 $t$ 于状态 $s_t$ 执行动作 $a_t$ 所获得的即时奖励。这称为策略 $\pi$ 的“价值函数”。简单来说，策略 $\pi$ 下状态 $s_0$ 的价值记作 $V^\pi(s_0)$，它是指机器人从状态 $s_0$ 出发并在每个时刻按策略 $\pi$ 执行动作时，所获得的期望 $\gamma$-折扣*回报*。

接下来我们把轨迹分成两个阶段：（i）第一阶段，即在执行动作 $a_0$ 后对应 $s_0 \to s_1$ 的阶段；（ii）第二阶段，即此后的一段轨迹 $\tau' = (s_1, a_1, r_1, \ldots)$。强化学习中所有算法的关键思想在于：状态 $s_0$ 的价值可以写成第一阶段获得的平均奖励，与对所有可能的下一个状态 $s_1$ 取平均的价值函数之和。这一思想颇为直观，它来自我们的马尔可夫假设：从当前状态出发的平均回报，等于从下一个状态出发的平均回报与转移到下一个状态的平均奖励之和。在数学上，我们把这两个阶段写为

$$V^\pi(s_0) = r(s_0, a_0) + \gamma\ E_{a_0 \sim \pi(s_0)} \Big[ E_{s_1 \sim P(s_1 \mid s_0, a_0)} \Big[ V^\pi(s_1) \Big] \Big].$$
:eqlabel:`eq_dynamic_programming`

这种分解非常强大：它是动态规划原理的基础，而所有强化学习算法都建立在动态规划原理之上。请注意，第二阶段含有两个期望：一个是在第一阶段使用随机策略选择动作 $a_0$ 的期望，另一个是由所选动作得到可能状态 $s_1$ 的期望。我们可以利用马尔可夫决策过程（MDP）中的转移概率，将 :eqref:`eq_dynamic_programming` 写为

$$V^\pi(s) = \sum_{a \in \mathcal{A}} \pi(a \mid s) \Big[ r(s,  a) + \gamma\  \sum_{s' \in \mathcal{S}} P(s' \mid s, a) V^\pi(s') \Big];\ \textrm{for all } s \in \mathcal{S}.$$
:eqlabel:`eq_dynamic_programming_val`

这里要注意的一个重要之处是：上述恒等式对所有状态 $s \in \mathcal{S}$ 都成立，因为我们可以考虑从该状态出发的任意轨迹，并把该轨迹分解为两个阶段。

## 动作价值函数

在实现中，通常有必要维护一个称为“动作价值”函数（action-value function）的量，它与价值函数密切相关。动作价值函数的定义是：从 $s_0$ 出发、但第一阶段动作固定为 $a_0$ 的轨迹的平均*回报*

$$Q^\pi(s_0, a_0) = r(s_0, a_0) + E_{a_t \sim \pi(s_t)} \Big[ \sum_{t=1}^\infty \gamma^t r(s_t, a_t) \Big],$$

注意，期望内部的求和从 $t=1,\ldots, \infty$ 开始，因为此情形下第一阶段的奖励是固定的。我们同样可以把轨迹分解为两部分，并写为

$$Q^\pi(s, a) = r(s, a) + \gamma \sum_{s' \in \mathcal{S}} P(s' \mid s, a) \sum_{a' \in \mathcal{A}} \pi(a' \mid s')\ Q^\pi(s', a');\ \textrm{ for all } s \in \mathcal{S}, a \in \mathcal{A}.$$
:eqlabel:`eq_dynamic_programming_q`

这一版本是 :eqref:`eq_dynamic_programming_val` 在动作价值函数上的类似形式。

## 最优随机策略

价值函数和动作价值函数都依赖于机器人所选择的策略。接下来我们考虑能达到最大平均*回报*的“最优策略”
$$\pi^* = \underset{\pi}{\mathrm{argmax}} V^\pi(s_0).$$

在机器人可能采取的所有随机策略中，最优策略 $\pi^*$ 对从状态 $s_0$ 出发的轨迹取得最大的平均折扣*回报*。我们把最优策略的价值函数和动作价值函数分别记为 $V^* \equiv V^{\pi^*}$ 和 $Q^* \equiv Q^{\pi^*}$。

我们来观察确定性策略的情形：在任意给定状态下，该策略下只有唯一一个可能的动作。这给出

$$\pi^*(s) = \underset{a \in \mathcal{A}}{\mathrm{argmax}} \Big[ r(s, a) + \gamma \sum_{s' \in \mathcal{S}} P(s' \mid s, a)\ V^*(s') \Big].$$

一个便于记住这一点的助记是：在状态 $s$ 处的最优动作（对确定性策略而言），是使第一阶段奖励 $r(s, a)$ 与从下一个状态 $s'$ 出发的轨迹的平均*回报*之和最大的动作，其中该平均是对第二阶段所有可能的下一个状态 $s'$ 求的。

## 动态规划原理

上一节在 :eqref:`eq_dynamic_programming` 或 :eqref:`eq_dynamic_programming_q` 中的推导，可以分别转化为计算最优价值函数 $V^*$ 或动作价值函数 $Q^*$ 的算法。观察

$$ V^*(s) = \sum_{a \in \mathcal{A}} \pi^*(a \mid s) \Big[ r(s,  a) + \gamma\  \sum_{s' \in \mathcal{S}} P(s' \mid s, a) V^*(s') \Big];\ \textrm{for all } s \in \mathcal{S}.$$

对于确定性最优策略 $\pi^*$，由于在状态 $s$ 只能执行一个动作，我们也可以写为

$$V^*(s) = \mathrm{argmax}_{a \in \mathcal{A}} \Big\{ r(s,a) + \gamma \sum_{s' \in \mathcal{S}} P(s' \mid s, a) V^*(s') \Big\}$$

对所有状态 $s \in \mathcal{S}$ 成立。这一恒等式称为“动态规划原理” :cite:`BellmanDPPaper,BellmanDPBook`。它由 Richard Bellman 在 20 世纪 50 年代提出，我们可以把它记为“最优轨迹的剩余部分也是最优的”。

## 值迭代

我们可以把动态规划原理转化为一个寻找最优价值函数的算法，称为值迭代。值迭代的关键思想是把上述恒等式看作一组约束，把不同状态 $s \in \mathcal{S}$ 处的 $V^*(s)$ 联系在一起。我们把价值函数初始化为任意值 $V_0(s)$，对所有状态 $s \in \mathcal{S}$ 都是如此。在第 $k^{\textrm{th}}$ 次迭代时，值迭代算法按如下方式更新价值函数

$$V_{k+1}(s) = \max_{a \in \mathcal{A}} \Big\{ r(s,  a) + \gamma\  \sum_{s' \in \mathcal{S}} P(s' \mid s, a) V_k(s') \Big\};\ \textrm{for all } s \in \mathcal{S}.$$

事实证明，当 $k \to \infty$ 时，值迭代算法估计的价值函数会收敛到最优价值函数，而与初始化 $V_0$ 无关，
$$V^*(s) = \lim_{k \to \infty} V_k(s);\ \textrm{for all states } s \in \mathcal{S}.$$

同样的值迭代算法也可以用动作价值函数等价地写成
$$Q_{k+1}(s, a) = r(s, a) + \gamma \max_{a' \in \mathcal{A}} \sum_{s' \in \mathcal{S}} P(s' \mid s, a) Q_k (s', a');\ \textrm{ for all } s \in \mathcal{S}, a \in \mathcal{A}.$$

此时，我们把 $Q_0(s, a)$ 初始化为任意值，对所有 $s \in \mathcal{S}$ 和 $a \in \mathcal{A}$ 都是如此。同样有 $Q^*(s, a) = \lim_{k \to \infty} Q_k(s, a)$ 对所有 $s \in \mathcal{S}$ 和 $a \in \mathcal{A}$ 成立。

## 策略评估

值迭代使我们能够计算最优价值函数，即最优确定性策略 $\pi^*$ 的 $V^{\pi^*}$。我们也可以用类似的迭代更新来计算与任何其他（可能是随机的）策略 $\pi$ 相关联的价值函数。我们同样把 $V^\pi_0(s)$ 初始化为任意值，对所有状态 $s \in \mathcal{S}$ 都是如此，并在第 $k^{\textrm{th}}$ 次迭代时执行更新

$$    V^\pi_{k+1}(s) = \sum_{a \in \mathcal{A}} \pi(a \mid s) \Big[ r(s,  a) + \gamma\  \sum_{s' \in \mathcal{S}} P(s' \mid s, a) V^\pi_k(s') \Big];\ \textrm{for all } s \in \mathcal{S}.$$

该算法称为策略评估，它可用于在给定策略的情况下计算价值函数。同样，事实证明当 $k \to \infty$ 时，这些更新会收敛到正确的价值函数，而与初始化 $V_0$ 无关，

$$V^\pi(s) = \lim_{k \to \infty} V^\pi_k(s);\ \textrm{for all states } s \in \mathcal{S}.$$

计算策略 $\pi$ 的动作价值函数 $Q^\pi(s, a)$ 的算法与此类似。

## 值迭代的实现
:label:`subsec_valueitercode`
下面我们展示如何针对 [Open AI Gym](https://gym.openai.com) 中一个称为 FrozenLake 的导航问题实现值迭代。首先需要像下面的代码那样设置环境。

```{.python .input}
%%tab all

%matplotlib inline
import numpy as np
import random
from d2l import torch as d2l

seed = 0  # Random number generator seed
gamma = 0.95  # Discount factor
num_iters = 10  # Number of iterations
random.seed(seed)  # Set the random seed to ensure results can be reproduced
np.random.seed(seed)

# Now set up the environment
env_info = d2l.make_env('FrozenLake-v1', seed=seed)
```

在 FrozenLake 环境中，机器人在一个 $4 \times 4$ 网格上移动（这些格子就是状态），动作有“上”（$\uparrow$）、“下”（$\rightarrow$）、“左”（$\leftarrow$）和“右”（$\rightarrow$）。环境中有若干洞（H）格子和冰面（F）格子，以及一个目标格子（G），而所有这些对机器人来说都是未知的。为使问题简单，我们假设机器人的动作是可靠的，即对所有 $s \in \mathcal{S}, a \in \mathcal{A}$ 都有 $P(s' \mid s, a) = 1$。如果机器人到达目标，本次试验就结束，机器人无论执行什么动作都获得奖励 $1$；在其他任何状态下，所有动作的奖励均为 $0$。机器人的目标是学习一个策略，从给定的起始位置（S）（即 $s_0$）到达目标位置（G），以最大化*回报*。

下面的函数实现了值迭代，其中`env_info`包含与 MDP 和环境相关的信息，`gamma`是折扣因子：

```{.python .input}
%%tab all

def value_iteration(env_info, gamma, num_iters):
    env_desc = env_info['desc']  # 2D array shows what each item means
    prob_idx = env_info['trans_prob_idx']
    nextstate_idx = env_info['nextstate_idx']
    reward_idx = env_info['reward_idx']
    num_states = env_info['num_states']
    num_actions = env_info['num_actions']
    mdp = env_info['mdp']

    V  = np.zeros((num_iters + 1, num_states))
    Q  = np.zeros((num_iters + 1, num_states, num_actions))
    pi = np.zeros((num_iters + 1, num_states))

    for k in range(1, num_iters + 1):
        for s in range(num_states):
            for a in range(num_actions):
                # Calculate \sum_{s'} p(s'\mid s,a) [r + \gamma v_k(s')]
                for pxrds in mdp[(s,a)]:
                    # mdp(s,a): [(p1,next1,r1,d1),(p2,next2,r2,d2),..]
                    pr = pxrds[prob_idx]  # p(s'\mid s,a)
                    nextstate = pxrds[nextstate_idx]  # Next state
                    reward = pxrds[reward_idx]  # Reward
                    Q[k,s,a] += pr * (reward + gamma * V[k - 1, nextstate])
            # Record max value and max action
            V[k,s] = np.max(Q[k,s,:])
            pi[k,s] = np.argmax(Q[k,s,:])
    d2l.show_value_function_progress(env_desc, V[:-1], pi[:-1])

value_iteration(env_info=env_info, gamma=gamma, num_iters=num_iters)
```

上图展示了策略（箭头表示动作）和价值函数（颜色的变化表明价值函数随时间如何从深色所示的初始值变化到浅色所示的最优值）。可以看到，值迭代在 10 次迭代后找到了最优价值函数，并且只要不是 H 格子，从任何状态出发都可以到达目标状态（G）。该实现的另一个有趣之处是，除了找到最优价值函数外，我们还自动找到了与该价值函数对应的最优策略 $\pi^*$。


## 小结
值迭代算法背后的主要思想是利用动态规划原理，求从给定状态出发可获得的最优平均回报。请注意，实现值迭代算法要求我们完全了解马尔可夫决策过程（MDP），例如转移函数和奖励函数。


## 练习

1. 试着把网格大小增加到 $8 \times 8$。与 $4 \times 4$ 网格相比，找到最优价值函数需要多少次迭代？
1. 值迭代算法的计算复杂度是多少？
1. 当 $\gamma$（即上面代码中的“gamma”）分别等于 $0$、$0.5$ 和 $1$ 时，再次运行值迭代算法并分析其结果。
1. $\gamma$ 的取值如何影响值迭代收敛所需的迭代次数？当 $\gamma=1$ 时会发生什么？

:begin_tab:`pytorch`
[讨论](https://discuss.d2l.ai/t/12005)
:end_tab:
