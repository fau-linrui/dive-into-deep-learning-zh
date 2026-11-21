```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(["pytorch"])
#required_libs("setuptools==66", "wheel==0.38.4", "gym==0.21.0")
```

# Q 学习
:label:`sec_qlearning`

上一节我们讨论了值迭代算法，它需要访问完整的马尔可夫决策过程（MDP），例如转移函数和奖励函数。本节我们将介绍 Q 学习（Q-learning） :cite:`Watkins.Dayan.1992`，这是一种无需了解 MDP 也能学习价值函数的算法。该算法体现了强化学习背后的核心思想：它使机器人能够自行获取数据。
<!-- ，而不是依赖专家。 -->

## Q 学习算法

回想一下，:ref:`sec_valueiter` 中针对动作价值函数的值迭代对应如下更新

$$Q_{k+1}(s, a) = r(s, a) + \gamma \sum_{s' \in \mathcal{S}} P(s' \mid s, a) \max_{a' \in \mathcal{A}} Q_k (s', a'); \ \textrm{for all } s \in \mathcal{S} \textrm{ and } a \in \mathcal{A}.$$

如我们所讨论的，实现该算法需要知道 MDP，特别是转移函数 $P(s' \mid s, a)$。Q 学习的关键思想是把上式中对所有 $s' \in \mathcal{S}$ 的求和替换为对机器人访问过的状态求和。这样我们就不必知道转移函数。

## Q 学习背后的优化问题

假设机器人使用策略 $\pi_e(a \mid s)$ 采取动作。与上一章一样，它收集一个包含 $n$ 条轨迹、每条轨迹有 $T$ 个时间步的数据集 $\{ (s_t^i, a_t^i)_{t=0,\ldots,T-1}\}_{i=1,\ldots, n}$。回想一下，值迭代实际上是一组约束，把不同状态和动作的动作价值 $Q^*(s, a)$ 联系在一起。我们可以用机器人使用 $\pi_e$ 收集到的数据实现值迭代的一个近似版本：

$$\hat{Q} = \min_Q \underbrace{\frac{1}{nT} \sum_{i=1}^n \sum_{t=0}^{T-1} (Q(s_t^i, a_t^i) - r(s_t^i, a_t^i) - \gamma \max_{a'} Q(s_{t+1}^i, a'))^2}_{\stackrel{\textrm{def}}{=} \ell(Q)}.$$
:eqlabel:`q_learning_optimization_problem`

让我们先观察该表达式与上面的值迭代之间的异同。如果机器人的策略 $\pi_e$ 等于最优策略 $\pi^*$，并且它收集了无限多的数据，那么这个优化问题将与值迭代背后的优化问题完全相同。但是，值迭代要求我们知道 $P(s' \mid s, a)$，而该优化目标中没有这一项。我们并没有作弊：由于机器人使用策略 $\pi_e$ 在状态 $s_t^i$ 执行动作 $a_t^i$，下一个状态 $s_{t+1}^i$ 就是从转移函数中抽取的一个样本。因此，该优化目标同样能够访问转移函数，只是以机器人所收集数据的形式隐式地访问。

我们的优化问题的变量是所有 $s \in \mathcal{S}$ 和 $a \in \mathcal{A}$ 对应的 $Q(s, a)$。我们可以用梯度下降来最小化该目标。对于数据集中的每一对 $(s_t^i, a_t^i)$，我们可以写出

$$\begin{aligned}Q(s_t^i, a_t^i) &\leftarrow Q(s_t^i, a_t^i) - \alpha \nabla_{Q(s_t^i,a_t^i)} \ell(Q) \\&=(1 - \alpha) Q(s_t^i,a_t^i) - \alpha \Big( r(s_t^i, a_t^i) + \gamma \max_{a'} Q(s_{t+1}^i, a') \Big),\end{aligned}$$
:eqlabel:`q_learning`

其中 $\alpha$ 是学习率。在实际问题中，机器人到达目标位置时，轨迹通常就结束了。这种终止状态的价值为零，因为机器人在该状态之后不再执行任何动作。我们应当修改更新以处理这类状态：

$$Q(s_t^i, a_t^i) =(1 - \alpha) Q(s_t^i,a_t^i) - \alpha \Big( r(s_t^i, a_t^i) + \gamma (1 - \mathbb{1}_{s_{t+1}^i \textrm{ is terminal}} )\max_{a'} Q(s_{t+1}^i, a') \Big).$$

其中 $\mathbb{1}_{s_{t+1}^i \textrm{ is terminal}}$ 是一个指示变量：当 $s_{t+1}^i$ 是终止状态时它取 1，否则取 0。不在数据集中的状态-动作元组 $(s, a)$ 的值被设为 $-\infty$。该算法称为 Q 学习。

给定这些更新的解 $\hat{Q}$（它是最优价值函数 $Q^*$ 的一个近似），我们可以很容易地利用下式得到与该价值函数对应的最优确定性策略

$$\hat{\pi}(s) = \mathrm{argmax}_{a} \hat{Q}(s, a).$$

有时可能存在多个确定性策略对应同一个最优价值函数；这种平局可以任意打破，因为它们的价值函数相同。

## Q 学习中的探索

机器人用来收集数据的策略 $\pi_e$ 对于保证 Q 学习良好运行至关重要。毕竟，我们用机器人收集的数据替换了基于转移函数 $P(s' \mid s, a)$ 对 $s'$ 的期望。如果策略 $\pi_e$ 不能到达状态-动作空间的不同部分，那么很容易想象我们的估计 $\hat{Q}$ 将是对最优 $Q^*$ 的糟糕近似。同样重要的是要注意，在这种情况下，*所有状态* $s \in \mathcal{S}$ 处对 $Q^*$ 的估计都会很差，而不仅仅是 $\pi_e$ 访问过的状态。这是因为 Q 学习目标（或值迭代）是一个把所有状态-动作对的价值联系在一起的约束。因此，选择正确的策略 $\pi_e$ 来收集数据至关重要。

我们可以通过选择一个完全随机的策略 $\pi_e$（即从 $\mathcal{A}$ 中均匀随机地采样动作）来缓解这一顾虑。这样的策略会访问所有状态，但需要大量轨迹才能做到这一点。

由此我们得到 Q 学习的第二个关键思想，即探索。典型的 Q 学习实现会把 $Q$ 的当前估计与策略 $\pi_e$ 结合起来，令

$$\pi_e(a \mid s) = \begin{cases}\mathrm{argmax}_{a'} \hat{Q}(s, a') & \textrm{with prob. } 1-\epsilon \\ \textrm{uniform}(\mathcal{A}) & \textrm{with prob. } \epsilon,\end{cases}$$
:eqlabel:`epsilon_greedy`

其中 $\epsilon$ 称为“探索参数”，由用户选择。策略 $\pi_e$ 称为探索策略。这个特定的 $\pi_e$ 称为 $\epsilon$-贪心探索策略，因为它以概率 $1-\epsilon$ 选择最优动作（在当前估计 $\hat{Q}$ 下），并以剩余概率 $\epsilon$ 进行随机探索。我们也可以使用所谓的 softmax 探索策略

$$\pi_e(a \mid s) = \frac{e^{\hat{Q}(s, a)/T}}{\sum_{a'} e^{\hat{Q}(s, a')/T}};$$

其中超参数 $T$ 称为温度。$\epsilon$-贪心策略中较大的 $\epsilon$ 值与 softmax 策略中较大的温度 $T$ 起着类似的作用。

需要注意的是，当我们选择依赖于动作价值函数当前估计 $\hat{Q}$ 的探索方式时，我们需要周期性地求解该优化问题。典型的 Q 学习实现在每次使用 $\pi_e$ 执行动作之后，用所收集数据集中的几个状态-动作对（通常取自机器人前一个时间步收集的数据）做一次小批量更新。

## Q 学习的“自我纠正”性质

机器人在 Q 学习期间收集的数据集随时间增长。随着机器人收集更多数据，探索策略 $\pi_e$ 和估计 $\hat{Q}$ 都会不断演化。这为我们理解 Q 学习为何能良好运行提供了关键洞见。考虑一个状态 $s$：如果某个动作 $a$ 在当前估计 $\hat{Q}(s,a)$ 下具有很大的值，那么 $\epsilon$-贪心和 softmax 两种探索策略都有更大的概率选择该动作。如果该动作实际上*不是*理想动作，那么由该动作产生的未来状态将得到较差的奖励。因此 Q 学习目标的下一次更新将降低值 $\hat{Q}(s,a)$，从而降低机器人下次访问状态 $s$ 时选择该动作的概率。坏动作（例如在 $\hat{Q}(s,a)$ 中价值被高估的动作）会被机器人探索，但在 Q 学习目标的下一次更新中其价值会得到纠正。好动作（例如 $\hat{Q}(s, a)$ 值很大的动作）会被机器人更频繁地探索，从而得到强化。这一性质可以用来证明：即使从一个随机策略 $\pi_e$ 开始，Q 学习也能收敛到最优策略 :cite:`Watkins.Dayan.1992`。

这种不仅能收集新数据、还能收集正确类型数据的能力是强化学习算法的核心特征，也正是它区别于监督学习之处。Q 学习借助深度神经网络（我们将在后面的 DQN 章节中看到）促成了强化学习的复兴 :cite:`mnih2013playing`。

## Q 学习的实现

下面我们展示如何在 [Open AI Gym](https://gym.openai.com) 的 FrozenLake 上实现 Q 学习。请注意，这与我们在 :ref:`sec_valueiter` 实验中考虑的设置相同。

```{.python .input}
%%tab all

%matplotlib inline
import numpy as np
import random
from d2l import torch as d2l

seed = 0  # Random number generator seed
gamma = 0.95  # Discount factor
num_iters = 256  # Number of iterations
alpha   = 0.9  # Learing rate
epsilon = 0.9  # Epsilon in epsilion gready algorithm
random.seed(seed)  # Set the random seed
np.random.seed(seed)

# Now set up the environment
env_info = d2l.make_env('FrozenLake-v1', seed=seed)
```

在 FrozenLake 环境中，机器人在一个 $4 \times 4$ 网格上移动（这些格子就是状态），动作有“上”（$\uparrow$）、“下”（$\rightarrow$）、“左”（$\leftarrow$）和“右”（$\rightarrow$）。环境中有若干洞（H）格子和冰面（F）格子，以及一个目标格子（G），而所有这些对机器人来说都是未知的。为使问题简单，我们假设机器人的动作是可靠的，即对所有 $s \in \mathcal{S}, a \in \mathcal{A}$ 都有 $P(s' \mid s, a) = 1$。如果机器人到达目标，本次试验就结束，机器人无论执行什么动作都获得奖励 $1$；在其他任何状态下，所有动作的奖励均为 $0$。机器人的目标是学习一个策略，从给定的起始位置（S）（即 $s_0$）到达目标位置（G），以最大化*回报*。

我们首先按如下方式实现 $\epsilon$-贪心方法：

```{.python .input}
%%tab all

def e_greedy(env, Q, s, epsilon):
    if random.random() < epsilon:
        return env.action_space.sample()

    else:
        return np.argmax(Q[s,:])

```

现在我们准备实现 Q 学习：

```{.python .input}
%%tab all

def q_learning(env_info, gamma, num_iters, alpha, epsilon):
    env_desc = env_info['desc']  # 2D array specifying what each grid item means
    env = env_info['env']  # 2D array specifying what each grid item means
    num_states = env_info['num_states']
    num_actions = env_info['num_actions']

    Q  = np.zeros((num_states, num_actions))
    V  = np.zeros((num_iters + 1, num_states))
    pi = np.zeros((num_iters + 1, num_states))

    for k in range(1, num_iters + 1):
        # Reset environment
        state, done = env.reset(), False
        while not done:
            # Select an action for a given state and acts in env based on selected action
            action = e_greedy(env, Q, state, epsilon)
            next_state, reward, done, _ = env.step(action)

            # Q-update:
            y = reward + gamma * np.max(Q[next_state,:])
            Q[state, action] = Q[state, action] + alpha * (y - Q[state, action])

            # Move to the next state
            state = next_state
        # Record max value and max action for visualization purpose only
        for s in range(num_states):
            V[k,s]  = np.max(Q[s,:])
            pi[k,s] = np.argmax(Q[s,:])
    d2l.show_Q_function_progress(env_desc, V[:-1], pi[:-1])

q_learning(env_info=env_info, gamma=gamma, num_iters=num_iters, alpha=alpha, epsilon=epsilon)

```

这一结果表明，Q 学习大约在 250 次迭代后就能找到该问题的最优解。然而，当把该结果与值迭代算法的结果（见 :ref:`subsec_valueitercode`）比较时，我们可以看到值迭代算法找到该问题最优解所需的迭代次数要少得多。这是因为值迭代算法能够访问完整的 MDP，而 Q 学习不能。


## 小结
Q 学习是最基础的强化学习算法之一。它一直处于强化学习近期成功的中心，尤其是在学习玩电子游戏方面 :cite:`mnih2013playing`。实现 Q 学习不需要我们完全了解马尔可夫决策过程（MDP），例如转移函数和奖励函数。

## 练习

1. 试着把网格大小增加到 $8 \times 8$。与 $4 \times 4$ 网格相比，找到最优价值函数需要多少次迭代？
1. 当 $\gamma$（即上面代码中的“gamma”）分别等于 $0$、$0.5$ 和 $1$ 时，再次运行 Q 学习算法并分析其结果。
1. 当 $\epsilon$（即上面代码中的“epsilon”）分别等于 $0$、$0.5$ 和 $1$ 时，再次运行 Q 学习算法并分析其结果。

:begin_tab:`pytorch`
[讨论](https://discuss.d2l.ai/t/12103)
:end_tab:
