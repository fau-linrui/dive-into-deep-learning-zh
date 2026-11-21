# MovieLens 数据集

可用于推荐研究的数据集有很多。其中，[MovieLens](https://movielens.org/) 数据集可能是比较流行的一个。MovieLens 是一个非商业性的基于 Web 的电影推荐系统。它创建于 1997 年，由明尼苏达大学的研究实验室 GroupLens 运行，目的是收集电影评分数据用于研究。MovieLens 数据对包括个性化推荐和社会心理学在内的若干研究至关重要。


## 获取数据


MovieLens 数据集由 [GroupLens](https://grouplens.org/datasets/movielens/) 网站托管。该网站提供了多个版本。我们将使用 MovieLens 100K 数据集 :cite:`Herlocker.Konstan.Borchers.ea.1999`。该数据集由 943 个用户对 1682 部电影的 $100,000$ 条评分组成，评分范围为 1 到 5 星。它经过清理，使得每个用户至少评过 20 部电影。此外还提供了一些简单的人口统计信息，例如用户和物品的年龄、性别、类别。我们可以下载 [ml-100k.zip](http://files.grouplens.org/datasets/movielens/ml-100k.zip) 并解压出 `u.data` 文件，其中包含 csv 格式的全部 $100,000$ 条评分。该文件夹中还有许多其他文件，每个文件的详细说明可以在数据集的 [README](http://files.grouplens.org/datasets/movielens/ml-100k-README.txt) 文件中找到。

首先，让我们导入运行本节实验所需的包。

```{.python .input  n=1}
#@tab mxnet
from d2l import mxnet as d2l
from mxnet import gluon, np
import os
import pandas as pd
```

然后，我们下载 MovieLens 100k 数据集，并把交互加载为 `DataFrame`。

```{.python .input  n=2}
#@tab mxnet
#@save
d2l.DATA_HUB['ml-100k'] = (
    'https://files.grouplens.org/datasets/movielens/ml-100k.zip',
    'cd4dcac4241c8a4ad7badc7ca635da8a69dddb83')

#@save
def read_data_ml100k():
    data_dir = d2l.download_extract('ml-100k')
    names = ['user_id', 'item_id', 'rating', 'timestamp']
    data = pd.read_csv(os.path.join(data_dir, 'u.data'), sep='\t',
                       names=names, engine='python')
    num_users = data.user_id.unique().shape[0]
    num_items = data.item_id.unique().shape[0]
    return data, num_users, num_items
```

## 数据集的统计信息

让我们加载数据并手动检查前五条记录。这是了解数据结构并验证数据是否被正确加载的有效方法。

```{.python .input  n=3}
#@tab mxnet
data, num_users, num_items = read_data_ml100k()
sparsity = 1 - len(data) / (num_users * num_items)
print(f'number of users: {num_users}, number of items: {num_items}')
print(f'matrix sparsity: {sparsity:f}')
print(data.head(5))
```

我们可以看到，每一行由四列组成，包括 1-943 的“用户 id”、1-1682 的“物品 id”、1-5 的“评分”以及“时间戳”。我们可以构造一个大小为 $n \times m$ 的交互矩阵，其中 $n$ 和 $m$ 分别是用户数和物品数。该数据集只记录已有的评分，因此我们也可以称其为评分矩阵；当矩阵的值表示确切评分时，我们将交替使用交互矩阵和评分矩阵这两个称呼。评分矩阵中的大多数值都是未知的，因为用户没有对大多数电影评分。我们还展示了该数据集的稀疏度。稀疏度定义为 `1 - number of nonzero entries / ( number of users * number of items)`。显然，交互矩阵极其稀疏（即稀疏度 = 93.695%）。现实世界的数据集可能遭受更大程度的稀疏，这一直是构建推荐系统的长期挑战。一个可行的解决方案是利用用户/物品特征等额外的辅助信息来缓解稀疏性。

接着我们绘制不同评分计数的分布。正如所料，它看起来是一个正态分布，大多数评分集中在 3-4 分。

```{.python .input  n=4}
#@tab mxnet
d2l.plt.hist(data['rating'], bins=5, ec='black')
d2l.plt.xlabel('Rating')
d2l.plt.ylabel('Count')
d2l.plt.title('Distribution of Ratings in MovieLens 100K')
d2l.plt.show()
```

## 划分数据集

我们把数据集划分为训练集和测试集。下面的函数提供两种划分模式：`random` 和 `seq-aware`。在 `random` 模式下，该函数不考虑时间戳、随机划分这 100k 条交互，默认使用 90% 的数据作为训练样本，其余 10% 作为测试样本。在 `seq-aware` 模式下，我们留出用户最近评过的一个物品作为测试，并把用户的历史交互作为训练集。用户的历史交互按时间戳从旧到新排序。该模式将在序列感知推荐一节中使用。

```{.python .input  n=5}
#@tab mxnet
#@save
def split_data_ml100k(data, num_users, num_items,
                      split_mode='random', test_ratio=0.1):
    """Split the dataset in random mode or seq-aware mode."""
    if split_mode == 'seq-aware':
        train_items, test_items, train_list = {}, {}, []
        for line in data.itertuples():
            u, i, rating, time = line[1], line[2], line[3], line[4]
            train_items.setdefault(u, []).append((u, i, rating, time))
            if u not in test_items or test_items[u][-1] < time:
                test_items[u] = (i, rating, time)
        for u in range(1, num_users + 1):
            train_list.extend(sorted(train_items[u], key=lambda k: k[3]))
        test_data = [(key, *value) for key, value in test_items.items()]
        train_data = [item for item in train_list if item not in test_data]
        train_data = pd.DataFrame(train_data)
        test_data = pd.DataFrame(test_data)
    else:
        mask = [True if x == 1 else False for x in np.random.uniform(
            0, 1, (len(data))) < 1 - test_ratio]
        neg_mask = [not x for x in mask]
        train_data, test_data = data[mask], data[neg_mask]
    return train_data, test_data
```

请注意，实践中除测试集之外，使用验证集是一种好的做法。不过为简洁起见，我们省略了它。在这种情况下，我们的测试集可以视为留出的验证集。

## 加载数据

在划分数据集之后，为方便起见，我们将把训练集和测试集转换为列表以及字典/矩阵。下面的函数逐行读取数据框，并枚举用户/物品从零开始的索引。然后该函数返回用户、物品、评分的列表，以及记录交互的字典/矩阵。我们可以把反馈类型指定为 `explicit` 或 `implicit`。

```{.python .input  n=6}
#@tab mxnet
#@save
def load_data_ml100k(data, num_users, num_items, feedback='explicit'):
    users, items, scores = [], [], []
    inter = np.zeros((num_items, num_users)) if feedback == 'explicit' else {}
    for line in data.itertuples():
        user_index, item_index = int(line[1] - 1), int(line[2] - 1)
        score = int(line[3]) if feedback == 'explicit' else 1
        users.append(user_index)
        items.append(item_index)
        scores.append(score)
        if feedback == 'implicit':
            inter.setdefault(user_index, []).append(item_index)
        else:
            inter[item_index, user_index] = score
    return users, items, scores, inter
```

之后，我们把上述步骤组合起来，它将在下一节中使用。结果用 `Dataset` 和 `DataLoader` 包装。请注意，用于训练数据的 `DataLoader` 的 `last_batch` 被设置为 `rollover` 模式（剩余的样本被滚动到下一轮），并且顺序被打乱。

```{.python .input  n=7}
#@tab mxnet
#@save
def split_and_load_ml100k(split_mode='seq-aware', feedback='explicit',
                          test_ratio=0.1, batch_size=256):
    data, num_users, num_items = read_data_ml100k()
    train_data, test_data = split_data_ml100k(
        data, num_users, num_items, split_mode, test_ratio)
    train_u, train_i, train_r, _ = load_data_ml100k(
        train_data, num_users, num_items, feedback)
    test_u, test_i, test_r, _ = load_data_ml100k(
        test_data, num_users, num_items, feedback)
    train_set = gluon.data.ArrayDataset(
        np.array(train_u), np.array(train_i), np.array(train_r))
    test_set = gluon.data.ArrayDataset(
        np.array(test_u), np.array(test_i), np.array(test_r))
    train_iter = gluon.data.DataLoader(
        train_set, shuffle=True, last_batch='rollover',
        batch_size=batch_size)
    test_iter = gluon.data.DataLoader(
        test_set, batch_size=batch_size)
    return num_users, num_items, train_iter, test_iter
```

## 小结

* MovieLens 数据集被广泛用于推荐研究。它是公开可用且免费的。
* 我们定义了下载和预处理 MovieLens 100k 数据集的函数，以供后续章节使用。


## 练习

* 你还能找到哪些类似的推荐数据集？
* 浏览 [https://movielens.org/](https://movielens.org/) 网站以获取关于 MovieLens 的更多信息。

:begin_tab:`mxnet`
[讨论](https://discuss.d2l.ai/t/399)
:end_tab:
