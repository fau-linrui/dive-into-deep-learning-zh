# 特征丰富的推荐系统

交互数据是用户偏好和兴趣最基本的体现。在之前介绍的模型中，它起着至关重要的作用。然而，交互数据通常极为稀疏，有时还可能带有噪声。为了解决这一问题，我们可以将物品特征、用户画像，甚至交互发生时所处的情境等辅助信息（side information）整合到推荐模型中。利用这些特征有助于进行推荐，因为这些特征可以有效地预测用户的兴趣，在交互数据匮乏时尤其如此。因此，推荐模型也必须具备处理这些特征的能力，使模型具有一定的内容/上下文感知能力。为了演示这类推荐模型，我们引入另一个任务：面向在线广告推荐的点击率（click-through rate，CTR）预测 :cite:`McMahan.Holt.Sculley.ea.2013`，并给出一个匿名广告数据集。定向广告服务引起了广泛关注，并且常常被构建为推荐引擎。推荐符合用户个人品味和兴趣的广告，对于提升点击率十分重要。


数字营销人员利用在线广告向客户展示广告。点击率是一个指标，用于衡量广告主所投放广告在每次曝光中获得的点击数量，它用百分比表示，计算公式如下：

$$ \textrm{CTR} = \frac{\#\textrm{Clicks}} {\#\textrm{Impressions}} \times 100 \% .$$

点击率是表明预测算法有效性的一个重要信号。点击率预测是一项预测网站上某项内容被点击的可能性的任务。点击率预测模型不仅可用于定向广告系统，还可用于一般的物品（例如电影、新闻、产品）推荐系统、电子邮件营销活动，甚至搜索引擎。它也与用户满意度、转化率密切相关，并且有助于设定营销活动目标，因为它可以帮助广告主设定切合实际的预期。

```{.python .input}
#@tab mxnet
from collections import defaultdict
from d2l import mxnet as d2l
from mxnet import gluon, np
import os
```

## 在线广告数据集

随着互联网和移动技术的长足发展，在线广告已成为一种重要的收入来源，并在互联网行业中产生了绝大部分收入。展示相关的广告或能激发用户兴趣的广告非常重要，这样可以把偶然到访的访客转化为付费客户。我们介绍的数据集是一个在线广告数据集。它包含 34 个字段，其中第一列表示目标变量，用以指示广告是否被点击（1 表示被点击，0 表示未被点击）。其余所有列都是类别特征。这些列可能代表广告 ID、网站或应用 ID、设备 ID、时间、用户画像等。由于匿名化和隐私方面的考虑，这些特征的真实语义并未公开。

下面的代码从我们的服务器下载数据集，并将其保存到本地数据文件夹中。

```{.python .input  n=15}
#@tab mxnet
#@save
d2l.DATA_HUB['ctr'] = (d2l.DATA_URL + 'ctr.zip',
                       'e18327c48c8e8e5c23da714dd614e390d369843f')

data_dir = d2l.download_extract('ctr')
```

数据集包含一个训练集和一个测试集，分别由 15000 和 3000 个样本（行）组成。

## 数据集包装器

为了便于加载数据，我们实现了一个 `CTRDataset`，它从 CSV 文件中加载广告数据集，并可供 `DataLoader` 使用。

```{.python .input  n=13}
#@tab mxnet
#@save
class CTRDataset(gluon.data.Dataset):
    def __init__(self, data_path, feat_mapper=None, defaults=None,
                 min_threshold=4, num_feat=34):
        self.NUM_FEATS, self.count, self.data = num_feat, 0, {}
        feat_cnts = defaultdict(lambda: defaultdict(int))
        self.feat_mapper, self.defaults = feat_mapper, defaults
        self.field_dims = np.zeros(self.NUM_FEATS, dtype=np.int64)
        with open(data_path) as f:
            for line in f:
                instance = {}
                values = line.rstrip('\n').split('\t')
                if len(values) != self.NUM_FEATS + 1:
                    continue
                label = np.float32([0, 0])
                label[int(values[0])] = 1
                instance['y'] = [np.float32(values[0])]
                for i in range(1, self.NUM_FEATS + 1):
                    feat_cnts[i][values[i]] += 1
                    instance.setdefault('x', []).append(values[i])
                self.data[self.count] = instance
                self.count = self.count + 1
        if self.feat_mapper is None and self.defaults is None:
            feat_mapper = {i: {feat for feat, c in cnt.items() if c >=
                               min_threshold} for i, cnt in feat_cnts.items()}
            self.feat_mapper = {i: {feat_v: idx for idx, feat_v in enumerate(feat_values)}
                                for i, feat_values in feat_mapper.items()}
            self.defaults = {i: len(feat_values) for i, feat_values in feat_mapper.items()}
        for i, fm in self.feat_mapper.items():
            self.field_dims[i - 1] = len(fm) + 1
        self.offsets = np.array((0, *np.cumsum(self.field_dims).asnumpy()
                                 [:-1]))
        
    def __len__(self):
        return self.count
    
    def __getitem__(self, idx):
        feat = np.array([self.feat_mapper[i + 1].get(v, self.defaults[i + 1])
                         for i, v in enumerate(self.data[idx]['x'])])
        return feat + self.offsets, self.data[idx]['y']
```

下面的示例加载训练数据并打印出第一条记录。

```{.python .input  n=16}
#@tab mxnet
train_data = CTRDataset(os.path.join(data_dir, 'train.csv'))
train_data[0]
```

可以看到，全部 34 个字段都是类别特征。每个值表示对应条目的独热索引。标签 $0$ 表示未被点击。这个 `CTRDataset` 也可以用来加载其他数据集，例如 Criteo 展示广告挑战赛[数据集](https://labs.criteo.com/2014/02/kaggle-display-advertising-challenge-dataset/)和 Avazu 点击率预测[数据集](https://www.kaggle.com/c/avazu-ctr-prediction)。

## 小结
* 点击率是衡量广告系统和推荐系统有效性的重要指标。
* 点击率预测通常被转化为二元分类问题。其目标是根据给定的特征预测某个广告/物品是否会被点击。

## 练习

* 你能用提供的 `CTRDataset` 加载 Criteo 和 Avazu 数据集吗？值得注意的是，Criteo 数据集由实值特征组成，因此你可能需要对代码略作修改。

:begin_tab:`mxnet`
[讨论](https://discuss.d2l.ai/t/405)
:end_tab:
