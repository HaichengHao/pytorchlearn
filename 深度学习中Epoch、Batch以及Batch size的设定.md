**[Epoch](https://zhida.zhihu.com/search?content_id=174872971&content_type=Article&match_order=1&q=Epoch&zhida_source=entity)（时期）：**

当一个完整的数据集通过了神经网络一次并且返回了一次，这个过程称为一次>epoch。（也就是说，所有训练样本在神经网络中都 进行了一次正向传播 和一次反向传播 ）

再通俗一点，一个Epoch就是将所有训练样本训练一次的过程。

然而，当一个Epoch的样本（也就是所有的训练样本）数量可能太过庞大（对于计算机而言），就需要把它分成多个小块，也就是就是分成多个[Batch](https://zhida.zhihu.com/search?content_id=174872971&content_type=Article&match_order=1&q=Batch&zhida_source=entity) 来进行训练。\*\*

*   **Batch（批 / 一批样本）：**

将整个训练样本分成若干个Batch。

*   **Batch\_Size（批大小）：**

每批样本的大小。

*   **Iteration（一次迭代）：**

训练一个Batch就是一次Iteration（这个概念跟程序语言中的迭代器相似）

*   **为什么要使用多于一个epoch?**

在神经网络中传递完整的数据集一次是不够的，而且我们需要将完整的数据集在同样的神经网络中传递多次。但请记住，我们使用的是有限的数据集，并且我们使用一个迭代过程即[梯度下降](https://zhida.zhihu.com/search?content_id=174872971&content_type=Article&match_order=1&q=%E6%A2%AF%E5%BA%A6%E4%B8%8B%E9%99%8D&zhida_source=entity)来优化学习过程。如下图所示。因此仅仅更新一次或者说使用一个epoch是不够的。

![](https://pic1.zhimg.com/v2-5fc01a0be98c4a88d5abecc1b1f413e2_1440w.jpg)

  

随着epoch数量增加，神经网络中的权重的更新次数也在增加，曲线从欠拟合变得过拟合。

那么，到底多少个epoch最合适，这个问题没有正确答案，对于不同的数据集，答案都不相同。

[Batch Size](https://zhida.zhihu.com/search?content_id=174872971&content_type=Article&match_order=1&q=Batch+Size&zhida_source=entity)
-------------------------------------------------------------------------------------------------------------------------------------

### **直观的理解：**

Batch Size定义：一次训练所选取的样本数。

Batch Size的大小影响模型的优化程度和速度。同时其直接影响到GPU内存的使用情况，假如GPU内存不大，该数值最好设置小一点。

### 为什么要提出Batch Size？

在没有使用Batch Size之前，这意味着网络在训练时，是一次把所有的数据（整个数据库）输入网络中，然后计算它们的梯度进行反向传播，由于在计算梯度时使用了整个数据库，所以计算得到的梯度方向更为准确。但在这情况下，计算得到不同梯度值差别巨大，难以使用一个全局的学习率，所以这时一般使用[Rprop](https://zhida.zhihu.com/search?content_id=174872971&content_type=Article&match_order=1&q=Rprop&zhida_source=entity)这种基于梯度符号的训练算法，单独进行梯度更新。

在小样本数的数据库中，不使用Batch Size是可行的，而且效果也很好。但是一旦是大型的数据库，一次性把所有数据输进网络，肯定会引起内存的爆炸。所以就提出Batch Size的概念。

### Batch Size合适的优点：

1、通过并行化提高内存的利用率。就是尽量让你的GPU满载运行，提高训练速度。

2、单个epoch的迭代次数减少了，参数的调整也慢了，假如要达到相同的识别精度，需要更多的epoch。

3、适当Batch Size使得梯度下降方向更加准确。

### Batch Size从小到大的变化对网络影响

1、没有Batch Size，梯度准确，只适用于小样本数据库

2、Batch Size=1，梯度变来变去，非常不准确，网络很难收敛。

3、Batch Size增大，梯度变准确，

4、Batch Size增大，梯度已经非常准确，再增加Batch Size也没有用

注意：Batch Size增大了，要到达相同的准确度，必须要增大epoch。

**GD（Gradient Descent）：**就是没有利用Batch Size，用基于整个数据库得到梯度，梯度准确，但数据量大时，计算非常耗时，同时神经网络常是非凸的，网络最终可能收敛到初始点附近的局部最优点。

**SGD（Stochastic Gradient Descent）：**就是Batch Size=1，每次计算一个样本，梯度不准确，所以学习率要降低。

**[mini-batch SGD](https://zhida.zhihu.com/search?content_id=174872971&content_type=Article&match_order=1&q=mini-batch+SGD&zhida_source=entity)：**就是选着合适Batch Size的SGD算法，mini-batch利用噪声梯度，一定程度上缓解了GD算法直接掉进初始点附近的局部最优值。同时梯度准确了，学习率要加大。

​

本文转自 <https://zhuanlan.zhihu.com/p/390341772>，如有侵权，请联系删除。