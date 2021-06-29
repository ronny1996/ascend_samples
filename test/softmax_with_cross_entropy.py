import paddle
import numpy as np


label = np.random.randint(low=0, high=2, size=[32, 1])
logits = (np.random.random(size=[32, 102]) - 0.5) * 1000
y = paddle.nn.functional.cross_entropy(input=paddle.to_tensor(logits), label=paddle.to_tensor(label))
print(y)