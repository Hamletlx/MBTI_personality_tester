import os
import cv2
import numpy as np

import torch
import torch.nn as nn


class MbtiModule(nn.Module):
    def __init__(self):
        super(MbtiModule, self).__init__()
        """
        输入是一个长度为7的Tensor
        """
        self.linear = nn.Sequential(
            nn.Linear(7, 32),
            nn.ReLU(),
        )

        self.gru = nn.GRU(32, 16, num_layers=1)

        self.result = nn.Sequential(
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, input):
        output = []
        for i in input:
            output.append(self.linear(i))
        output = torch.stack(output)
        output = self.gru(output)[0]
        return self.result(output[-1])


if __name__ == '__main__':

    # 创建模型实例
    module = MbtiModule()
    print('参数总数：', sum(p.numel() for p in module.parameters()))

    # 构造示例输入数据
    sequence_length = 20
    batch_size = 10
    length_size = 7
    target_tensor = torch.randn(
        sequence_length, batch_size, length_size)
    print('输入数据：', target_tensor.shape)

    # 进行前向传播
    output = module(target_tensor)
    print('输出数据：', output.shape)
    print('输出结果：', output)
