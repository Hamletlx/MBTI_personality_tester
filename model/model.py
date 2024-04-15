import numpy as np
import PIL.Image as Image

import torch
import torch.nn as nn
from torchvision import models
from torchvision import transforms


class Model(object):
    def __init__(self):
        self.classes = ['angry', 'disgust', 'fear',
                        'happy', 'neutral', 'sad', 'surprise']
        self.num_classes = len(self.classes)
        self.model_weight = 'weight/pretrained_model.pth'
        self.model = self.build_model()
        self.transform = self.build_transform([112, 112])

    def build_model(self):
        model = models.mobilenet_v2()
        last_channel = model.last_channel
        classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(last_channel, self.num_classes),
        )
        model.classifier = classifier
        state_dict = torch.load(
            self.model_weight, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
        model.eval()
        return model

    def build_transform(self, input_size, rgb_mean=[0.5, 0.5, 0.5], rgb_std=[0.5, 0.5, 0.5]):
        transform = transforms.Compose([
            transforms.Resize([int(128 * input_size[1] / 112),
                              int(128 * input_size[0] / 112)]),
            transforms.CenterCrop([input_size[1], input_size[0]]),
            transforms.ToTensor(),
            transforms.Normalize(mean=rgb_mean, std=rgb_std),
        ])
        return transform

    def pre_process(self, images):
        image_tensors = []
        for img in images:
            pilimg = Image.fromarray(img)
            image_tensor = self.transform(pilimg)
            image_tensors.append(torch.unsqueeze(image_tensor, dim=0))
        image_tensors = torch.cat(image_tensors)
        return image_tensors

    def post_process(self, output):
        prob_scores = self.softmax(np.asarray(output), axis=1)
        if isinstance(prob_scores, torch.Tensor):
            prob_scores = prob_scores.cpu().detach().numpy()
        pred_index = np.argmax(prob_scores, axis=1)
        pred_class = [self.classes[i] for i in pred_index]
        pred_score = np.max(prob_scores, axis=1)
        return pred_index, pred_class, pred_score

    @staticmethod
    def softmax(x, axis=1):
        # 计算每行的最大值
        row_max = x.max(axis=axis)
        # 每行元素都需要减去对应的最大值，否则求exp(x)会溢出，导致inf情况
        row_max = row_max.reshape(-1, 1)
        x = x - row_max
        # 计算e的指数次幂
        x_exp = np.exp(x)
        x_sum = np.sum(x_exp, axis=axis, keepdims=True)
        s = x_exp / x_sum
        return s

    def forward(self, input_tensor):
        """
        :param input_tensor: input tensor
        :return:
        """
        with torch.no_grad():
            out_tensor = self.model(input_tensor)
        return out_tensor

    def detect(self, images):
        # 图像预处理
        input_tensor = self.pre_process(images)
        output = self.forward(input_tensor)
        # 模型输出后处理
        pred_index, pred_class, pred_score = self.post_process(output)
        return pred_index, pred_class, pred_score
