import os
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

import torch
from mmpretrain.evaluation import ConfusionMatrix

# 1. 定义自定义评估器，继承原生混淆矩阵 并重写 compute_metrics 方法，在计算指标的基础上自动保存文本和热力图
class ConfusionMatrixSave(ConfusionMatrix):
    """扩展原生混淆矩阵，自动保存文本+热力图"""
    def __init__(self, num_classes, save_dir='./confusion_matrix'):
        super().__init__(num_classes=num_classes)
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        # 新增缓存：存储每轮计算后的混淆矩阵
        self._cm = None

    def process(self, data_batch, data_samples):
        # 调用父类处理逻辑
        super().process(data_batch, data_samples)

    def compute_metrics(self, results):
        # 先调用父类计算指标
        metrics = super().compute_metrics(results)
        # 通用获取混淆矩阵：优先读取返回结果，其次兼容可能的内部属性
        if isinstance(metrics, dict) and 'result' in metrics:
            cm_tensor = metrics['result']
        elif hasattr(self, '_confusion_matrix'):
            cm_tensor = self._confusion_matrix
        elif hasattr(self, 'confusion_matrix'):
            cm_tensor = self.confusion_matrix
        else:
            raise RuntimeError("未获取到混淆矩阵张量")

        self._cm = cm_tensor.detach().cpu().numpy()

        # 1. 保存 txt 文本
        txt_path = os.path.join(self.save_dir, "confusion_matrix.txt")
        np.savetxt(txt_path, self._cm, fmt="%d", encoding="utf-8")

        # 2. 绘制并保存热力图
        plt.figure(figsize=(12, 10))
        plt.imshow(self._cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()

        tick_marks = np.arange(self.num_classes)
        plt.xticks(tick_marks, tick_marks, rotation=45)
        plt.yticks(tick_marks, tick_marks)

        thresh = self._cm.max() / 2.
        for i in range(self._cm.shape[0]):
            for j in range(self._cm.shape[1]):
                plt.text(j, i, format(self._cm[i, j], 'd'),
                         horizontalalignment="center",
                         color="white" if self._cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')

        img_path = os.path.join(self.save_dir, "confusion_matrix.png")
        plt.savefig(img_path, dpi=300, bbox_inches='tight')
        plt.close()

        return metrics