import sys
import os

# 1. 把当前目录加入模块搜索路径
CURR_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(CURR_DIR)
from tools.custom_evaluators import ConfusionMatrixSave
from mmengine.registry import METRICS
# 注册自定义评估器
METRICS.register_module(module=ConfusionMatrixSave, name='ConfusionMatrixSave')

# 3. 调用mmpretrain官方训练入口
from tools.train import main

if __name__ == '__main__':
    # 传入你的配置文件路径
    sys.argv.extend(['resnet50_plant.py'])
    main()
#目前ConfusionMatrixSave还存在问题。
#运行方式 python train.py