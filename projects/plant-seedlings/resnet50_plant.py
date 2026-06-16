# 1、使用【绝对路径】继承基配置
_base_ = [
    '../../configs/_base_/models/resnet50.py',
    '../../configs/_base_/schedules/imagenet_bs256.py',
    '../../configs/_base_/default_runtime.py'
]

# 2、全局参数
max_epochs = 300
batch_size = 16
num_classes = 12
data_root = 'G:/AI_datasets/plant-seedlings-classification'

# 修改分类头
model = dict(
    type='ImageClassifier',
    head=dict(
        num_classes=num_classes,
        loss=dict(type='FocalLoss', loss_weight=1.0)
    )
)

dataset_type = 'CustomDataset'
# 3、数据增强
# 训练数据流水线 + 异常图片检测
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomResizedCrop', scale=224),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='PackInputs'),
]

# 验证数据流水线 + 异常图片检测
val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='ResizeEdge', scale=256, edge='short'),
    dict(type='CenterCrop', crop_size=224),
    dict(type='PackInputs'),
]

# 4. 数据集加载（ImageFolder 读取文件夹分类）
# 训练集：关闭多进程 + 关闭持久进程
train_dataloader = dict(
    batch_size=batch_size,
    num_workers=0, #windows中多进程会有问题，暂时设置为0
    persistent_workers=False,  # num_workers>0时需要开启，num_workers=0时需要关闭
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix='train',
        pipeline=train_pipeline
    )
)

# 验证集：同步关闭持久进程
val_dataloader = dict(
    batch_size=batch_size,
    num_workers=0,
    persistent_workers=False,  # 关键修复
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix='val',
        pipeline=val_pipeline
    )
)
val_cfg = dict()
# ========= 核心：多评估器 = 分类指标 + 混淆矩阵 =========
val_evaluator = [
    # 1. 准确率 Top1
    dict(type='Accuracy', topk=(1,)),
    # 2. 单标签分类指标：精确率、召回率、F1-score（支持类别失衡评估）
    dict(
        type='SingleLabelMetric',
        items=['precision', 'recall', 'f1-score'],
        average='macro'  # macro：每类单独计算再平均，适合类别失衡
    ),
    # 3. 混淆矩阵（训练结束打印 + 保存文件）
     dict(
        type='ConfusionMatrix',
        num_classes=num_classes
    )
]

# 测试集三参数统一置空
test_dataloader = None
test_cfg = None
test_evaluator = None

# 5、训练优化器 & 学习率调度
optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001)
)

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.0001,
        by_epoch=True,
        begin=0,
        end=40,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=260,
        by_epoch=True,
        begin=40,
        end=300,
        convert_to_iter_based=True)
]

train_cfg = dict(max_epochs=max_epochs, val_interval=1)

# 6、日志 & 权重保存
default_hooks = dict(
    checkpoint=dict(interval=5, save_best='auto'),
    logger=dict(type='LoggerHook', interval=10)
)
# 7. 随机种子，保证复现
randomness = dict(seed=0, deterministic=False, diff_rank_seed=False)



"""
继承的是配置字典里的字段，而不是「裸顶层变量」。
字典字段（dataloader /model 这类 dict）
规则：
当前文件只写部分字段 → 和基文件同名字典做深度合并，同名 key 覆盖，缺失 key 沿用基文件。
"""