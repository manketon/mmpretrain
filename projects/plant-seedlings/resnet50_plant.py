# 使用【绝对路径】继承基配置，彻底解决文件找不到问题
_base_ = [
    'F:/workspace/py_thirdpart/projectsFromMyGitHub/mmpretrain/configs/_base_/models/resnet50.py',
    'F:/workspace/py_thirdpart/projectsFromMyGitHub/mmpretrain/configs/_base_/schedules/imagenet_bs256.py',
    'F:/workspace/py_thirdpart/projectsFromMyGitHub/mmpretrain/configs/_base_/default_runtime.py'
]

# 全局参数
max_epochs = 80
batch_size = 16
num_classes = 12
data_root = 'G:/AI_datasets/plant-seedlings-classification'

# 修改分类头
model = dict(
    type='ImageClassifier',
    head=dict(
        num_classes=num_classes,
    )
)

dataset_type = 'CustomDataset'

# 训练数据流水线
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomResizedCrop', scale=224),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='PackInputs'),
]

# 验证数据流水线
val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(256, -1)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='PackInputs'),
]

# 训练集加载器 Windows 固定 num_workers=0
train_dataloader = dict(
    batch_size=batch_size,
    num_workers=1,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix='train',
        pipeline=train_pipeline
    )
)

# 验证集全套配置（三参数成对出现）
val_dataloader = dict(
    batch_size=batch_size,
    num_workers=1,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix='val',
        pipeline=val_pipeline
    )
)
val_cfg = dict()
val_evaluator = dict(type='Accuracy', topk=(1,))

# 测试集三参数统一置空
test_dataloader = None
test_cfg = None
test_evaluator = None

# 训练优化器 & 学习率调度
optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
)

# learning rate scheduler
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

# 日志 & 权重保存
default_hooks = dict(
    checkpoint=dict(interval=5, save_best='auto'),
    logger=dict(type='LoggerHook', interval=10)
)
randomness = dict(seed=0, diff_rank_seed=True)