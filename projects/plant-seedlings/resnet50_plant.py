# 1. 基础参数
max_epochs = 80
batch_size = 16       # 根据显存调整，显存小改为8
num_classes = 12      # 植物幼苗固定12类
data_root = "G:/AI_datasets/plant-seedlings-classification"  # 改成你的数据集绝对路径

# 2. 数据增强 Pipeline
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomResizedCrop', scale=224),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='PackInputs'),
]

val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(256, -1)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='PackInputs'),
]

# 3. 数据集加载（ImageFolder 读取文件夹分类）
train_dataloader = dict(
    batch_size=batch_size,
    num_workers=4,  # Windows 改为 0
    dataset=dict(
        type='ImageFolder',
        data_root=data_root,
        data_prefix='train',
        pipeline=train_pipeline
    )
)

val_dataloader = dict(
    batch_size=batch_size,
    num_workers=4,
    dataset=dict(
        type='ImageFolder',
        data_root=data_root,
        data_prefix='val',
        pipeline=val_pipeline
    )
)

val_evaluator = dict(type='Accuracy', topk=(1,))

# 4. 模型：ResNet50 + 加载ImageNet预训练权重（迁移学习）
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(3,),
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')
    ),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=num_classes,
        in_channels=2048,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5)
    )
)

# 5. 优化器 & 学习率调度（小数据集调低lr）
optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
)

param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=50),
    dict(
        type='CosineAnnealingLR',
        T_max=max_epochs,
        begin=0,
        end=max_epochs,
        by_epoch=True
    )
]

# 6. 训练、保存、日志配置
train_cfg = dict(by_epoch=True, max_epochs=max_epochs, val_interval=5)

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=10),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=10),  # 每10轮存权重
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='VisualizationHook', enable=False)
)

# 7. 随机种子，保证复现
randomness = dict(seed=0, deterministic=False, diff_rank_seed=False)