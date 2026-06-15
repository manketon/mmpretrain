_base_ = [
    '../../configs/_base_/models/resnet50.py',
    '../../configs/_base_/datasets/imagenet_bs32.py',
    '../../configs/_base_/schedules/imagenet_bs256.py',
    '../../configs/_base_/default_runtime.py',
]

max_epochs = 80
batch_size = 16  # 根据显存调整，显存小可改为 8
num_classes = 12  # 植物幼苗固定 12 类
data_root = 'G:/AI_datasets/plant-seedlings-classification'  # 需要包含 train/ 和 val/ 两个目录

# 模型头：修改分类输出维度
model = dict(
    type='ImageClassifier',    # 指定模型的类别
    head=dict(
        num_classes=num_classes,
    )
)

# 3. 数据集加载配置（文件夹分类格式）
dataset_type = 'CustomDataset'
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

train_dataloader = dict(
    batch_size=batch_size,   # 根据显存调整，8G显存建议8~16
    num_workers=4,   # Windows 建议 <=4，过高会报错
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix='train',  # 训练集子文件夹
        pipeline=train_pipeline
    )
)

val_dataloader = dict(
    batch_size=batch_size,
    num_workers=4,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix='val',    # 验证集子文件夹
        pipeline=val_pipeline
    )
)

# 4. 训练轮次、优化器、学习率
max_epochs = 100
optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
)
param_scheduler = [
    dict(type='StepLR', step=30, gamma=0.1)
]

# 5. 训练轮次、日志、保存权重
train_cfg = dict(max_epochs=max_epochs, val_interval=1)  # 每轮验证一次
val_cfg = dict() 
test_cfg = dict()                    # 测试流程配置 空字典表示使用默认配置
default_hooks = dict(
    checkpoint=dict(interval=5, save_best='auto'),  # 每5轮存权重，保存最优模型
    logger=dict(type='LoggerHook', interval=10)
)
randomness = dict(seed=0, diff_rank_seed=True)  # 固定随机种子，保证结果可复现
