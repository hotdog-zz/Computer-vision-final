auto_scale_lr = dict(base_batch_size=128)
custom_hooks = [
    dict(interval=1, type='ValLossHook'),
]
data_preprocessor = dict(
    mean=[
        129.304,
        124.07,
        112.434,
    ],
    num_classes=100,
    std=[
        68.17,
        65.392,
        70.418,
    ],
    to_rgb=True)
dataset_type = 'CIFAR100'
default_hooks = dict(
    checkpoint=dict(interval=1, type='CheckpointHook'),
    logger=dict(interval=100, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(enable=False, type='VisualizationHook'))
default_scope = 'mmpretrain'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
launcher = 'pytorch'
load_from = None
log_level = 'INFO'
model = dict(
    backbone=dict(
        depth=152,
        init_cfg=dict(
            checkpoint=
            'https://download.openmmlab.com/mmclassification/v0/resnet/resnet152_8xb32_in1k_20210901-4d7582fa.pth',
            prefix='backbone',
            type='Pretrained'),
        num_stages=4,
        out_indices=(3, ),
        style='pytorch',
        type='ResNet'),
    head=dict(
        in_channels=2048,
        loss=dict(
            label_smooth_val=0.2, mode='classy_vision',
            type='LabelSmoothLoss'),
        num_classes=100,
        type='LinearClsHead'),
    neck=dict(type='GlobalAveragePooling'),
    train_cfg=dict(augments=dict(alpha=1.0, type='CutMix')),
    type='ImageClassifier')
optim_wrapper = dict(
    optimizer=dict(lr=0.01, momentum=0.9, type='SGD', weight_decay=5e-05))
param_scheduler = dict(
    by_epoch=True, gamma=0.1, milestones=[
        30,
        70,
    ], type='MultiStepLR')
randomness = dict(deterministic=False, seed=None)
resume = False
test_cfg = dict()
test_dataloader = dict(
    batch_size=32,
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        data_root='data/cifar100/',
        pipeline=[
            dict(backend='pillow', scale=(
                224,
                224,
            ), type='Resize'),
            dict(type='PackInputs'),
        ],
        split='test',
        type='CIFAR100'),
    num_workers=2,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    topk=(
        1,
        5,
    ), type='Accuracy')
test_pipeline = [
    dict(backend='pillow', scale=(
        224,
        224,
    ), type='Resize'),
    dict(type='PackInputs'),
]
train_cfg = dict(by_epoch=True, max_epochs=100, val_interval=1)
train_dataloader = dict(
    batch_size=64,
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        data_root='data/cifar100',
        pipeline=[
            dict(backend='pillow', scale=(
                224,
                224,
            ), type='Resize'),
            dict(direction='horizontal', prob=0.5, type='RandomFlip'),
            dict(type='PackInputs'),
        ],
        split='train',
        type='CIFAR100'),
    num_workers=2,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(backend='pillow', scale=(
        224,
        224,
    ), type='Resize'),
    dict(direction='horizontal', prob=0.5, type='RandomFlip'),
    dict(type='PackInputs'),
]
val_cfg = dict()
val_dataloader = dict(
    batch_size=32,
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        data_root='data/cifar100/',
        pipeline=[
            dict(backend='pillow', scale=(
                224,
                224,
            ), type='Resize'),
            dict(type='PackInputs'),
        ],
        split='test',
        type='CIFAR100'),
    num_workers=2,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    topk=(
        1,
        5,
    ), type='Accuracy')
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    type='UniversalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='TensorboardVisBackend'),
    ])
work_dir = 'resnet152'
