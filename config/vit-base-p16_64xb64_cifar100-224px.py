auto_scale_lr = dict(base_batch_size=1024)
custom_hooks = [
    dict(interval=1, type='ValLossHook'),
    dict(momentum=0.001, type='EMAHook'),
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
        arch='base',
        drop_rate=0.1,
        img_size=224,
        init_cfg=[
            dict(
                checkpoint=
                'https://download.openmmlab.com/mmclassification/v0/vit/vit-base-p16_pt-32xb128-mae_in1k_20220623-4c544545.pth',
                prefix='backbone',
                type='Pretrained'),
        ],
        patch_size=16,
        type='VisionTransformer'),
    head=dict(
        in_channels=768,
        loss=dict(
            label_smooth_val=0.05, mode='original', type='LabelSmoothLoss'),
        num_classes=100,
        type='VisionTransformerClsHead'),
    neck=None,
    train_cfg=dict(augments=dict(alpha=1.0, type='CutMix')),
    type='ImageClassifier')
optim_wrapper = dict(
    optimizer=dict(
        betas=(
            0.9,
            0.95,
        ),
        eps=1e-08,
        lr=0.0004,
        type='AdamW',
        weight_decay=0.3),
    paramwise_cfg=dict(
        bias_decay_mult=0.0,
        custom_keys=dict({
            '.cls_token': dict(decay_mult=0.0),
            '.pos_embed': dict(decay_mult=0.0)
        }),
        norm_decay_mult=0.0))
param_scheduler = [
    dict(
        by_epoch=True,
        convert_to_iter_based=True,
        end=15,
        start_factor=0.01,
        type='LinearLR'),
    dict(by_epoch=True, gamma=0.5, milestones=[
        40,
        100,
    ], type='MultiStepLR'),
]
randomness = dict(deterministic=False, seed=None)
resume = False
test_cfg = dict()
test_dataloader = dict(
    batch_size=32,
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        data_root='data/cifar100/',
        pipeline=[
            dict(
                backend='pillow',
                edge='short',
                interpolation='bicubic',
                scale=224,
                type='ResizeEdge'),
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
    dict(
        backend='pillow',
        edge='short',
        interpolation='bicubic',
        scale=224,
        type='ResizeEdge'),
    dict(type='PackInputs'),
]
train_cfg = dict(by_epoch=True, max_epochs=50, val_interval=1)
train_dataloader = dict(
    batch_size=128,
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        data_root='data/cifar100',
        pipeline=[
            dict(
                backend='pillow',
                edge='short',
                interpolation='bicubic',
                scale=224,
                type='ResizeEdge'),
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
    dict(
        backend='pillow',
        edge='short',
        interpolation='bicubic',
        scale=224,
        type='ResizeEdge'),
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
            dict(
                backend='pillow',
                edge='short',
                interpolation='bicubic',
                scale=224,
                type='ResizeEdge'),
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
work_dir = 'vit'
