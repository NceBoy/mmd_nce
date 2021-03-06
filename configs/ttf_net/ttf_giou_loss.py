# model settings
dataset_type = 'CocoWiderfaceDataset'
data_root = '/root/dataset/'

base_lr = 0.02
warmup_iters = 500

model = dict(
    type='TTFNet',
    backbone=dict(
        type='RepVGGNet',
        stem_channels=32,
        stage_channels=(32, 64, 96, 128),
        block_per_stage=(1, 3, 6, 8, 6, 6),
        kernel_size=[3, 3, 3, 3],
        num_out=4,
    ),

    neck=dict(
        type='FuseFPN',
        in_channels=[32, 64, 96, 128],
        out_channels=64,
        conv_cfg=dict(type="NormalConv",
                      info={"norm_cfg": None})),
    bbox_head=dict(
        type='TTFHead',
        planes=(64, 64, 64),
        base_down_ratio=32,
        head_conv=64,
        wh_conv=64,
        hm_head_conv_num=2,
        wh_head_conv_num=2,
        num_classes=1,
        wh_offset_base=16,
        wh_agnostic=True,
        wh_gaussian=True,
        norm_cfg=dict(type='SyncBN'),
        alpha=0.54,
        hm_weight=1.,
        wh_weight=5.,
        use_dla=True))
cudnn_benchmark = True
# training and testing settings
train_cfg = dict(
    vis_every_n_iters=100,
    debug=False)
test_cfg = dict(
    score_thr=0.01,
    max_per_img=100)
# dataset settings

img_norm_cfg = dict(
    mean=[127.5, 127.5, 127.5], std=[128, 128, 128], to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='MinIoURandomCrop',
        min_ious=(0.6, 0.7, 0.8, 0.9),
        min_crop_size=0.3),
    dict(
        type='Resize',
        img_scale=[(512, 512)],
        multiscale_mode='value',
        keep_ratio=False),
    dict(type='PhotoMetricDistortion'),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='Normalize',
        mean=[127.5, 127.5, 127.5],
        std=[128, 128, 128],
        to_rgb=True),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(320, 320),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=False),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[127.5, 127.5, 127.5],
                std=[128, 128, 128],
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=64,
    workers_per_gpu=4,
    train=
        dict(
        type=dataset_type,
        ann_file=data_root+'WIDER_annotations/WIDERFaceTrainCOCO.json',
        img_prefix=data_root+'WIDER_train/images',
        pipeline=train_pipeline),

    val=dict(
        type=dataset_type,
        ann_file=data_root+'WIDER_annotations/WIDERFaceValCOCO.json',
        img_prefix=data_root+'WIDER_val/images',
        pipeline=test_pipeline),

    test=dict(
        type=dataset_type,
        ann_file=data_root+'WIDER_annotations/WIDERFaceValCOCO.json',
        img_prefix=data_root+'WIDER_val/images',
        pipeline=test_pipeline)
            )

evaluation = dict(start=50, interval=2, metric='bbox')

# optimizer = dict(type='AdamW', lr=0.001)
# optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))

optimizer = dict(type='SGD', lr=base_lr, momentum=0.937, weight_decay=0.0005)
optimizer_config = dict(grad_clip=None)


lr_config = dict(
    policy='CosineAnnealing',
    min_lr=base_lr/1000,
    warmup='linear',
    warmup_iters=warmup_iters,
    warmup_ratio=0.001)
total_epochs = 300

checkpoint_config = dict(interval=2)


log_config = dict(
    interval=20,
    hooks=[
        dict(type='TextLoggerHook')
    ])

# yapf:enable
# runtime settings


device_ids = range(1)
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = 'work_dirs/ttfnet_RepVGG_eiou'
load_from = None
resume_from = None
workflow = [('train', 1)]