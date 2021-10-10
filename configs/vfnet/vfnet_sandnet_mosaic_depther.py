# fp16 = dict(loss_scale=512.)
dataset_type = 'CocoDataset'
data_root = '/usr/videodate/dataset/coco/'
base_lr = 0.01
warmup_iters = 2000
interval = 10
model = dict(
    type='VFNet',
    backbone=dict(
        type='GeneralSandNet',
        stem_channels=32,
        stage_channels=(32, 36, 48, 64, 64, 72),
        block_per_stage=(1, 2, 4, 4, 1, 1),
        expansion=[1, 4, 4, 4, 4, 4],
        kernel_size=[3, 3, 3, 3, 3, 3],
        conv_cfg=dict(type="RepVGGConv"),
        num_out=5,
    ),
    neck=dict(
        type='YLFPNv2',
        in_channels=[144, 192, 256, 256, 288],
        out_channels=64,
        conv_cfg=dict(type="SepConv")
        ),
    bbox_head=dict(
        type='VFNetMultiAnchor',
        norm_cfg=dict(type='BN', requires_grad=True),
        num_classes=4,
        in_channels=64,
        stacked_convs=1,
        feat_channels=64,
        strides=[8, 16, 32, 64, 128],
        center_sampling=False,
        dcn_on_last_conv=False,
        denorm=True,
        use_atss=True,
        use_vfl=True,
        # bbox_coder=dict(_delete_=True, type='TBLRBBoxCoder', normalizer=4.0),
        loss_cls=dict(
            type='VarifocalLoss',
            use_sigmoid=True,
            alpha=0.75,
            gamma=2.0,
            iou_weighted=True,
            loss_weight=1.0),
        loss_bbox=dict(type='CIoULoss', loss_weight=1.5),
        anchor_generator=dict(
            type='AnchorGenerator',
            ratios=[0.5, 1.0, 2.0],
            octave_base_scale=6,
            scales_per_octave=1,
            center_offset=0.0,
            strides=[8, 16, 32, 64, 128]),
        loss_bbox_refine=dict(type='CIoULoss', loss_weight=2.0),
        objectness=True))

# training and testing settings
train_cfg = dict(
    assigner=dict(type='ATSSAssigner', topk=9),
    allowed_border=-1,
    pos_weight=-1,
    debug=False)
test_cfg = dict(
    nms_pre=1000,
    min_bbox_size=0,
    score_thr=0.05,
    nms=dict(type='nms', iou_threshold=0.3),
    max_per_img=100)

img_scale = (640, 640)
img_norm_cfg = dict(
    mean=[127.5, 127.5, 127.5], std=[128, 128, 128], to_rgb=True)

train_pipeline = [
    dict(type='Mosaic', img_scale=img_scale, pad_val=127.5),
    dict(
        type='RandomAffine',
        scaling_ratio_range=(0.1, 2),
        border=(-img_scale[0] // 2, -img_scale[1] // 2)),
    dict(
        type='PhotoMetricDistortion',
        brightness_delta=32,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        hue_delta=18),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Resize', keep_ratio=True),
    dict(type='Pad', pad_to_square=True, pad_val=114.0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]

train_dataset = dict(
    type='MultiImageMixDataset',
    dataset=dict(
        type=dataset_type,
        ann_file=data_root + "annotations/coco_half_person_80_train.json",
        img_prefix=data_root + 'train2017',
        classes=['person', 'bottle', 'chair', 'potted plant'],
        pipeline=[
            dict(type='LoadImageFromFile', to_float32=True),
            dict(type='LoadAnnotations', with_bbox=True)
        ],
        filter_empty_gt=True,
    ),
    pipeline=train_pipeline,
    dynamic_scale=img_scale)

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=img_scale,
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Pad', size=img_scale, pad_val=114.0),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img'])
        ])
]

data = dict(
    samples_per_gpu=24,
    workers_per_gpu=4,
    train=train_dataset,
    val=dict(
        type=dataset_type,
        ann_file=data_root + "annotations/coco_half_person_82_camera_fake_personval.json",
        img_prefix=data_root + 'val2017',
        classes=['person', 'bottle', 'chair', 'potted plant'],
        filter_empty_gt=True,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + "annotations/coco_half_person_82_camera_fake_personval.json",
        img_prefix=data_root + 'val2017',
        classes=['person', 'bottle', 'chair', 'potted plant'],
        filter_empty_gt=True,
        pipeline=test_pipeline))


evaluation = dict(interval=2, metric='bbox', classwise=True)

custom_hooks = [
    dict(type='YOLOXModeSwitchHook', num_last_epochs=15, priority=48),
    dict(
        type='SyncRandomSizeHook',
        ratio_range=(14, 26),
        img_scale=img_scale,
        interval=interval,
        priority=48),
    dict(
        type='SyncNormHook',
        num_last_epochs=15,
        interval=interval,
        priority=48),
]

optimizer = dict(type='AdamW', lr=0.001)
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=base_lr/1000,
    warmup='linear',
    warmup_iters=warmup_iters,
    warmup_ratio=0.001)


total_epochs = 120
# learning policy

checkpoint_config = dict(interval=1)
log_config = dict(
    interval=20,
    hooks=[dict(type='TextLoggerHook'),
           dict(type='TensorboardLoggerHook')])


device_ids = range(0, 2)
dist_params = dict(backend='nccl')
log_level = 'INFO'

work_dir = './tools/work_dirs/vfnet_sandnet_mosaic_depther'
load_from = None
resume_from = './tools/work_dirs/vfnet_sandnet_mosaic_depther/latest.pth'
# resume_from = None
workflow = [('train', 1)]
gpu_ids = range(0, 2)

