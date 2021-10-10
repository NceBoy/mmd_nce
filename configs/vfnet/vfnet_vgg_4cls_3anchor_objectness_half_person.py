# fp16 = dict(loss_scale=512.)
dataset_type = 'CocoDataset'
data_root = '/usr/videodate/dataset/coco/'
base_lr = 0.01
warmup_iters = 2000
model = dict(
    type='VFNet',
    backbone=dict(
        type='VGGNet',
        stem_channels=32,
        stage_channels=(32, 64, 96, 128, 192, 256),
        block_per_stage=(1, 3, 6, 8, 2, 2),
        kernel_size=[3, 3, 3, 3, 3, 3],
        conv_cfg=dict(type="RepVGGConv"),
        num_out=5,
    ),
    neck=dict(
        type='YLFPNv2',
        in_channels=[64, 96, 128, 192, 256],
        out_channels=64,
        conv_cfg=dict(type="SepConv")
        ),
    bbox_head=dict(
        type='VFNetMultiAnchor',
        norm_cfg=dict(type='BN', requires_grad=True),
        num_classes=4,
        in_channels=64,
        stacked_convs=2,
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
    nms=dict(type='nms', iou_threshold=0.6),
    max_per_img=100)

train_pipline = [
            dict(type='LoadImageFromFile', to_float32=True),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                type='Resize',
                img_scale=[(1333, 960), (1333, 480)],
                multiscale_mode='range',
                keep_ratio=True),
            dict(type='PhotoMetricDistortion', brightness_delta=48),
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

val_pipline = [
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1333, 800),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
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
    samples_per_gpu=28,
    workers_per_gpu=4,
    train=dict(
        type='CocoDataset',
        ann_file=data_root + "annotations/coco_half_person_80_train.json",
        img_prefix=data_root + 'train2017',
        classes=['person', 'bottle', 'chair', 'potted plant'],
        pipeline=train_pipline,
        filter_empty_gt=True),

    val=dict(
        type='CocoDataset',
        ann_file=data_root + "annotations/coco_half_person_82_camera_fake_personval.json",
        img_prefix=data_root + 'val2017',
        classes=['person', 'bottle', 'chair', 'potted plant'],
        pipeline=val_pipline,
        filter_empty_gt=True),
    test=dict(
        type='CocoDataset',
        ann_file=data_root + "annotations/coco_half_person_82_camera_fake_personval.json",
        img_prefix=data_root + 'val2017',
        classes=['person', 'bottle', 'chair', 'potted plant'],
        pipeline=val_pipline,
        filter_empty_gt=True),
        )

evaluation = dict(interval=2, metric='bbox', classwise=True)



# optimizer = dict(type='SGD', lr=0.1, momentum=0.937, weight_decay=0.0005)
optimizer = dict(type='AdamW', lr=0.001)
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.01,
    step=[90, 110])


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

work_dir = './tools/work_dirs/vfnet_vgg_4cls_3anchor_objectness_half_person'
load_from = None
resume_from = './tools/work_dirs/vfnet_vgg_4cls_3anchor_objectness_half_person/latest.pth'
# resume_from = None
workflow = [('train', 1)]
gpu_ids = range(0, 2)

