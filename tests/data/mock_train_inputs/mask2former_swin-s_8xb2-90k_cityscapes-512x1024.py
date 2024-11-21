default_scope = 'mmseg'
env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='SegLocalVisualizer',
    vis_backends=[dict(type='LocalVisBackend')],
    name='visualizer')
log_processor = dict(by_epoch=False)
log_level = 'INFO'
load_from = None
resume = False
tta_model = dict(type='SegTTAModel')
dataset_type = 'CityscapesDataset'
data_root = 'data/cityscapes/'
crop_size = (512, 1024)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(
        type='RandomChoiceResize',
        scales=[
            512, 614, 716, 819, 921, 1024, 1126, 1228, 1331, 1433, 1536, 1638,
            1740, 1843, 1945, 2048
        ],
        resize_type='ResizeShortestEdge',
        max_size=4096),
    dict(type='RandomCrop', crop_size=(512, 1024), cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(2048, 1024), keep_ratio=True),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]
img_ratios = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
tta_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(
        type='TestTimeAug',
        transforms=[[{
            'type': 'Resize',
            'scale_factor': 0.5,
            'keep_ratio': True
        }, {
            'type': 'Resize',
            'scale_factor': 0.75,
            'keep_ratio': True
        }, {
            'type': 'Resize',
            'scale_factor': 1.0,
            'keep_ratio': True
        }, {
            'type': 'Resize',
            'scale_factor': 1.25,
            'keep_ratio': True
        }, {
            'type': 'Resize',
            'scale_factor': 1.5,
            'keep_ratio': True
        }, {
            'type': 'Resize',
            'scale_factor': 1.75,
            'keep_ratio': True
        }],
            [{
                'type': 'RandomFlip',
                'prob': 0.0,
                'direction': 'horizontal'
            }, {
                'type': 'RandomFlip',
                'prob': 1.0,
                'direction': 'horizontal'
            }], [{
                'type': 'LoadAnnotations'
            }], [{
                'type': 'PackSegInputs'
            }]])
]
train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type='CityscapesDataset',
        data_root='data/cityscapes/',
        data_prefix=dict(
            img_path='leftImg8bit/train', seg_map_path='gtFine/train'),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),
            dict(
                type='RandomChoiceResize',
                scales=[
                    512, 614, 716, 819, 921, 1024, 1126, 1228, 1331, 1433,
                    1536, 1638, 1740, 1843, 1945, 2048
                ],
                resize_type='ResizeShortestEdge',
                max_size=4096),
            dict(type='RandomCrop', crop_size=(512, 1024), cat_max_ratio=0.75),
            dict(type='RandomFlip', prob=0.5),
            dict(type='PhotoMetricDistortion'),
            dict(type='PackSegInputs')
        ]))
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='CityscapesDataset',
        data_root='data/cityscapes/',
        data_prefix=dict(
            img_path='leftImg8bit/val', seg_map_path='gtFine/val'),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='Resize', scale=(2048, 1024), keep_ratio=True),
            dict(type='LoadAnnotations'),
            dict(type='PackSegInputs')
        ]))
test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='CityscapesDataset',
        data_root='data/cityscapes/',
        data_prefix=dict(
            img_path='leftImg8bit/val', seg_map_path='gtFine/val'),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='Resize', scale=(2048, 1024), keep_ratio=True),
            dict(type='LoadAnnotations'),
            dict(type='PackSegInputs')
        ]))
val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    size=(512, 1024),
    test_cfg=dict(size_divisor=32))
num_classes = 19
model = dict(
    type='EncoderDecoder',
    data_preprocessor=dict(
        type='SegDataPreProcessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_val=0,
        seg_pad_val=255,
        size=(512, 1024),
        test_cfg=dict(size_divisor=32)),
    backbone=dict(
        type='SwinTransformer',
        embed_dims=96,
        depths=[2, 2, 18, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.3,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        with_cp=False,
        frozen_stages=-1,
        init_cfg=dict(
            type='Pretrained',
            checkpoint=
            'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/swin/swin_small_patch4_window7_224_20220317-7ba6d6dd.pth'
        )),
    decode_head=dict(
        type='Mask2FormerHead',
        in_channels=[96, 192, 384, 768],
        strides=[4, 8, 16, 32],
        feat_channels=256,
        out_channels=256,
        num_classes=19,
        num_queries=100,
        num_transformer_feat_level=3,
        align_corners=False,
        pixel_decoder=dict(
            type='mmdet.MSDeformAttnPixelDecoder',
            num_outs=3,
            norm_cfg=dict(type='GN', num_groups=32),
            act_cfg=dict(type='ReLU'),
            encoder=dict(
                num_layers=6,
                layer_cfg=dict(
                    self_attn_cfg=dict(
                        embed_dims=256,
                        num_heads=8,
                        num_levels=3,
                        num_points=4,
                        im2col_step=64,
                        dropout=0.0,
                        batch_first=True,
                        norm_cfg=None,
                        init_cfg=None),
                    ffn_cfg=dict(
                        embed_dims=256,
                        feedforward_channels=1024,
                        num_fcs=2,
                        ffn_drop=0.0,
                        act_cfg=dict(type='ReLU', inplace=True))),
                init_cfg=None),
            positional_encoding=dict(num_feats=128, normalize=True),
            init_cfg=None),
        enforce_decoder_input_project=False,
        positional_encoding=dict(num_feats=128, normalize=True),
        transformer_decoder=dict(
            return_intermediate=True,
            num_layers=9,
            layer_cfg=dict(
                self_attn_cfg=dict(
                    embed_dims=256,
                    num_heads=8,
                    attn_drop=0.0,
                    proj_drop=0.0,
                    dropout_layer=None,
                    batch_first=True),
                cross_attn_cfg=dict(
                    embed_dims=256,
                    num_heads=8,
                    attn_drop=0.0,
                    proj_drop=0.0,
                    dropout_layer=None,
                    batch_first=True),
                ffn_cfg=dict(
                    embed_dims=256,
                    feedforward_channels=2048,
                    num_fcs=2,
                    act_cfg=dict(type='ReLU', inplace=True),
                    ffn_drop=0.0,
                    dropout_layer=None,
                    add_identity=True)),
            init_cfg=None),
        loss_cls=dict(
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=2.0,
            reduction='mean',
            class_weight=[
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.1
            ]),
        loss_mask=dict(
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=True,
            reduction='mean',
            loss_weight=5.0),
        loss_dice=dict(
            type='mmdet.DiceLoss',
            use_sigmoid=True,
            activate=True,
            reduction='mean',
            naive_dice=True,
            eps=1.0,
            loss_weight=5.0),
        train_cfg=dict(
            num_points=12544,
            oversample_ratio=3.0,
            importance_sample_ratio=0.75,
            assigner=dict(
                type='mmdet.HungarianAssigner',
                match_costs=[
                    dict(type='mmdet.ClassificationCost', weight=2.0),
                    dict(
                        type='mmdet.CrossEntropyLossCost',
                        weight=5.0,
                        use_sigmoid=True),
                    dict(
                        type='mmdet.DiceCost',
                        weight=5.0,
                        pred_act=True,
                        eps=1.0)
                ]),
            sampler=dict(type='mmdet.MaskPseudoSampler'))),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
embed_multi = dict(lr_mult=1.0, decay_mult=0.0)
optimizer = dict(
    type='AdamW', lr=0.0001, weight_decay=0.05, eps=1e-08, betas=(0.9, 0.999))
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=0.0001,
        weight_decay=0.05,
        eps=1e-08,
        betas=(0.9, 0.999)),
    clip_grad=dict(max_norm=0.01, norm_type=2),
    paramwise_cfg=dict(
        custom_keys=dict({
            'backbone':
                dict(lr_mult=0.1, decay_mult=1.0),
            'query_embed':
                dict(lr_mult=1.0, decay_mult=0.0),
            'query_feat':
                dict(lr_mult=1.0, decay_mult=0.0),
            'level_embed':
                dict(lr_mult=1.0, decay_mult=0.0),
            'backbone.patch_embed.norm':
                dict(lr_mult=0.1, decay_mult=0.0),
            'backbone.norm':
                dict(lr_mult=0.1, decay_mult=0.0),
            'absolute_pos_embed':
                dict(lr_mult=0.1, decay_mult=0.0),
            'relative_position_bias_table':
                dict(lr_mult=0.1, decay_mult=0.0),
            'backbone.stages.0.blocks.0.norm':
                dict(lr_mult=0.1, decay_mult=0.0),
            'backbone.stages.0.blocks.1.norm':
                dict(lr_mult=0.1, decay_mult=0.0),
            'backbone.stages.1.blocks.0.norm':
                dict(lr_mult=0.1, decay_mult=0.0),
            'backbone.stages.1.blocks.1.norm':
                dict(lr_mult=0.1, decay_mult=0.0),
            'backbone.stages.2.blocks.0.norm':
                dict(lr_mult=0.1, decay_mult=0.0),
            'backbone.stages.2.blocks.1.norm':
                dict(lr_mult=0.1, decay_mult=0.0),
            'backbone.stages.2.blocks.2.norm':
                dict(lr_mult=0.1, decay_mult=0.0),
            'backbone.stages.2.blocks.3.norm':
                dict(lr_mult=0.1, decay_mult=0.0),
            'backbone.stages.2.blocks.4.norm':
                dict(lr_mult=0.1, decay_mult=0.0),
            'backbone.stages.2.blocks.5.norm':
                dict(lr_mult=0.1, decay_mult=0.0),
            'backbone.stages.3.blocks.0.norm':
                dict(lr_mult=0.1, decay_mult=0.0),
            'backbone.stages.3.blocks.1.norm':
                dict(lr_mult=0.1, decay_mult=0.0),
            'backbone.stages.0.downsample.norm':
                dict(lr_mult=0.1, decay_mult=0.0),
            'backbone.stages.1.downsample.norm':
                dict(lr_mult=0.1, decay_mult=0.0),
            'backbone.stages.2.downsample.norm':
                dict(lr_mult=0.1, decay_mult=0.0),
            'backbone.stages.2.blocks.6.norm':
                dict(lr_mult=0.1, decay_mult=0.0),
            'backbone.stages.2.blocks.7.norm':
                dict(lr_mult=0.1, decay_mult=0.0),
            'backbone.stages.2.blocks.8.norm':
                dict(lr_mult=0.1, decay_mult=0.0),
            'backbone.stages.2.blocks.9.norm':
                dict(lr_mult=0.1, decay_mult=0.0),
            'backbone.stages.2.blocks.10.norm':
                dict(lr_mult=0.1, decay_mult=0.0),
            'backbone.stages.2.blocks.11.norm':
                dict(lr_mult=0.1, decay_mult=0.0),
            'backbone.stages.2.blocks.12.norm':
                dict(lr_mult=0.1, decay_mult=0.0),
            'backbone.stages.2.blocks.13.norm':
                dict(lr_mult=0.1, decay_mult=0.0),
            'backbone.stages.2.blocks.14.norm':
                dict(lr_mult=0.1, decay_mult=0.0),
            'backbone.stages.2.blocks.15.norm':
                dict(lr_mult=0.1, decay_mult=0.0),
            'backbone.stages.2.blocks.16.norm':
                dict(lr_mult=0.1, decay_mult=0.0),
            'backbone.stages.2.blocks.17.norm':
                dict(lr_mult=0.1, decay_mult=0.0)
        }),
        norm_decay_mult=0.0))
param_scheduler = [
    dict(
        type='PolyLR',
        eta_min=0,
        power=0.9,
        begin=0,
        end=90000,
        by_epoch=False)
]
train_cfg = dict(type='IterBasedTrainLoop', max_iters=90000, val_interval=5000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook', by_epoch=False, interval=5000,
        save_best='mIoU'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'))
auto_scale_lr = dict(enable=False, base_batch_size=16)
pretrained = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/swin/swin_small_patch4_window7_224_20220317-7ba6d6dd.pth'
depths = [2, 2, 18, 2]
backbone_norm_multi = dict(lr_mult=0.1, decay_mult=0.0)
backbone_embed_multi = dict(lr_mult=0.1, decay_mult=0.0)
custom_keys = dict({
    'backbone':
        dict(lr_mult=0.1, decay_mult=1.0),
    'backbone.patch_embed.norm':
        dict(lr_mult=0.1, decay_mult=0.0),
    'backbone.norm':
        dict(lr_mult=0.1, decay_mult=0.0),
    'absolute_pos_embed':
        dict(lr_mult=0.1, decay_mult=0.0),
    'relative_position_bias_table':
        dict(lr_mult=0.1, decay_mult=0.0),
    'query_embed':
        dict(lr_mult=1.0, decay_mult=0.0),
    'query_feat':
        dict(lr_mult=1.0, decay_mult=0.0),
    'level_embed':
        dict(lr_mult=1.0, decay_mult=0.0),
    'backbone.stages.0.blocks.0.norm':
        dict(lr_mult=0.1, decay_mult=0.0),
    'backbone.stages.0.blocks.1.norm':
        dict(lr_mult=0.1, decay_mult=0.0),
    'backbone.stages.1.blocks.0.norm':
        dict(lr_mult=0.1, decay_mult=0.0),
    'backbone.stages.1.blocks.1.norm':
        dict(lr_mult=0.1, decay_mult=0.0),
    'backbone.stages.2.blocks.0.norm':
        dict(lr_mult=0.1, decay_mult=0.0),
    'backbone.stages.2.blocks.1.norm':
        dict(lr_mult=0.1, decay_mult=0.0),
    'backbone.stages.2.blocks.2.norm':
        dict(lr_mult=0.1, decay_mult=0.0),
    'backbone.stages.2.blocks.3.norm':
        dict(lr_mult=0.1, decay_mult=0.0),
    'backbone.stages.2.blocks.4.norm':
        dict(lr_mult=0.1, decay_mult=0.0),
    'backbone.stages.2.blocks.5.norm':
        dict(lr_mult=0.1, decay_mult=0.0),
    'backbone.stages.3.blocks.0.norm':
        dict(lr_mult=0.1, decay_mult=0.0),
    'backbone.stages.3.blocks.1.norm':
        dict(lr_mult=0.1, decay_mult=0.0),
    'backbone.stages.0.downsample.norm':
        dict(lr_mult=0.1, decay_mult=0.0),
    'backbone.stages.1.downsample.norm':
        dict(lr_mult=0.1, decay_mult=0.0),
    'backbone.stages.2.downsample.norm':
        dict(lr_mult=0.1, decay_mult=0.0),
    'backbone.stages.2.blocks.6.norm':
        dict(lr_mult=0.1, decay_mult=0.0),
    'backbone.stages.2.blocks.7.norm':
        dict(lr_mult=0.1, decay_mult=0.0),
    'backbone.stages.2.blocks.8.norm':
        dict(lr_mult=0.1, decay_mult=0.0),
    'backbone.stages.2.blocks.9.norm':
        dict(lr_mult=0.1, decay_mult=0.0),
    'backbone.stages.2.blocks.10.norm':
        dict(lr_mult=0.1, decay_mult=0.0),
    'backbone.stages.2.blocks.11.norm':
        dict(lr_mult=0.1, decay_mult=0.0),
    'backbone.stages.2.blocks.12.norm':
        dict(lr_mult=0.1, decay_mult=0.0),
    'backbone.stages.2.blocks.13.norm':
        dict(lr_mult=0.1, decay_mult=0.0),
    'backbone.stages.2.blocks.14.norm':
        dict(lr_mult=0.1, decay_mult=0.0),
    'backbone.stages.2.blocks.15.norm':
        dict(lr_mult=0.1, decay_mult=0.0),
    'backbone.stages.2.blocks.16.norm':
        dict(lr_mult=0.1, decay_mult=0.0),
    'backbone.stages.2.blocks.17.norm':
        dict(lr_mult=0.1, decay_mult=0.0)
})
