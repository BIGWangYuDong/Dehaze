model = dict(type='Saliency_Net_inair2uw')
dataset_type = 'AlignedDataset'

data_root_train = '/home/dong/python-project/Dehaze/DATA/Train/'                  # data root, default = DATA
data_root_test = '/home/dong/python-project/Dehaze/DATA/Test/'
train_ann_file_path = 'train.txt'        # txt file for loading images, default = train.txt
val_ann_file_path = 'test.txt'          # txt file for loading images (validate during training process), default = test.txt
test_ann_file_path = 'test.txt'         # txt file for loading images, default = test.txt


img_norm_cfg = dict(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
train_pipeline = [dict(type='LoadImageFromFile', gt_type='color'),
                  dict(type='Resize', img_scale=(256,256), keep_ratio=True),
                  dict(type='RandomCrop', img_scale=(224,224)),
                  dict(type='RandomFlip', flip_ratio=0.5),
                  dict(type='Pad', size_divisor=32, mode='resize'),
                  dict(type='ImageToTensor'),
                  dict(type='Normalize', **img_norm_cfg),
                  ]
test_pipeling = [dict(type='LoadImageFromFile'),]
data = dict(
    samples_per_gpu=4,                                  # batch size, default = 4
    workers_per_gpu=4,                                  # multi process, default = 4, debug uses 0
    val_samples_per_gpu=1,                              # validate batch size, default = 1
    val_workers_per_gpu=4,                              # validate multi process, default = 4
    train=dict(                                         # load data in training process, debug uses 0
        type=dataset_type,
        ann_file=data_root_train + train_ann_file_path,
        img_prefix=data_root_train + 'train/',
        gt_prefix=data_root_train + 'gt/',
        pipeline=train_pipeline),
    val=dict(                                           # load data in validate process
        type=dataset_type,
        ann_file=data_root_test + val_ann_file_path,
        img_prefix=data_root_test + 'train/',
        gt_prefix=data_root_test + 'gt/',
        pipeline=test_pipeling),
    test=dict(                                          # load data in test process
        type=dataset_type,
        ann_file=data_root_test + test_ann_file_path,
        img_prefix=data_root_test + 'train/',
        gt_prefix=data_root_test + 'gt/',
        pipeline=test_pipeling))

train_cfg = dict(train_backbone=True)
loss = dict(
    loss_1=dict(type='CrossEntropyLoss', loss_weight=1.0),
    loss_2=dict(type='L1Loss', loss_weight=1.0))

test_cfg = dict(metrics=['SSIM', 'MSE', 'PSNR'])

optimizer = dict(type='Adam', lr=0.0001, betas=[0.5, 0.999])    # optimizer with type, learning rate, and betas.
lr_config = dict()                                              # learning rate change method
log_config = dict(
    interval=1,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        dict(type='TensorboardLoggerHook'),
        dict(type='VisdomLoggerHook')
    ])

total_epoch = 20
total_iters = None                      # epoch before iters,
work_dir = './checkpoints/shortcut'     #
load_from = None                        # only load network parameters
resume_from = None                      # resume training
save_freq_iters = 500                   # saving frequent (saving every XX iters)
save_freq_epoch = 1                     # saving frequent (saving every XX epoch(s))
log_level = 'INFO'                      # The level of logging.