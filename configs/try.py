model = ''
dataset_type = ''

data_root = ''                  # data root, default = DATA
train_ann_file_path = ''        # txt file for loading images, default = train.txt
val_ann_file_path = ''          # txt file for loading images (validate during training process), default = test.txt
test_ann_file_path = ''         # txt file for loading images, default = test.txt

img_norm_cfg = ''

train_pipeline = []
test_pipeling = []
data = dict(
    samples_per_gpu=4,                                  # batch size, default = 4
    workers_per_gpu=4,                                  # multi process, default = 4
    val_samples_per_gpu=1,                              # validate batch size, default = 1
    val_workers_per_gpu=4,                              # validate multi process, default = 4
    train=dict(                                         # load data in training process
        type=dataset_type,
        ann_file=data_root + train_ann_file_path,
        pipeline=train_pipeline),
    val=dict(                                           # load data in validate process
        type=dataset_type,
        ann_file=data_root + val_ann_file_path,
        pipeline=test_pipeling),
    test=dict(                                          # load data in test process
        type=dataset_type,
        ann_file=data_root + test_ann_file_path,
        pipeline=test_pipeling))

train_cfg = dict(train_backbone=True)
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