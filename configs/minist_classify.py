import os

setup = dict(
    do_train=True,
    do_predict=True,
    tensorboard=False,
    device="cpu",
    max_checkpoints=3,
    checkpoint_path=os.path.join("checkpoints", "minist_classify"),
    log_every_n_step=200,
    save_ckpt_n_step=2000,
)
task = dict(name="MinistClassify")
data = dict(
    data_root="http://data-rainy.oss-cn-beijing.aliyuncs.com/data/exp1-data.zip"
)
train = dict(
    lr=0.0005,
    batch_size=32,
    epochs=10,
    accumulate_step=1,
    init_checkpoint=None,
    max_step=None,
)
predict = dict(batch_size=32, output_root="predictions")
model = dict()
