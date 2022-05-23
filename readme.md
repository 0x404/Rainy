<div id="top"></div>

<br />
<div align="center">
  <a href="https://github.com/0x404/rainy">
    <img src="http://image-hosting-404.oss-cn-beijing.aliyuncs.com/img/rainy%20(2).png" alt="Logo" width="128" height="128">
  </a>

  <h3 align="center">Rainy</h3>

  <p align="center">
    Learn deepLearning framework
    <br />
    <br />
    <a href="www.0x404.cn">View Demo</a>
    ·
    <a href="https://github.com/0x404/rainy/issues">Report Bug</a>
    ·
    <a href="https://github.com/0x404/rainy/issues">Request Feature</a>
  </p>
</div>

## 目录
   * [使用说明](#使用说明)
   * [训练参数说明](#训练参数说明)
   * [依赖安装](#依赖安装)
   * [Docker](#docker)
   * [TODO](#todo)


## 使用说明

* 支持从config文件训练（目前支持yaml和python文件），配置文件参考configs/iamge_classify.py

  ```shell
  # 使用python文件作为config
  python3 launch.py --config configs/iamge_classify.py

  # 使用yaml文件作为config
  python3 launch.py --config configs/relation_extract.yaml
  ```

* 支持以配置文件为主，参数微调（推荐）

  ```shell
  python3 launch.py --config configs/iamge_classify.py --train_max_step 50000 --lr 0.001
  ```

* 支持参数命令训练

  ```shell
  python3 launch.py --task ImageClassify --data_root Dataset/ --checkpoint_path Checkpoints/ImageClassify/
  ```  

## 训练参数说明

```python
# setup 基本参数
setup = dict(
    do_train=True,          # 是否训练
    do_predict=True,        # 是否预测
    tensorboard=False,      # 是否开启tensorboard
    device="cpu",           # 在哪个设备上训练，cuda或者cpu
    max_checkpoints=3,      # 最多保存的断点个数
    checkpoint_path=os.path.join("checkpoints", "relation_exract"), # 断点保存位置
    log_every_n_step=200,   # 每多少步输出一次log信息
    save_ckpt_n_step=2000,  # 每多少步进行一次validation并保存断点
)
# task 任务相关参数
task = dict(
  name="RelationExtract",   # 所使用的task
)
# data 数据集相关参数
data = dict(
    # 数据集位置，可以是本地路径也可以是url远程路径（目前仅支持.zip文件）
    data_root="http://data-rainy.oss-cn-beijing.aliyuncs.com/data/exp3-data.zip"
)
# train 训练相关参数
train = dict(
    lr=0.0005,              # 学习率
    batch_size=32,          # 批次大小
    epochs=80,              # epoch轮数
    accumulate_step=1,      # 梯度累加步数，1相当于不使用
    init_checkpoint=None,   # 从哪个checkpoint初始化，默认不初始化
    max_step=None,          # 最多训练多少步，为None则按epochs算
)
# predict 预测相关参数
predict = dict(
    batch_size=32,              # 批次大小            
    output_root="predictions",  # 输出位置
)
#model 网络模型相关参数
model = dict()

```

## 依赖安装

建议使用docker进行训练，见[docker](#docker)

本地环境，请使用如下命令安装所需依赖：

```shell
pip3 install -r docker/requirements.txt
```

## Docker

创建镜像：

```shell
docker build --file docker/Dockerfile -t rainy .
```

创建容器，在容器内训练：

```shell
# start container
docker run -it --name ml-rainy rainy

# do your train pipeline
python3 launch.py --config configs/iamge_classify.py

```

## TODO

- [x] 灵活支持GPU，CPU
- [x] 支持本地数据集和远程数据集
- [x] 支持python文件作为config
- [x] 添加单元测试，CI集成
- [ ] 重构runner，支持各类hook，方便用户定制操作
- [ ] 重构logger，定义多类logger，简化log输出
- [ ] 支持将训练的结果（checkpoints，tensorboard，predictions，runlog）打包上传到指定路径
- [ ] 丰富模型，边学习边实践
- [ ] 支持多卡训练
