# Rainy-ML

## 使用说明

* 支持命令行训练

  ```shell
  python3 launch.py --task ImageClassify --data_root Dataset/ --checkpoint_path Checkpoints/ImageClassify/
  ```

* 支持从config文件训练（目前仅支持yaml），配置文件参考configs/ImageClassify.yaml

  ```shell
  python3 launch.py --config configs/ImageClassify.yaml
  ```

* 同时支持以文件为主，参数微调

  ```shell
  python3 launch.py --config configs/ImageClassify.yaml --max_train_step 50000
  ```

  

## 训练参数说明

```shell
# 必须填写
--task                   # 训练所选择的task
--data_root		# 数据集文件夹
--do_train              # 进行训练，默认为True
--do_predict            # 进行预测推理，默认为False

# 建议修改
--lr			# 学习率，默认为0.00005
--epochs		# 训练轮数，默认为5
--train_batch_size	# 训练数据batch size，默认为32
--eval_batch_size	# valid数据btach size，默认与train相同
--checkpoint_path	# 断点保存所在文件夹，默认为Checkpoints/

# 优化选项
--accumulate_step	# 梯度累加部署，默认为1(表示不开启)
--max_train_step	# 最多训练的步数，默认为None，由epoch和数据集决定
--tensorboard		# 使用tensorboard，默认输出为当前目录下runs/
--init_checkpoint	# 从指定文件夹或者文件初始化模型，默认为None

# 建议保持不变
--max_checkpoints	# 最多保存的断点数量，默认为3
--log_every_n_step	# 每隔多少步输出log信息，默认200step
--save_ckpt_n_step	# 每隔多少步做一次validation，保存断点，默认2000
--config                # 从config文件中读取配置

```



## TODO

- [ ] 灵活支持GPU，CPU
- [ ] 支持多卡训练
- [ ] 添加单元测试，CI集成
- [ ] 丰富模型，边学习边实践
- [ ] 保存断点重构，更灵活