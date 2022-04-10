import os
import yaml
import argparse
from utils import get_logger

logger = get_logger("Config")


class Config:
    def __init__(self):
        self.parser = argparse.ArgumentParser(
            description="Model hyper parameters and configs"
        )
        self.parser.add_argument(
            "--do_train", action="store_true", default=True, help="do training loop"
        )
        self.parser.add_argument(
            "--do_predict", action="store_true", default=False, help="do prediction"
        )
        self.parser.add_argument(
            "--predict_path",
            type=str,
            default="predictions.txt",
            help="prediction result path",
        )
        self.parser.add_argument(
            "--data_root",
            type=str,
            default="Data/",
            help="path where training/evalution data are stored",
        )
        self.parser.add_argument(
            "--checkpoint_path",
            type=str,
            default="Checkpoints/",
            help="path to load/stroe checkpoints",
        )
        self.parser.add_argument(
            "--max_checkpoints", type=int, default=3, help="max num of checkpoints"
        )
        self.parser.add_argument(
            "--init_checkpoint",
            type=str,
            default=None,
            help="init form specified checkpoint",
        )
        self.parser.add_argument(
            "--lr", type=float, default=0.00005, help="learning rate"
        )
        self.parser.add_argument(
            "--train_batch_size", type=int, default=32, help="train batch size"
        )
        self.parser.add_argument(
            "--eval_batch_size", type=int, default=None, help="eval batch size"
        )
        self.parser.add_argument("--epochs", type=int, default=5, help="epochs num")
        self.parser.add_argument(
            "--accumulate_step",
            type=int,
            default=1,
            help="step of accumulated gradient",
        )
        self.parser.add_argument(
            "--log_every_n_step",
            type=int,
            default=200,
            help="show log message every n step",
        )
        self.parser.add_argument(
            "--save_ckpt_n_step",
            type=int,
            default=2000,
            help="save checkpoints every n step",
        )
        self.parser.add_argument(
            "--max_train_step", type=int, default=None, help="max step of training loop"
        )
        self.parser.add_argument(
            "--task", type=str, default="ImageClassify", help="task name of training"
        )
        self.parser.add_argument(
            "--config", type=str, default=None, help="parser config from file"
        )
        self.parser.add_argument("--tensorboard", action="store_true", default=False)
        self.args = self.parser.parse_args()
        self.default = vars(self.parser.parse_args([]))
        self._check_args()

    def _load_config_file(self):
        """load config from file"""
        with open(self.args.config, mode="r") as f:
            config = yaml.safe_load(f)
        for key in config:
            if key not in self.default:
                config.pop(key)
        for key, val in self.default.items():
            if key not in config:
                config[key] = val
        return config

    def _check_args(self):
        """Check arguments' behavior is legal"""
        args = self.args
        default_config = None
        if args.config is not None:
            if not os.path.isfile(args.config):
                raise ValueError(f"config file {args.config} is not avaliable!")
            default_config = self._load_config_file()

        if default_config is not None:
            for key, val in default_config.items():
                if getattr(args, key) == self.default.get(key):
                    setattr(args, key, val)

        if not os.path.isdir(args.data_root):
            raise ValueError(f"data root {args.data_root} is not avaliable!")

        if args.init_checkpoint is not None:
            if not os.path.isdir(args.init_checkpoint) and not os.path.isfile(
                args.init_checkpoint
            ):
                raise ValueError(
                    f"init checkpoint {args.init_checkpoint} is not avaliable!"
                )

        if not os.path.isdir(args.checkpoint_path):
            os.makedirs(args.checkpoint_path)
            logger.warning(
                f"checkpoint path {args.checkpoint_path} is not exists, create automatically!"
            )

        if args.train_batch_size <= 0:
            raise ValueError(
                f"train batch size {args.train_batch_size} is supposed to be bigger than 0!"
            )
        if args.eval_batch_size is None:
            args.eval_batch_size = args.train_batch_size

        if args.lr <= 0:
            raise ValueError(
                f"learning rate {args.lr} is supposed to be bigger than 0!"
            )
        if args.epochs <= 0:
            raise ValueError(
                f"epoch num {args.epochs} is supposed to be bigger than 0!"
            )
        if args.accumulate_step <= 0:
            raise ValueError(
                f"accumulate step {args.accumulate_step} is supposed to be bigger than 0!"
            )
        if args.save_ckpt_n_step <= 0:
            raise ValueError(
                f"save checkpoint n step {args.save_ckpt_n_step} is supposed to be bigger than 0!"
            )
        if args.log_every_n_step <= 0:
            raise ValueError(
                f"log every n step {args.log_every_n_step} is supposed to be bigger than 0!"
            )
