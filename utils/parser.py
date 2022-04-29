"""System argument parser"""
import sys
import os
import argparse
from pathlib import Path
from torch.cuda import is_available
from utils.config import Config
from utils import get_logger
from utils.oss import OSS

# pylint: disable=logging-fstring-interpolation
logger = get_logger("Config")


class Parser:
    """System argument parser"""

    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument("--config", type=str, required=True)

        # general setup
        self.parser.add_argument("--do_train", action="store_true")
        self.parser.add_argument("--do_predict", action="store_true")
        self.parser.add_argument("--tensorboard", action="store_true")
        self.parser.add_argument("--cpu", action="store_true")
        self.parser.add_argument("--gpu", action="store_true")
        self.parser.add_argument("--max_checkpoints", type=int)
        self.parser.add_argument("--checkpoint_path", type=str)
        self.parser.add_argument("--log_every_n_step", type=int)
        self.parser.add_argument("--save_ckpt_n_step", type=int)

        # task config
        self.parser.add_argument("--task", type=str)

        # data config
        self.parser.add_argument("--data_root", type=str)

        # train config
        self.parser.add_argument("--lr", type=float)
        self.parser.add_argument("--train_batch_size", type=int)
        self.parser.add_argument("--epochs", type=int)
        self.parser.add_argument("--accumulate_step", type=int)
        self.parser.add_argument("--init_checkpoint", type=str)
        self.parser.add_argument("--train_max_step", type=int)

        # predict config
        self.parser.add_argument("--predict_batch_size", type=int)
        self.parser.add_argument("--output_root", type=str)

        self.args = vars(self.parser.parse_args())
        self.config = Config.from_file(self.args.get("config"))
        Parser.set_default(self.config)

        self._update_config()
        self._check_data_conifg()
        self._check_ckpt_config()

    def _update_config(self):
        """Update config according to input arguments. e.g.
        lr is 0.01 in config file `test_config.py`, and use the following command:
        python3 launch --config test_conifg.py --lr 0.05
        then lr become 0.05
        """
        updates = self.input_args

        # modify input args, check device option
        # device = cpu will be set in default, so we don't set here
        if "gpu" in updates and "cpu" in updates:
            updates["device"] = "cuda" if is_available() else "cpu"

        if "gpu" in updates and not is_available():
            logger.error("torch detected cuda is not avaliable, switched to cpu")
            updates["device"] = "cpu"
        elif "gpu" in updates:
            updates["device"] = "cuda"

        # update config
        for config_type, config in self.config.items():
            for update_key, update_value in updates.items():
                # train_batch_size refer to train/batch_size
                # transfer to batch_size so that we can update config
                if update_key.startswith(config_type):
                    update_key = update_key[len(config_type) + 1 :]
                if update_key in config:
                    config[update_key] = update_value

    def _check_data_conifg(self):
        """Check data config, make dirs and download data if needed"""
        data_root = self.config.data.data_root
        if data_root.startswith("http://"):
            try:
                local_dir = Path("remote-data/").absolute()
                download_path = local_dir.joinpath(OSS.basename(data_root, ""))
                extract_path = local_dir.joinpath(OSS.basename(data_root, ".zip"))

                if extract_path.exists():
                    logger.info(f"{str(extract_path)} file exists!")
                else:
                    if not OSS.is_download(data_root, local_dir):
                        logger.info(f"fetching {data_root} to local remote-data/ ...")
                        OSS.download(data_root, local_dir)
                    logger.info(
                        f"unziping {str(download_path)} to {str(extract_path)} ..."
                    )
                    OSS.extract(download_path, extract_path, remove=True)
                    logger.info(f"unzip succed! remove {str(download_path)}")

                # set config's data_root to new extracted data path
                self.config.data.data_root = str(extract_path)
            except Exception:
                logger.error(f"fetching {data_root} failed!")
                raise RuntimeError("fetching file failed")
        # check local data root
        if not os.path.isdir(self.config.data.data_root):
            raise ValueError(
                f"data root {self.config.data.data_root} should be a directory"
            )

    def _check_ckpt_config(self):
        """Check checkpoint config, make dirs if needed"""
        init_ckpt = self.config.train.init_checkpoint
        if init_ckpt is not None:
            if not os.path.isdir(init_ckpt) and not os.path.isfile(init_ckpt):
                raise ValueError(f"init checkpoint {init_ckpt} is not avaliable!")

        checkpoint_path = self.config.setup.checkpoint_path
        if not os.path.isdir(checkpoint_path):
            os.makedirs(checkpoint_path)
            logger.warning(
                f"checkpoint path {checkpoint_path} is not exists, create automatically!"
            )

    @property
    def input_args(self):
        """Get a map including all input arguments.
        For example, result is {'lr': 0.1, 'epochs':5} for following command:
        `python3 launch --lr 0.1 --epochs 5`

        Returns:
            Dict: as describe above.
        """
        input_options = [opt for opt in sys.argv if opt.startswith("--")]
        input_options = [opt[2:] for opt in input_options]
        input_options = {key: self.args[key] for key in input_options}
        return input_options

    @staticmethod
    def check_config(config):
        """Check config"""
        assert isinstance(config, (Config, dict))
        required = {
            "setup": ["do_train", "do_predict"],
            "task": ["name"],
            "model": [],
            "data": ["data_root"],
            "train": ["lr", "batch_size", "epochs"],
            "predict": [],
        }
        for key, value in required.items():
            if key not in config:
                raise KeyError(f"{key} config should be set in config file")
            for attr_required in value:
                if attr_required not in config[key]:
                    raise KeyError(
                        f"{key}.{attr_required} should be set in config file"
                    )

    @staticmethod
    def set_default(config):
        """Set default value to conifg"""
        assert isinstance(config, (Config, dict))
        Parser.check_config(config)

        default = {
            "setup": {
                "tensorboard": False,
                "device": "cuda" if is_available() else "cpu",
                "max_checkpoints": 3,
                "checkpoint_path": os.path.join("ckpts", config.task.name),
                "log_every_n_step": 200,
                "save_ckpt_n_step": 2000,
            },
            "predict": {
                "batch_size": config.train.batch_size,
                "output_root": "predictions",
            },
            "train": {"accumulate_step": 1, "init_checkpoint": None, "max_step": None},
        }
        for cfg_key, cfg_value in default.items():
            config_toset = config[cfg_key]
            for default_k, default_v in cfg_value.items():
                if default_k not in config_toset:
                    setattr(config_toset, default_k, default_v)
