import os
from pathlib import Path
import sys
import shutil
import torch
from utils import Config, OSS, Parser


class TestUtils:
    def test_config(self):
        result_dict = dict(
            a=dict(
                b="a string", c=dict(c_b="b_string", d=dict(d_b="d string", x=1, y=2))
            )
        )
        py_config = Config.from_file("test/example_config.py")
        yaml_config = Config.from_file("test/example_config.yaml")
        assert str(py_config) == str(result_dict)
        assert str(yaml_config) == str(result_dict)

    def test_oss(self):
        """Test OSS.
        Step1. test download.
        Step2. test is_download.
        Step3. test extract.
        """
        data_url = "http://data-rainy.oss-cn-beijing.aliyuncs.com/data/exp3-data.zip"
        local_data_path = Path("test/test-data/exp3-data.zip")
        extract_path = Path("test/test-data/exp3-data")

        # step1: test download
        if local_data_path.exists():
            os.remove(str(local_data_path))
        assert not local_data_path.exists()
        OSS.download(data_url, "test/test-data")
        assert local_data_path.exists()

        # step2: test is_download
        assert OSS.is_download(data_url, "test/test-data")

        # step3: test extract
        if extract_path.exists():
            shutil.rmtree(str(extract_path))
        OSS.extract(local_data_path)
        assert extract_path.exists()

        # step4: clear
        shutil.rmtree("test/test-data")

    def test_parser(self):
        """Test Parser.
        Step1. test set default.
        Step2. test parser system argument.
        """
        config = dict(
            setup=dict(do_train=False, do_predict=False),
            task=dict(name="MinistClassify"),
            data=dict(data_root="test/"),
            train=dict(lr=0.0005, batch_size=32, epochs=10),
            predict=dict(),
            model=dict(),
        )
        config = Config(config)
        expected = dict(
            setup=dict(
                do_train=False,
                do_predict=False,
                tensorboard=False,
                device="cuda" if torch.cuda.is_available() else "cpu",
                max_checkpoints=3,
                checkpoint_path=os.path.join("ckpts", "MinistClassify"),
                log_every_n_step=200,
                save_ckpt_n_step=2000,
            ),
            task=dict(name="MinistClassify"),
            data=dict(data_root="test/"),
            train=dict(
                lr=0.0005,
                batch_size=32,
                epochs=10,
                accumulate_step=1,
                init_checkpoint=None,
                max_step=None,
            ),
            predict=dict(batch_size=32, output_root="predictions"),
            model=dict(),
        )

        # test set default
        Parser.set_default(config)
        assert str(config) == str(expected)

        # test parse sys argument
        sys.argv = [
            "test_utils.py",
            "--config",
            "configs/test.py",
            "--do_train",
            "--do_predict",
            "--gpu",
        ]
        parser = Parser()
        assert parser.config.setup.do_train
        assert parser.config.setup.do_predict
        assert parser.config.setup.device == (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        sys.argv = [
            "test_utils.py",
            "--config",
            "configs/test.py",
            "--tensorboard",
            "--train_batch_size",
            "64",
            "--output_root",
            "output",
            "--train_max_step",
            "10",
        ]
        parser = Parser()
        assert not parser.config.setup.do_train
        assert not parser.config.setup.do_predict
        assert parser.config.setup.tensorboard
        assert parser.config.train.batch_size == 64
        assert parser.config.train.max_step == 10
        assert parser.config.predict.output_root == "output"
