"""Model saver"""
# pylint: disable=consider-using-f-string
# pylint: disable=logging-fstring-interpolation
import os
import shutil
import logging
from pathlib import Path
import torch

logger = logging.getLogger("Saver")


class Saver:
    """Model Saver, Managing saving and loading of models"""

    def __init__(self, model, config):
        self.config = config.setup
        self.task_name = config.task.name
        self.model = model
        self.metric = {}
        self._init_metric()

    def _init_metric(self):
        """init metric from checkpoint path"""
        save_path = self.config.checkpoint_path
        max_ckpts = self.config.max_checkpoints
        files = [
            file for file in os.listdir(save_path) if file.endswith(self.task_name)
        ]
        for file in files:
            self.metric[float(file[:6])] = os.path.join(save_path, file)
        if len(files) > max_ckpts:
            logger.error(
                "detected {} checkpoints in {}, but max checkpoints is {}".format(
                    len(files), save_path, max_ckpts
                )
            )
            value = sorted(self.metric.keys())
            for key_to_del in value[: len(files) - max_ckpts]:
                shutil.rmtree(self.metric[key_to_del])
                logger.error(
                    "delete checkpoint {}, remain checkpoint : {}".format(
                        self.metric[key_to_del], len(self.metric) - 1
                    )
                )
                del self.metric[key_to_del]
        else:
            logger.info(
                "detected {} checkpoints in checkpoint {}".format(len(files), save_path)
            )

    def _save(self, save_path):
        """Save model to save path"""
        os.makedirs(save_path)
        state_dict_path = os.path.join(save_path, "state_dict.pth")
        model_path = os.path.join(save_path, "net.model")
        torch.save(self.model.state_dict(), state_dict_path)
        torch.save(self.model.state_dict(), model_path)
        max_metric = max(self.metric.keys())
        logger.info(f"model saved to {save_path}, best metric : {max_metric}")

    def save_model(self, accuracy):
        """Save model according to accuracy.

        The saver maintain a metric dict to save `max_checkpoints` checkpoints
        with the highest accuracy.
        If the given accuracy is better than current metric, the saver will remove
        an old checkpoint, and save the current model.
        Otherwise, the saver will do nothing.

        Args:
            accuracy (float): the accuracy of current model.
        """
        save_name = f"{accuracy:.4f}-{self.task_name}"
        save_path = os.path.join(self.config.checkpoint_path, save_name)
        if accuracy in self.metric:
            logger.info(f"current accuracy {accuracy} has been recorded, skiped!")
            return
        if len(self.metric) < self.config.max_checkpoints:
            self.metric[accuracy] = save_path
            self._save(save_path)
        else:
            min_acc = min(self.metric.keys())
            if min_acc >= accuracy:
                logger.info(f"current accuracy {accuracy}, skiped!")
                return
            shutil.rmtree(self.metric[min_acc])
            del self.metric[min_acc]
            self.save_model(accuracy)

    def load_best_model(self):
        """Load best model from checkpoint dir"""
        assert len(self.metric) > 0, "no model been saved!"
        best_acc = max(self.metric.keys())
        file_path = os.path.join(self.metric[best_acc], "state_dict.pth")
        state_dict = torch.load(file_path)
        self.model.load_state_dict(state_dict)
        logger.info(f"load best model from {self.metric[best_acc]}")

    def resume_from_file(self, file_path):
        """Resume from checkpoint file

        Args:
            file_path (str or Path): checkpoint file path, dir or file,
                             if file, endswith `.pth` or `.model`.

        Raises:
            RuntimeError: checkpoint is not avaliable.
        """
        assert isinstance(file_path, (str, Path))
        if isinstance(file_path, str):
            file_path = Path(file_path)

        assert file_path.exists(), f"{str(file_path)} no exists"

        if file_path.suffix == ".pth":
            state_dict = torch.load(file_path)
            self.model.load_state_dict(state_dict)
            logger.info(f"resume model from {file_path}")
            return

        if file_path.is_dir():
            files = os.listdir(file_path)
            for file in files:
                file = file_path.joinpath(file)
                if file.suffix == ".pth":
                    self.resume_from_file(file)
                    return
        raise RuntimeError(f"resume file {file_path} not avaliable!")
