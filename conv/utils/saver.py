import os
import torch
import logging
import shutil

logger = logging.getLogger("Saver")


class Saver:
    """Model Saver, Managing saving and loading of models"""
    def __init__(self, model, config):
        self.config = config
        self.model = model
        self.metric = dict()
        self._init_metric()
    
    def _init_metric(self):
        """init metric from checkpoint path"""
        save_path = self.config.checkpoint_path
        max_ckpts = self.config.max_checkpoints
        task = self.config.task
        files = [file for file in os.listdir(save_path) if file.endswith(task)]
        for file in files:
            self.metric[float(file[:6])] = os.path.join(save_path, file)
        if len(files) > max_ckpts:
            logger.error(f"checkpoint path {save_path}, detected {len(files)} checkpoints, but max checkpoints is {max_ckpts}")
            value = sorted(self.metric.keys())
            for v in value[:len(files) - max_ckpts]:
                shutil.rmtree(self.metric[v])
                logger.error(f"delete checkpoint {self.metric[v]}, remain checkpoint : {len(self.metric) - 1}")
                del self.metric[v]
        
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
        save_name = f"{accuracy:.4f}-{self.config.task}"
        save_path = os.path.join(self.config.checkpoint_path, save_name)
        if len(self.metric) < self.config.max_checkpoints:
            self.metric[accuracy] = save_path
            self._save(save_path)
        else:
            min_acc = min(self.metric.keys())
            if min_acc >= accuracy:
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
            file_path (str): checkpoint file path, dir or file,
                             if file, endswith `.pth` or `.model`.

        Raises:
            RuntimeError: checkpoint is not avaliable.
        """
        if file_path.endswith(".pth"):
            assert os.path.isfile(file_path)
            state_dict = torch.load(file_path)
            self.model.load_state_dict(state_dict)
            logger.info(f"resume model from {file_path}")
            return
        if file_path.endswith(".model"):
            assert os.path.isfile(file_path)
            self.model = torch.load(file_path)
            logger.info(f"resume model from {file_path}")
            return
        if os.path.isdir(file_path):
            files = os.listdir(file_path)
            for file in files:
                if file.endswith(".pth") or file.endswith(".model"):
                    self.resume_from_file(os.path.join(file_path, file))
                    return
            raise RuntimeError(f"resume file {file_path} not avaliable!")
