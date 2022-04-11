"""runner for training loop"""
import torch
import task
from utils import get_logger, Saver
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


logger = get_logger(__name__)


class Runner:
    def __init__(self, config):
        self.config = config
        self.task = getattr(task, config.task, None)
        if self.task is None:
            raise ValueError(f"task {config.task} is not supported!")
        self.task = self.task(config)
        self.model = self.task.model
        self.train_dataset = self.task.train_dataset
        self.valid_dataset = self.task.valid_dataset
        self.pred_dataset = self.task.pred_dataset
        self.model_saver = Saver(self.model, config)
        self.device = torch.device("cuda" if config.gpu is not None else "cpu")
        logger.info(f"training on {self.device}")
        if config.init_checkpoint is not None:
            self.model_saver.resume_from_file(config.init_checkpoint)
        self.writer = None
        if config.tensorboard:
            self.writer = SummaryWriter()

    def train(self):
        """Do training loop.

        The Runner fetch model, optimizer and loos_function from task config,
        and do the training loop, evaluation, logging and summary write.
        """
        config = self.config
        train_loader = DataLoader(
            self.train_dataset, batch_size=self.config.train_batch_size, shuffle=True
        )
        model = self.model
        optimizer = self.task.optimizer
        optimizer.zero_grad()
        loss_function = self.task.loss_function
        lr_scheduler = getattr(self.task, "lr_scheduler", None)

        total_step = self.config.max_train_step
        if total_step is None:
            total_step = self.config.epochs * len(train_loader)
        total_step = min(total_step, self.config.epochs * len(train_loader))

        logger.info("********** Running training **********")
        logger.info(f"  Num Examples = {len(train_loader)}")
        logger.info(f"  Num Epochs = {self.config.epochs}")
        logger.info(f"  Global Total Step = {total_step}")
        logger.info(f"  Train Batch Size = {self.config.train_batch_size}")
        logger.info(f"  Accumulate Gradient Step = {self.config.accumulate_step}")
        logger.info(f"  Model Structure = {model}")

        completed_step = 0
        for epoch in range(config.epochs):
            for step, batch_data in enumerate(train_loader):
                inputs, labels = batch_data
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                if epoch == 0 and step == 0:
                    logger.info(f"Input Shape: {inputs.shape}")
                    logger.info(f"Input Dtype: {inputs.dtype}")
                    logger.info(f"Labels Shape: {labels.shape}")
                    logger.info(f"Labels Dtype: {labels.dtype}")

                outputs = model(inputs)
                loss = loss_function(outputs, labels)
                loss.backward()

                if step % config.accumulate_step == 0 or step == len(train_loader) - 1:
                    optimizer.step()
                    optimizer.zero_grad()
                    completed_step += 1
                    if self.writer is not None:
                        self.writer.add_scalar("loss", loss.item(), completed_step)

                if (
                    completed_step % config.log_every_n_step == 0
                    or completed_step == total_step
                ):
                    if completed_step == 0:
                        continue
                    logger.info(
                        f"[{100 * completed_step / total_step:.2f}%]\tepoch:{epoch}\tstep:{completed_step}\tloss:{loss.item()}"
                    )

                if completed_step % config.save_ckpt_n_step == 0:
                    if completed_step == 0:
                        continue
                    accuracy = self.eval(is_training=True)
                    self.model_saver.save_model(accuracy)
                    if self.writer is not None:
                        self.writer.add_scalar("accuracy", accuracy, completed_step)

                if completed_step >= total_step:
                    logger.info(
                        f"reach max training step {total_step}, breaking from training loop."
                    )
                    if self.writer is not None:
                        self.writer.flush()
                        self.writer.close()
                    break

            if completed_step >= total_step:
                break
            if lr_scheduler is not None:
                lr_scheduler.step()

    def eval(self, is_training=False):
        """Evaluation

        Args:
            is_training (bool, optional): whther is training. Defaults to False.

        Returns:
            float: the accuracy of this evaluation.
        """
        eval_loader = DataLoader(
            self.valid_dataset, batch_size=self.config.eval_batch_size
        )
        eval_type = "Evalution" if is_training else "Prediction"
        logger.info(f"********** Running {eval_type} **********")
        logger.info(f"  Num Examples = {len(eval_loader)}")
        logger.info(f"  Batch Size = {eval_loader.batch_size}")
        total = 0
        correct = 0
        with torch.no_grad():
            for _, batch_data in enumerate(tqdm(eval_loader, desc=eval_type)):
                inputs, labels = batch_data
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(inputs)
                _, pred = torch.max(outputs, dim=1)
                batch_size = pred.shape[0]
                correct += sum(
                    [1 if pred[i] == labels[i] else 0 for i in range(batch_size)]
                )
                total += batch_size
        accuracy = correct / total
        logger.info(f"{eval_type} Finish!\tAccuracy:{accuracy * 100:.2f}%")
        return accuracy

    def predict(self):
        """Do prediction.

        Load best model from checkpoint and do evaluation,
        then run the test dataset and write predictions to `predict_path`.
        """
        self.model_saver.load_best_model()
        self.eval()
        pred_loader = DataLoader(self.pred_dataset, batch_size=1)
        predictions = list()
        with torch.no_grad():
            for data in tqdm(pred_loader, desc="test"):
                input, _ = data
                input = input.to(self.device)
                output = self.model(input)
                _, pred = torch.max(output, dim=1)
                predictions.append(pred.item())
        with open(self.config.predict_path, mode="w") as f:
            for pred in predictions:
                f.write(f"{pred}\n")
