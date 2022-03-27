import argparse
from email.policy import default


class Config:
    
    def __init__(self):
        self.parse = argparse.ArgumentParser(description='Model hyper parameters and configs')
        
        self.parse.add_argument("--data_root", type='str', default='Data/', help='path where training/evalution data are stored')
        self.parse.add_argument("--checkpoint_path", type='str', default='Checkpoints/', help='path to load/stroe checkpoints')
        self.parse.add_argument("--init_checkpoint", type='str', default='init form specified checkpoint')
        self.parse.add_argument("--lr", type=float, default=0.00005, help='learning rate')
        self.parse.add_argument("--batch_size", type=int, default=32, help='batch size')
        self.parse.add_argument("--epochs", type=int, default=5, help="epochs num")
        self.parse.add_argument("--accumulate_step", type=int, default=1, help='step of accumulated gradient')
        self.parse.add_argument("--log_every_n_step", type=int, default=200, help="show log message every n step")
        self.parse.add_argument("--save_ckpt_n_step", type=int, default=2000, help="save checkpoints every n step")
        self.parse.add_argument("--max_train_step", type=int, default=None, help="max step of training loop")
        