import argparse
from pathlib import Path
from pprint import pprint

import torch

from src.callbacks import Callbacks, CheckpointSaver, Logger, TensorBoard
from src.runner import Runner
from src.utils import read_config, set_global_seeds


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    parser.add_argument('--fold', type=str, default=0)
    return parser.parse_args()


def create_callbacks(name, dumps):
    log_dir = Path(dumps['path']) / dumps['logs'] / name
    save_dir = Path(dumps['path']) / dumps['weights'] / name
    callbacks = Callbacks(
        [
            Logger(log_dir),
            CheckpointSaver(
                metric_name='DiceMetric',
                save_dir=save_dir,
                save_name='epoch_{epoch}.pth',
                num_checkpoints=4,
                mode='max'
            ),
            TensorBoard(str(log_dir))
        ]
    )
    return callbacks


def main():
    args = parse_args()
    set_global_seeds(666)
    config = read_config(args.config, stage="TRAIN")
    pprint(config)
    config['train_params']['name'] = f'{config["train_params"]["name"]}/fold{args.fold}'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    callbacks = create_callbacks(config['train_params']['name'], config['dumps'])
    trainer = Runner(stages=config['stages'], factory=factory, callbacks=callbacks, device=device)
    trainer.fit(data_factory)


if __name__ == '__main__':
    main()