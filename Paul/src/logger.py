# src/logger.py
import wandb
import os

def init_wandb(config: dict, project: str = 'isic-2024', name: str = None):
    wandb.login(key=os.environ['WANDB_API_KEY'])
    wandb.init(
        project = project,
        name    = name,
        config  = config
    )

def log_metrics(metrics: dict, step: int = None):
    wandb.log(metrics, step=step)

def finish_wandb():
    wandb.finish()