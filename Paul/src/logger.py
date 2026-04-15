# src/logger.py
import os
from typing import Optional

import wandb

def init_wandb(config: dict, project: str = 'isic-2024', name: Optional[str] = None):
    api_key = os.getenv('WANDB_API_KEY')
    init_kwargs = {
        'project': project,
        'name': name,
        'config': config,
    }

    if api_key:
        try:
            wandb.login(key=api_key)
        except Exception as exc:
            print(f"[wandb] login failed ({exc}). Running with W&B disabled.")
            init_kwargs['mode'] = 'disabled'
    else:
        print('[wandb] WANDB_API_KEY not set. Running with W&B disabled.')
        init_kwargs['mode'] = 'disabled'

    wandb.init(**init_kwargs)

def log_metrics(metrics: dict, step: Optional[int] = None):
    wandb.log(metrics, step=step)

def finish_wandb():
    wandb.finish()