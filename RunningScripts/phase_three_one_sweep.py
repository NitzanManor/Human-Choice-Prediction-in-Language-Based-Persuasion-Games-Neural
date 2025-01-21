import wandb
YOUR_WANDB_USERNAME = "nitzan-manor"
project = "NLP_project"

command = [
        "${ENVIRONMENT_VARIABLE}",
        "${interpreter}",
        "StrategyTransfer.py",
        "${project}",
        "${args}"
    ]

sweep_config = {
    "name": "Phase 3.1 Sweep",
    "method": "grid",
    "metric": {
        "goal": "maximize",
        "name": "AUC.test.max"
    },
    "parameters": {
        "architecture": {"values": ["SpecialTransformer"]},
        "seed": {"values": list(range(1, 6))},
        "use_fc": {"values": [False]},
        "use_positional_encoding": {"values": [False]},
        "use_residuals": {"values": [False]},
        "dropout": {"values": [0.1]},
        "features": {"values": ["EFs"]},
        "loss": {"values": ["CE"]},
        "total_epochs": {"values": [20]}
    },
    "command": command
}

# Initialize a new sweep
sweep_id = wandb.sweep(sweep=sweep_config, project=project)
print("run this line to run your agent in a screen:")
print(f"screen -dmS \"sweep_agent\" wandb agent {YOUR_WANDB_USERNAME}/{project}/{sweep_id}")