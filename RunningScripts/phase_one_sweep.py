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
    "name": "Phase 1 Sweep",
    "method": "grid",
    "metric": {
        "goal": "maximize",
        "name": "AUC.test.max"
    },
    "parameters": {
        "architecture": {"values": ["SpecialTransformer"]},
        "seed": {"values": list(range(1, 4))},
        "use_fc": {"values": [True, False]},
        "use_positional_encoding": {"values": [True, False]},
        "use_residuals": {"values": [True, False]},
        "features": {"values": ["EFs", "BERT_TOKEN", "GPT2_TOKEN", "ROBERTA_TOKEN"]},
        "total_epochs": {"values": [10]}
    },
    "command": command
}

# Initialize a new sweep
sweep_id = wandb.sweep(sweep=sweep_config, project=project)
print("run this line to run your agent in a screen:")
print(f"screen -dmS \"sweep_agent\" wandb agent {YOUR_WANDB_USERNAME}/{project}/{sweep_id}")