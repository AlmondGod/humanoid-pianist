import os
import pickle
import numpy as np
import argparse
import jax
import jax.numpy as jnp
from pathlib import Path
import time
import random
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Any, Optional, Sequence
from dataclasses import dataclass
import json
import tyro
from tqdm import tqdm, trange  # Add this import at the top

import sys
sys.path.append('.')  # Add the root directory to path

from RLHF.cpl import CPL, CPLConfig

@dataclass
class Args:
    """Arguments for training CPL reward model."""
    # Dataset arguments
    data_dir: str = "RLHF/preference_data/2025-03-13-00-24-50"
    dataset: str = "cpl_dataset.pkl"
    
    # Training arguments
    output_dir: str = "RLHF/reward_models"
    num_epochs: int = 10000
    batch_size: int = 64
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    dropout_rate: float = 0.1
    conservative_weight: float = 0.0
    grad_clip: float = 1.0
    seed: int = 42
    eval_interval: int = 100
    save_interval: int = 500
    
    # Model architecture
    hidden_dims: str = "256,256,256"
    
    # Resume training
    resume: Optional[str] = None

def load_dataset(data_path: str):
    """Load CPL dataset."""
    print(f"Loading dataset from {data_path}")
    with open(data_path, 'rb') as f:
        dataset = pickle.load(f)
    
    print(f"Loaded {len(dataset)} preference pairs")
    
    # Get dimensions from the first example
    example = dataset[0]
    state_dim = example["chosen"]["observations"][0].shape[0]
    action_dim = example["chosen"]["actions"][0].shape[0]
    
    print(f"State dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")
    
    return dataset, state_dim, action_dim


def prepare_batch(dataset, batch_indices):
    """Prepare a batch of data for training."""
    chosen_states = []
    chosen_actions = []
    rejected_states = []
    rejected_actions = []
    
    for idx in batch_indices:
        pair = dataset[idx]
        
        # Get chosen trajectory
        chosen_states_traj = pair["chosen"]["observations"]
        chosen_actions_traj = pair["chosen"]["actions"]
        
        # Get rejected trajectory
        rejected_states_traj = pair["rejected"]["observations"]
        rejected_actions_traj = pair["rejected"]["actions"]
        
        # Sample a random state-action pair from each trajectory
        chosen_idx = np.random.randint(len(chosen_actions_traj))
        rejected_idx = np.random.randint(len(rejected_actions_traj))
        
        chosen_states.append(chosen_states_traj[chosen_idx])
        chosen_actions.append(chosen_actions_traj[chosen_idx])
        rejected_states.append(rejected_states_traj[rejected_idx])
        rejected_actions.append(rejected_actions_traj[rejected_idx])
    
    # Convert to arrays
    chosen_states = np.array(chosen_states)
    chosen_actions = np.array(chosen_actions)
    rejected_states = np.array(rejected_states)
    rejected_actions = np.array(rejected_actions)
    
    return chosen_states, chosen_actions, rejected_states, rejected_actions


def evaluate_model(model, dataset, num_samples=100):
    """Evaluate model on random samples from dataset."""
    # Sample random preference pairs
    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
    
    correct_predictions = 0
    avg_confidence = 0.0
    
    for idx in indices:
        pair = dataset[idx]
        
        # Get chosen trajectory
        chosen_states = pair["chosen"]["observations"]
        chosen_actions = pair["chosen"]["actions"]
        
        # Get rejected trajectory
        rejected_states = pair["rejected"]["observations"]
        rejected_actions = pair["rejected"]["actions"]
        
        # Compute preference probability
        prob = model.preference_probability(chosen_states, chosen_actions, rejected_states, rejected_actions)
        
        # Count correct predictions (prob > 0.5 means chosen is preferred)
        if prob > 0.5:
            correct_predictions += 1
        
        avg_confidence += prob if prob > 0.5 else (1 - prob)
    
    accuracy = correct_predictions / len(indices)
    avg_confidence = avg_confidence / len(indices)
    
    return {
        "eval_accuracy": accuracy,
        "eval_confidence": avg_confidence,
    }


def plot_training_curves(metrics_history, plots_dir):
    """Plot training curves."""
    plots_dir = Path(plots_dir)  # Ensure it's a Path object
    
    # Plot loss
    plt.figure(figsize=(10, 6))
    plt.plot(metrics_history["epoch"], metrics_history["loss"], label="Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(plots_dir / "loss.png")
    plt.close()
    
    # Plot accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(metrics_history["epoch"], metrics_history["accuracy"], label="Training Accuracy")
    if "eval_accuracy" in metrics_history:
        # Create a list of eval epochs and accuracies, filtering out None values
        eval_epochs = [e for e, acc in zip(metrics_history["epoch"], metrics_history["eval_accuracy"]) if acc is not None]
        eval_accuracies = [acc for acc in metrics_history["eval_accuracy"] if acc is not None]
        plt.plot(eval_epochs, eval_accuracies, label="Eval Accuracy", marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Model Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig(plots_dir / "accuracy.png")
    plt.close()
    
    # Plot reward difference
    plt.figure(figsize=(10, 6))
    plt.plot(metrics_history["epoch"], metrics_history["mean_reward_diff"], label="Mean Reward Difference")
    plt.xlabel("Epoch")
    plt.ylabel("Reward Difference")
    plt.title("Reward Difference between Chosen and Rejected")
    plt.legend()
    plt.grid(True)
    plt.savefig(plots_dir / "reward_diff.png")
    plt.close()


# Add this function to convert NumPy types to Python native types
def convert_to_json_serializable(obj):
    """Convert NumPy/JAX types to JSON serializable types."""
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    else:
        return obj


def main():
    args = tyro.cli(Args)
    
    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # Create timestamped output directory
    timestamp = time.strftime('%Y-%m-%d-%H-%M-%S')
    base_output_dir = Path(args.output_dir)
    output_dir = base_output_dir / timestamp
    
    # Create subdirectories
    checkpoints_dir = output_dir / "checkpoints"
    plots_dir = output_dir / "plots"
    logs_dir = output_dir / "logs"
    
    for dir_path in [checkpoints_dir, plots_dir, logs_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Save args
    with open(logs_dir / "args.json", 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # Load dataset
    data_path = Path(args.data_dir) / args.dataset
    dataset, state_dim, action_dim = load_dataset(str(data_path))
    
    # Create model config
    hidden_dims = [int(dim) for dim in args.hidden_dims.split(',')]
    config = CPLConfig(
        hidden_dims=hidden_dims,
        activation="gelu",
        dropout_rate=args.dropout_rate,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        conservative_weight=args.conservative_weight,
    )
    
    # Create model
    cpl_model = CPL(
        state_dim=state_dim,
        action_dim=action_dim,
        config=config,
        seed=args.seed,
    )
    
    # Load checkpoint if resuming
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        cpl_model.load_checkpoint(args.resume)
    
    # Initialize metrics history
    metrics_history = {
        "epoch": [],
        "loss": [],
        "accuracy": [],
        "mean_chosen_reward": [],
        "mean_rejected_reward": [],
        "mean_reward_diff": [],
        "eval_accuracy": [],
        "eval_confidence": [],
    }
    
    # Training loop
    print(f"Starting training for {args.num_epochs} epochs...")
    start_time = time.time()
    
    pbar = tqdm(total=args.num_epochs, desc="Training")
    for epoch in range(1, args.num_epochs + 1):
        epoch_start_time = time.time()
        
        # Create batches
        indices = np.arange(len(dataset))
        np.random.shuffle(indices)
        batches = [indices[i:i+args.batch_size] for i in range(0, len(indices), args.batch_size)]
        
        # Train on batches
        epoch_metrics = {
            "loss": [],
            "accuracy": [],
            "mean_chosen_reward": [],
            "mean_rejected_reward": [],
            "mean_reward_diff": [],
        }
        
        for batch_indices in batches:
            # Prepare batch data
            chosen_states, chosen_actions, rejected_states, rejected_actions = prepare_batch(dataset, batch_indices)
            
            # Update model
            metrics = cpl_model.update_step(chosen_states, chosen_actions, rejected_states, rejected_actions)
            
            # Record metrics
            for key, value in metrics.items():
                epoch_metrics[key].append(value)
        
        # Compute average metrics for epoch
        avg_metrics = {k: np.mean(v) for k, v in epoch_metrics.items()}
        
        # Evaluate model and update progress bar description
        if epoch % args.eval_interval == 0:
            eval_metrics = evaluate_model(cpl_model, dataset)
            pbar.set_description(f"Loss: {avg_metrics['loss']:.4f}, Acc: {avg_metrics['accuracy']:.4f}, Eval Acc: {eval_metrics['eval_accuracy']:.4f}")
        else:
            eval_metrics = {"eval_accuracy": None, "eval_confidence": None}
            pbar.set_description(f"Loss: {avg_metrics['loss']:.4f}, Acc: {avg_metrics['accuracy']:.4f}")
        
        # Update progress bar
        pbar.update(1)
        
        # Record metrics
        metrics_history["epoch"].append(epoch)
        for key in ["loss", "accuracy", "mean_chosen_reward", "mean_rejected_reward", "mean_reward_diff"]:
            metrics_history[key].append(avg_metrics[key])
        metrics_history["eval_accuracy"].append(eval_metrics["eval_accuracy"])
        metrics_history["eval_confidence"].append(eval_metrics["eval_confidence"])
        
        # Save checkpoint
        if epoch % args.save_interval == 0:
            checkpoint_path = checkpoints_dir / f"checkpoint_epoch_{epoch:03d}.pkl"
            cpl_model.save_checkpoint(str(checkpoint_path))
            print(f"Saved checkpoint to {checkpoint_path}")
        
        # Save latest checkpoint
        latest_path = checkpoints_dir / "checkpoint_latest.pkl"
        cpl_model.save_checkpoint(str(latest_path))
        
        # Save metrics
        with open(logs_dir / "metrics.json", 'w') as f:
            serializable_metrics = convert_to_json_serializable(metrics_history)
            json.dump(serializable_metrics, f, indent=2)
        
        # Plot training curves
        plot_training_curves(metrics_history, plots_dir)
        
        # Print epoch time
        epoch_time = time.time() - epoch_start_time
        # print(f"Epoch {epoch} completed in {epoch_time:.2f}s")
    
    pbar.close()
    
    # Save final model
    final_path = checkpoints_dir / "checkpoint_final.pkl"
    cpl_model.save_checkpoint(str(final_path))
    
    # Print training summary
    total_time = time.time() - start_time
    print(f"Training completed in {total_time:.2f}s")
    print(f"Final loss: {metrics_history['loss'][-1]:.4f}")
    print(f"Final accuracy: {metrics_history['accuracy'][-1]:.4f}")
    
    if metrics_history["eval_accuracy"][-1] is not None:
        print(f"Final eval accuracy: {metrics_history['eval_accuracy'][-1]:.4f}")
        print(f"Final confidence: {metrics_history['eval_confidence'][-1]:.4f}")
    
    print(f"Model saved to {final_path}")


if __name__ == "__main__":
    main()
