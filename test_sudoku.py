"""
Simplified training script for Sudoku using TRM
"""
import os
import math
import torch
from torch import nn
from torch.utils.data import DataLoader
import time
from typing import Any, Dict

from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig
from utils.functions import load_model_class
from models.sparse_embedding import CastedSparseEmbeddingSignSGD_Distributed
from models.ema import EMAHelper
from adam_atan2 import AdamATan2

# ============================================================================
# HYPERPARAMETERS - EDIT THESE
# ============================================================================
CONFIG = {
    # Data
    'data_path': 'data/sudoku-extreme-1k-aug-1000',
    'global_batch_size': 32,
    
    # Model architecture
    'arch_name': 'recursive_reasoning.trm@TinyRecursiveReasoningModel_ACTV1',
    'loss_name': 'losses@ACTLossHead',
    'loss_type': 'stablemax_cross_entropy',  # Uses valid_mask parameter
    
    # Architecture config
    'hidden_size': 256,
    'expansion': 1.5,
    'num_heads': 8,
    'H_cycles': 3,
    'L_cycles': 6,
    'L_layers': 2,
    'H_layers': 0,  # Not used in TRM
    'mlp_t': True,  # Use MLP on L instead of transformer
    'pos_encodings': 'none',
    'puzzle_emb_ndim': 256,
    'puzzle_emb_len': 16,
    'no_ACT_continue': True,
    
    # Halting config
    'halt_max_steps': 30,
    'halt_exploration_prob': 0.25,
    
    # Training
    'epochs': 50000,
    'eval_interval': 5000,
    'lr': 1e-4,
    'lr_min_ratio': 0.1,
    'lr_warmup_steps': 100,
    'weight_decay': 1.0,
    'beta1': 0.9,
    'beta2': 0.98,
    
    # Puzzle embedding optimizer
    'puzzle_emb_lr': 1e-4,
    'puzzle_emb_weight_decay': 1.0,
    
    # EMA
    'use_ema': True,
    'ema_rate': 0.999,
    
    # Other
    'seed': 0,
    'forward_dtype': 'bfloat16',
    'print_every': 50,
    
    # Checkpoint
    'checkpoint_path': None,  # Set to path if you want to save checkpoints
    'load_checkpoint': None,  # Set to checkpoint path to resume
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def cosine_schedule_with_warmup_lr_lambda(
    current_step: int, 
    base_lr: float, 
    num_warmup_steps: int, 
    num_training_steps: int, 
    min_ratio: float = 0.1
):
    """Cosine learning rate schedule with warmup"""
    if current_step < num_warmup_steps:
        return base_lr * float(current_step) / float(max(1, num_warmup_steps))
    
    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    return base_lr * (min_ratio + max(0.0, (1 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * 0.5 * 2.0 * progress))))


def create_model(config: Dict, metadata):
    """Create model with loss head"""
    model_cfg = dict(
        batch_size=config['global_batch_size'],
        vocab_size=metadata.vocab_size,
        seq_len=metadata.seq_len,
        num_puzzle_identifiers=metadata.num_puzzle_identifiers,
        puzzle_emb_ndim=config['puzzle_emb_ndim'],
        H_cycles=config['H_cycles'],
        L_cycles=config['L_cycles'],
        H_layers=config['H_layers'],
        L_layers=config['L_layers'],
        hidden_size=config['hidden_size'],
        expansion=config['expansion'],
        num_heads=config['num_heads'],
        pos_encodings=config['pos_encodings'],
        halt_max_steps=config['halt_max_steps'],
        halt_exploration_prob=config['halt_exploration_prob'],
        forward_dtype=config['forward_dtype'],
        mlp_t=config['mlp_t'],
        puzzle_emb_len=config['puzzle_emb_len'],
        no_ACT_continue=config['no_ACT_continue'],
    )
    
    # Load model classes
    model_cls = load_model_class(config['arch_name'], prefix="models.")
    loss_head_cls = load_model_class(config['loss_name'], prefix="models.")
    
    with torch.device("cuda"):
        model = model_cls(model_cfg)
        print(f"\nModel created with {sum(p.numel() for p in model.parameters()):,} parameters")
        model = loss_head_cls(model, loss_type=config['loss_type'])
        
        if config['load_checkpoint'] is not None:
            print(f"Loading checkpoint from {config['load_checkpoint']}")
            state_dict = torch.load(config['load_checkpoint'], map_location="cuda")
            model.load_state_dict(state_dict, assign=True)
        
        if "DISABLE_COMPILE" not in os.environ:
            model = torch.compile(model)
    
    return model


def create_optimizers(model, config):
    """Create optimizers for model parameters and puzzle embeddings"""
    if config['puzzle_emb_ndim'] == 0:
        optimizers = [
            AdamATan2(
                model.parameters(),
                lr=0,
                weight_decay=config['weight_decay'],
                betas=(config['beta1'], config['beta2'])
            )
        ]
        optimizer_lrs = [config['lr']]
    else:
        optimizers = [
            CastedSparseEmbeddingSignSGD_Distributed(
                model.model.puzzle_emb.buffers(),
                lr=0,
                weight_decay=config['puzzle_emb_weight_decay'],
                world_size=1
            ),
            AdamATan2(
                model.parameters(),
                lr=0,
                weight_decay=config['weight_decay'],
                betas=(config['beta1'], config['beta2'])
            )
        ]
        optimizer_lrs = [config['puzzle_emb_lr'], config['lr']]
    
    return optimizers, optimizer_lrs


def train_step(model, batch, optimizers, optimizer_lrs, carry, step, total_steps, config):
    """Single training step"""
    batch = {k: v.cuda() for k, v in batch.items()}
    
    # Initialize carry if needed
    if carry is None:
        with torch.device("cuda"):
            carry = model.initial_carry(batch)
    
    # Forward pass
    carry, loss, metrics, _, _ = model(carry=carry, batch=batch, return_keys=[])
    
    # Backward pass
    ((1 / config['global_batch_size']) * loss).backward()
    
    # Compute learning rate
    for optim, base_lr in zip(optimizers, optimizer_lrs):
        lr = cosine_schedule_with_warmup_lr_lambda(
            step, base_lr, config['lr_warmup_steps'], total_steps, config['lr_min_ratio']
        )
        for param_group in optim.param_groups:
            param_group['lr'] = lr
    
    # Optimizer step
    for optim in optimizers:
        optim.step()
        optim.zero_grad()
    
    # Process metrics
    if len(metrics):
        count = max(metrics["count"].item(), 1)
        processed_metrics = {
            k: v.item() / (config['global_batch_size'] if k.endswith("loss") else count)
            for k, v in metrics.items()
        }
        return carry, processed_metrics
    
    return carry, {}


@torch.inference_mode()
def evaluate(model, eval_loader, config):
    """Evaluate model on test set"""
    model.eval()
    
    total_metrics = {}
    total_count = 0
    
    for set_name, batch, global_batch_size in eval_loader:
        batch = {k: v.cuda() for k, v in batch.items()}
        
        with torch.device("cuda"):
            carry = model.initial_carry(batch)
        
        # Run inference until all sequences halt
        inference_steps = 0
        while True:
            carry, loss, metrics, preds, all_finish = model(
                carry=carry, batch=batch, return_keys=[]
            )
            inference_steps += 1
            if all_finish:
                break
        
        # Accumulate metrics
        count = metrics["count"].item()
        for k, v in metrics.items():
            if k not in total_metrics:
                total_metrics[k] = 0
            total_metrics[k] += v.item()
        total_count += count
    
    # Average metrics
    result = {
        'accuracy': total_metrics.get('accuracy', 0) / max(total_count, 1),
        'exact_accuracy': total_metrics.get('exact_accuracy', 0) / max(total_count, 1),
        'avg_steps': total_metrics.get('steps', 0) / max(total_count, 1),
        'count': total_count
    }
    
    model.train()
    return result


def main():
    # TEMPORARY: Disable compilation for faster startup during testing
    # Comment out this line once you confirm training works
    os.environ['DISABLE_COMPILE'] = '1'
    
    # Set seed
    torch.manual_seed(CONFIG['seed'])
    torch.cuda.manual_seed(CONFIG['seed'])
    
    # print("=" * 80)
    # print("SUDOKU TRAINING WITH TRM")
    # print("=" * 80)
    # print(f"\nConfiguration:")
    # for k, v in CONFIG.items():
    #     if k not in ['arch_name', 'loss_name']:
    #         print(f"  {k:25s}: {v}")
    
    # # Create datasets
    # print("\n" + "=" * 80)
    # print("LOADING DATASETS")
    # print("=" * 80)
    
    # Calculate epochs per iteration for dataset
    train_epochs_per_iter = CONFIG['eval_interval'] if CONFIG['eval_interval'] is not None else CONFIG['epochs']
    
    train_dataset = PuzzleDataset(
        PuzzleDatasetConfig(
            seed=CONFIG['seed'],
            dataset_paths=[CONFIG['data_path']],
            rank=0,
            num_replicas=1,
            test_set_mode=False,
            epochs_per_iter=train_epochs_per_iter,
            global_batch_size=CONFIG['global_batch_size'],
        ),
        split="train"
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=None,
        num_workers=1,
        prefetch_factor=8,
        pin_memory=True,
        persistent_workers=True
    )
    train_metadata = train_dataset.metadata
    
    test_dataset = PuzzleDataset(
        PuzzleDatasetConfig(
            seed=CONFIG['seed'],
            dataset_paths=[CONFIG['data_path']],
            rank=0,
            num_replicas=1,
            test_set_mode=True,
            epochs_per_iter=1,
            global_batch_size=CONFIG['global_batch_size'],
        ),
        split="test"
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=None,
        num_workers=1,
        prefetch_factor=8,
        pin_memory=True,
        persistent_workers=True
    )
    
    print(f"\nDataset info:")
    print(f"  Vocab size: {train_metadata.vocab_size}")
    print(f"  Sequence length: {train_metadata.seq_len}")
    print(f"  Num puzzle identifiers: {train_metadata.num_puzzle_identifiers}")
    
    # Calculate total steps
    total_steps = int(
        CONFIG['epochs'] * train_metadata.total_groups * 
        train_metadata.mean_puzzle_examples / CONFIG['global_batch_size']
    )
    print(f"  Total training steps: {total_steps:,}")
    
    # Create model
    print("\n" + "=" * 80)
    print("CREATING MODEL")
    print("=" * 80)
    model = create_model(CONFIG, train_metadata)
    
    # Create optimizers
    optimizers, optimizer_lrs = create_optimizers(model, CONFIG)
    print(f"Created {len(optimizers)} optimizer(s)")
    
    # Setup EMA
    ema_helper = None
    if CONFIG['use_ema']:
        print("Setting up EMA")
        ema_helper = EMAHelper(mu=CONFIG['ema_rate'])
        ema_helper.register(model)
    
    # Training loop
    # print("\n" + "=" * 80)
    # print("STARTING TRAINING")
    # print("=" * 80)
    
    step = 0
    carry = None
    best_test_acc = 0.0
    best_test_exact_acc = 0.0
    best_acc_epoch = 0
    best_exact_epoch = 0
    
    total_iters = CONFIG['epochs'] // train_epochs_per_iter
    
    for iter_id in range(total_iters):
        epoch_start = iter_id * train_epochs_per_iter
        epoch_end = epoch_start + train_epochs_per_iter
        
        epoch_start_time = time.time()
        
        # Training
        model.train()
        epoch_metrics = []
        batch_count = 0
        
        print(f"\nEpoch {epoch_start}-{epoch_end}: Processing batches...", flush=True)
        
        for set_name, batch, global_batch_size in train_loader:
            if batch_count == 0:
                print(f"  Starting first batch (this may take a while on first run)...", flush=True)
            
            step += 1
            carry, metrics = train_step(
                model, batch, optimizers, optimizer_lrs, 
                carry, step, total_steps, CONFIG
            )
            
            if metrics:
                epoch_metrics.append(metrics)
            
            # Update EMA
            if ema_helper is not None:
                ema_helper.update(model)
            
            batch_count += 1
            
            # Print progress every 100 batches
            if batch_count % 2000 == 0:
                recent_metrics = epoch_metrics[-min(100, len(epoch_metrics)):]
                avg_acc = sum(m.get('accuracy', 0) for m in recent_metrics) / len(recent_metrics)
                avg_exact = sum(m.get('exact_accuracy', 0) for m in recent_metrics) / len(recent_metrics)
                avg_steps = sum(m.get('steps', 0) for m in recent_metrics) / len(recent_metrics)
                print(f"  Batch {batch_count:4d}: Acc={avg_acc:.4f}, Exact={avg_exact:.4f}, Steps={avg_steps:.1f}", flush=True)
        
        epoch = epoch_end
        epoch_time = time.time() - epoch_start_time
        
        # Evaluation
        test_metrics = None
        eval_time = 0
        if epoch % CONFIG['eval_interval'] == 0 or epoch == CONFIG['epochs']:
            eval_start_time = time.time()
            
            # Use EMA model for evaluation if available
            if ema_helper is not None:
                eval_model = ema_helper.ema_copy(model)
            else:
                eval_model = model
            
            test_metrics = evaluate(eval_model, test_loader, CONFIG)
            eval_time = time.time() - eval_start_time
            
            test_acc = test_metrics['accuracy']
            test_exact_acc = test_metrics['exact_accuracy']
            
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                best_acc_epoch = epoch
            
            if test_exact_acc > best_test_exact_acc:
                best_test_exact_acc = test_exact_acc
                best_exact_epoch = epoch
            
            # Save checkpoint
            if CONFIG['checkpoint_path'] is not None:
                os.makedirs(CONFIG['checkpoint_path'], exist_ok=True)
                save_path = os.path.join(CONFIG['checkpoint_path'], f"step_{step}.pt")
                torch.save(eval_model.state_dict(), save_path)
                print(f"Saved checkpoint to {save_path}")
            
            if ema_helper is not None:
                del eval_model
        
        # Print progress every N epochs
        if epoch % CONFIG['print_every'] == 0:
            avg_train_acc = sum(m.get('accuracy', 0) for m in epoch_metrics) / max(len(epoch_metrics), 1)
            avg_train_exact = sum(m.get('exact_accuracy', 0) for m in epoch_metrics) / max(len(epoch_metrics), 1)
            
            print(f"H={CONFIG['H_cycles']} L={CONFIG['L_cycles']} | "
                  f"Epoch {epoch:5d}/{CONFIG['epochs']} ({CONFIG['epochs']-epoch:5d} left) | "
                  f"{epoch_time:.1f}s", end="")
            
            if test_metrics:
                print(f" - eval {eval_time:.1f}s | "
                      f"Train Acc: {avg_train_acc:.4f} | "
                      f"Train Exact: {avg_train_exact:.4f} | "
                      f"Test Acc: {test_metrics['accuracy']:.4f} | "
                      f"Test Exact: {test_metrics['exact_accuracy']:.4f} | "
                      f"Best Test Acc: {best_test_acc:.4f} (Epoch {best_acc_epoch}) | "
                      f"Best Test Exact: {best_test_exact_acc:.4f} (Epoch {best_exact_epoch}) | "
                      f"Avg Steps: {test_metrics['avg_steps']:.1f}")
            else:
                print(f" | Train Acc: {avg_train_acc:.4f} | Train Exact: {avg_train_exact:.4f}")
    
    print("\n" + "=" * 80)
    print(f"TRAINING COMPLETE!")
    print(f"Best Test Accuracy: {best_test_acc:.4f} at epoch {best_acc_epoch}")
    print(f"Best Test Exact Match: {best_test_exact_acc:.4f} at epoch {best_exact_epoch}")
    print("=" * 80)
    
    # Final evaluation
    print("\nFinal evaluation on test set...")
    if ema_helper is not None:
        final_model = ema_helper.ema_copy(model)
    else:
        final_model = model
    
    final_metrics = evaluate(final_model, test_loader, CONFIG)
    print(f"Final Test Accuracy: {final_metrics['accuracy']:.4f}")
    print(f"Final Test Exact Match: {final_metrics['exact_accuracy']:.4f}")
    print(f"Final Avg Steps: {final_metrics['avg_steps']:.1f}")


if __name__ == "__main__":
    main()