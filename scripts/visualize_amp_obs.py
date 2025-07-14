#!/usr/bin/env python3

import torch
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
from typing import Dict, List, Tuple
import math

def load_amp_data(tensor_path: str, metadata_path: str) -> Tuple[torch.Tensor, List[Dict]]:
    """Load AMP observation tensor and metadata."""
    tensor = torch.load(tensor_path, map_location='cpu')
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    return tensor, metadata

def parse_dimensions(metadata: List[Dict]) -> Dict:
    """Parse observation dimensions and create dimension mapping."""
    dim_mapping = {}
    current_dim = 0
    
    for obs_info in metadata:
        obs_type = obs_info["obs_type"]
        obs_dim = obs_info["obs_dim"]
        
        dim_mapping[obs_type] = {
            "start_dim": current_dim,
            "end_dim": current_dim + obs_dim,
            "info": obs_info
        }
        current_dim += obs_dim
    
    return dim_mapping

def visualize_joint_distributions(tensor: torch.Tensor, dim_mapping: Dict, output_dir: str):
    """Visualize joint position and velocity distributions using num_cols and num_rows layout."""
    plt.style.use('seaborn-v0_8')
    
    for obs_type, dim_info in dim_mapping.items():
        if "joint" not in obs_type:
            continue
            
        start_dim = dim_info["start_dim"]
        end_dim = dim_info["end_dim"]
        obs_info = dim_info["info"]
        
        # Extract data for this observation type
        data = tensor[:, start_dim:end_dim].numpy()
        
        # Get joint names and history steps
        joint_names = obs_info.get("joint_names", [])
        history_steps = obs_info.get("history_steps", [0])
        
        if not joint_names:
            print(f"No joint names found for {obs_type}, skipping...")
            continue
        
        # Calculate dimensions per joint per history step
        dims_per_joint = data.shape[1] // (len(joint_names) * len(history_steps))
        
        # Reshape data: (num_frames, num_history_steps, num_joints, dims_per_joint)
        try:
            reshaped_data = data.reshape(data.shape[0], len(history_steps), len(joint_names), dims_per_joint)
        except ValueError:
            print(f"Could not reshape data for {obs_type}, skipping...")
            continue
        
        # Calculate total number of subplots needed
        total_subplots = len(joint_names) * dims_per_joint
        
        # Calculate optimal grid layout
        num_cols = min(4, total_subplots)  # Max 4 columns
        num_rows = math.ceil(total_subplots / num_cols)
        
        # Create visualization
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(5 * num_cols, 4 * num_rows))
        fig.suptitle(f'{obs_type} Distribution', fontsize=16)
        
        # Handle case where we have only one subplot
        if total_subplots == 1:
            axes = np.array([axes])
        elif num_rows == 1:
            axes = axes.reshape(1, -1)
        elif num_cols == 1:
            axes = axes.reshape(-1, 1)
        
        # Plot each joint dimension
        plot_idx = 0
        for joint_idx, joint_name in enumerate(joint_names):
            for dim_idx in range(dims_per_joint):
                if plot_idx >= total_subplots:
                    break
                    
                # Calculate subplot position
                row = plot_idx // num_cols
                col = plot_idx % num_cols
                ax = axes[row, col]
                
                # Plot distribution for current timestep (history_step = 0)
                current_data = reshaped_data[:, 0, joint_idx, dim_idx]
                
                # Create histogram
                ax.hist(current_data, bins=50, alpha=0.7, density=True, 
                       label=f'Current (t=0)', color='skyblue')
                
                # Add statistics
                mean_val = np.mean(current_data)
                std_val = np.std(current_data)
                ax.axvline(mean_val, color='red', linestyle='--', alpha=0.8, 
                          label=f'Mean: {mean_val:.3f}')
                ax.axvline(mean_val + std_val, color='red', linestyle=':', alpha=0.6,
                          label=f'±1σ')
                ax.axvline(mean_val - std_val, color='red', linestyle=':', alpha=0.6)
                
                # Plot history steps if available
                if len(history_steps) > 1:
                    colors = plt.cm.viridis(np.linspace(0, 1, len(history_steps)))
                    for hist_idx, hist_step in enumerate(history_steps[1:], 1):
                        hist_data = reshaped_data[:, hist_idx, joint_idx, dim_idx]
                        ax.hist(hist_data, bins=50, alpha=0.3, density=True, 
                               color=colors[hist_idx], label=f't={-hist_step}')
                
                # Set labels and title
                if dims_per_joint == 1:
                    ax.set_title(f'{joint_name}')
                else:
                    ax.set_title(f'{joint_name} - Dim {dim_idx}')
                
                ax.set_xlabel('Value')
                ax.set_ylabel('Density')
                ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3)
                
                # Add statistics text box
                stats_text = f'μ={mean_val:.3f}\nσ={std_val:.3f}\nmin={np.min(current_data):.3f}\nmax={np.max(current_data):.3f}'
                ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                       verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                       fontsize=8)
                
                plot_idx += 1
        
        # Hide unused subplots
        for idx in range(plot_idx, num_rows * num_cols):
            row = idx // num_cols
            col = idx % num_cols
            if row < num_rows and col < num_cols:
                axes[row, col].set_visible(False)
        
        plt.tight_layout()
        
        # Save plot
        output_path = Path(output_dir) / f'{obs_type}_distribution.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved {obs_type} distribution plot to {output_path}")
        print(f"  Layout: {num_rows} rows × {num_cols} cols for {total_subplots} subplots")

def visualize_body_distributions(tensor: torch.Tensor, dim_mapping: Dict, output_dir: str):
    """Visualize body-related distributions using num_cols and num_rows layout."""
    plt.style.use('seaborn-v0_8')
    
    for obs_type, dim_info in dim_mapping.items():
        if "body" not in obs_type:
            continue
            
        start_dim = dim_info["start_dim"]
        end_dim = dim_info["end_dim"]
        obs_info = dim_info["info"]
        
        # Extract data for this observation type
        data = tensor[:, start_dim:end_dim].numpy()
        
        # Get body names and history steps
        body_names = obs_info.get("body_names", [])
        history_steps = obs_info.get("history_steps", [0])
        
        if not body_names:
            print(f"No body names found for {obs_type}, skipping...")
            continue
        
        # Calculate dimensions per body per history step
        if "ori" in obs_type:
            # Orientation matrices are flattened (2x3 = 6 dims per body)
            dims_per_body = 6
            dim_labels = ['R00', 'R01', 'R02', 'R10', 'R11', 'R12']
        else:
            # Position and velocity are 3D
            dims_per_body = 3
            dim_labels = ['X', 'Y', 'Z']
        
        # Reshape data
        try:
            reshaped_data = data.reshape(data.shape[0], len(history_steps), len(body_names), dims_per_body)
        except ValueError:
            print(f"Could not reshape data for {obs_type}, skipping...")
            continue
        
        # Calculate total number of subplots needed
        total_subplots = len(body_names) * dims_per_body
        
        # Calculate optimal grid layout
        num_cols = min(4, total_subplots)  # Max 4 columns
        num_rows = math.ceil(total_subplots / num_cols)
        
        # Create visualization
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(5 * num_cols, 4 * num_rows))
        fig.suptitle(f'{obs_type} Distribution', fontsize=16)
        
        # Handle case where we have only one subplot
        if total_subplots == 1:
            axes = np.array([axes])
        elif num_rows == 1:
            axes = axes.reshape(1, -1)
        elif num_cols == 1:
            axes = axes.reshape(-1, 1)
        
        # Plot each body dimension
        plot_idx = 0
        for body_idx, body_name in enumerate(body_names):
            for dim_idx in range(dims_per_body):
                if plot_idx >= total_subplots:
                    break
                    
                # Calculate subplot position
                row = plot_idx // num_cols
                col = plot_idx % num_cols
                ax = axes[row, col]
                
                # Plot distribution for current timestep
                current_data = reshaped_data[:, 0, body_idx, dim_idx]
                
                # Create histogram
                ax.hist(current_data, bins=50, alpha=0.7, density=True, color='lightcoral')
                
                # Add statistics
                mean_val = np.mean(current_data)
                std_val = np.std(current_data)
                ax.axvline(mean_val, color='red', linestyle='--', alpha=0.8, 
                          label=f'Mean: {mean_val:.3f}')
                ax.axvline(mean_val + std_val, color='red', linestyle=':', alpha=0.6,
                          label=f'±1σ')
                ax.axvline(mean_val - std_val, color='red', linestyle=':', alpha=0.6)
                
                # Set labels and title
                ax.set_title(f'{body_name} - {dim_labels[dim_idx]}')
                ax.set_xlabel('Value')
                ax.set_ylabel('Density')
                ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3)
                
                # Add statistics text box
                stats_text = f'μ={mean_val:.3f}\nσ={std_val:.3f}\nmin={np.min(current_data):.3f}\nmax={np.max(current_data):.3f}'
                ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                       verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                       fontsize=8)
                
                plot_idx += 1
        
        # Hide unused subplots
        for idx in range(plot_idx, num_rows * num_cols):
            row = idx // num_cols
            col = idx % num_cols
            if row < num_rows and col < num_cols:
                axes[row, col].set_visible(False)
        
        plt.tight_layout()
        
        # Save plot
        output_path = Path(output_dir) / f'{obs_type}_distribution.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved {obs_type} distribution plot to {output_path}")
        print(f"  Layout: {num_rows} rows × {num_cols} cols for {total_subplots} subplots")

def create_summary_plot(tensor: torch.Tensor, metadata: List[Dict], output_dir: str):
    """Create a summary plot of all observations."""
    plt.style.use('seaborn-v0_8')
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('AMP Observations Summary', fontsize=16)
    
    # Plot 1: Observation dimensions
    obs_names = [obs["obs_type"] for obs in metadata]
    obs_dims = [obs["obs_dim"] for obs in metadata]
    
    axes[0, 0].bar(range(len(obs_names)), obs_dims)
    axes[0, 0].set_xticks(range(len(obs_names)))
    axes[0, 0].set_xticklabels(obs_names, rotation=45, ha='right')
    axes[0, 0].set_ylabel('Dimension')
    axes[0, 0].set_title('Observation Dimensions')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Data distribution overview
    data_sample = tensor[::max(1, len(tensor)//1000), :].numpy()  # Sample for performance
    axes[0, 1].hist(data_sample.flatten(), bins=100, alpha=0.7)
    axes[0, 1].set_xlabel('Value')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Overall Data Distribution')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Temporal statistics
    mean_per_frame = tensor.mean(dim=1).numpy()
    std_per_frame = tensor.std(dim=1).numpy()
    
    frame_indices = np.arange(len(mean_per_frame))
    sample_indices = np.linspace(0, len(frame_indices)-1, min(1000, len(frame_indices)), dtype=int)
    
    axes[1, 0].plot(frame_indices[sample_indices], mean_per_frame[sample_indices], alpha=0.7, label='Mean')
    axes[1, 0].plot(frame_indices[sample_indices], std_per_frame[sample_indices], alpha=0.7, label='Std')
    axes[1, 0].set_xlabel('Frame')
    axes[1, 0].set_ylabel('Value')
    axes[1, 0].set_title('Temporal Statistics')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Correlation matrix (sample)
    sample_data = tensor[::max(1, len(tensor)//100), :].numpy()
    if sample_data.shape[1] > 50:
        # Sample dimensions if too many
        dim_indices = np.linspace(0, sample_data.shape[1]-1, 50, dtype=int)
        sample_data = sample_data[:, dim_indices]
    
    corr_matrix = np.corrcoef(sample_data.T)
    im = axes[1, 1].imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    axes[1, 1].set_title('Dimension Correlation Matrix')
    plt.colorbar(im, ax=axes[1, 1])
    
    plt.tight_layout()
    
    # Save plot
    output_path = Path(output_dir) / 'summary_plot.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved summary plot to {output_path}")

def main():
    root_dir = "amp_obs/expert/"

    tensor_path = f"{root_dir}/tensor.pt"
    metadata_path = f"{root_dir}/metadata.json"
    output_dir = f"{root_dir}/vis_output"
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("Loading AMP observation data...")
    tensor, metadata = load_amp_data(tensor_path, metadata_path)
    
    print(f"Loaded tensor shape: {tensor.shape}")
    print(f"Total observations: {len(metadata)}")
    
    # Parse dimensions
    dim_mapping = parse_dimensions(metadata)
    
    # Create visualizations
    print("Creating visualizations...")
    
    # # Summary plot
    # create_summary_plot(tensor, metadata, output_dir)
    
    # Joint distributions
    visualize_joint_distributions(tensor, dim_mapping, output_dir)
    
    # Body distributions
    visualize_body_distributions(tensor, dim_mapping, output_dir)
    
    print(f"\nVisualization complete! Check {output_dir} for output plots.")

if __name__ == "__main__":
    main()