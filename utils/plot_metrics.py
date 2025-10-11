#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plotting script for training metrics from evaluation_metrics.txt

This script reads the evaluation metrics file and creates plots for:
- Training loss over epochs
- IoU (Intersection over Union) over epochs  
- nIoU (normalized IoU) over epochs
- PD (Probability of Detection) over epochs
- FA (False Alarm rate) over epochs

Usage:
    python utils/plot_metrics.py --input workdirs/fix_mask_cache/evaluation_metrics.txt --output plots/
"""

import argparse
import os
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style as style
from typing import Dict, List, Tuple
import seaborn as sns
from scipy.signal import savgol_filter
from scipy.ndimage import uniform_filter1d

# Set style for better looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def parse_metrics_file(file_path: str) -> Dict[str, List[float]]:
    """
    Parse the evaluation metrics file and extract data.
    
    Args:
        file_path (str): Path to the evaluation_metrics.txt file
        
    Returns:
        Dict[str, List[float]]: Dictionary containing lists of values for each metric
    """
    metrics = {
        'epoch': [],
        'loss': [],
        'iou': [],
        'niou': [],
        'pd': [],
        'fa': []
    }
    
    # Regular expression to parse each line
    pattern = r'epoch:\s*(\d+),\s*loss:\s*([\d.]+),\s*iou:\s*([\d.]+),\s*niou:\s*([\d.]+),\s*pd:\s*([\d.]+),\s*fa:\s*([\d.]+)'
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:  # Skip empty lines
                    continue
                    
                match = re.match(pattern, line)
                if match:
                    epoch, loss, iou, niou, pd, fa = match.groups()
                    metrics['epoch'].append(int(epoch))
                    metrics['loss'].append(float(loss))
                    metrics['iou'].append(float(iou))
                    metrics['niou'].append(float(niou))
                    metrics['pd'].append(float(pd))
                    metrics['fa'].append(float(fa))
                else:
                    print(f"Warning: Could not parse line {line_num}: {line}")
                    
    except FileNotFoundError:
        print(f"Error: File {file_path} not found!")
        return {}
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return {}
    
    print(f"Successfully parsed {len(metrics['epoch'])} data points from {file_path}")
    return metrics

def smooth_curve(y: np.ndarray, method: str = 'savgol', window_size: int = None, **kwargs) -> np.ndarray:
    """
    Smooth a curve using different methods.
    
    Args:
        y (np.ndarray): Input data to smooth
        method (str): Smoothing method ('savgol', 'moving_average', 'exponential')
        window_size (int): Window size for smoothing (auto-calculated if None)
        **kwargs: Additional parameters for specific methods
        
    Returns:
        np.ndarray: Smoothed data
    """
    if len(y) < 5:  # Not enough data to smooth
        return y
    
    # Auto-calculate window size if not provided
    if window_size is None:
        window_size = max(5, min(21, len(y) // 10))
        if window_size % 2 == 0:  # Make sure it's odd for savgol
            window_size += 1
    
    if method == 'savgol':
        # Savitzky-Golay filter
        poly_order = min(3, window_size - 1)
        try:
            return savgol_filter(y, window_size, poly_order)
        except:
            # Fallback to moving average if savgol fails
            return uniform_filter1d(y.astype(float), size=window_size)
    
    elif method == 'moving_average':
        # Simple moving average
        return uniform_filter1d(y.astype(float), size=window_size)
    
    elif method == 'exponential':
        # Exponential smoothing
        alpha = kwargs.get('alpha', 0.1)
        smoothed = np.zeros_like(y)
        smoothed[0] = y[0]
        for i in range(1, len(y)):
            smoothed[i] = alpha * y[i] + (1 - alpha) * smoothed[i-1]
        return smoothed
    
    else:
        return y

def create_individual_plots(metrics: Dict[str, List[float]], output_dir: str) -> None:
    """
    Create individual plots for each metric.
    
    Args:
        metrics (Dict[str, List[float]]): Parsed metrics data
        output_dir (str): Directory to save plots
    """
    if not metrics or not metrics['epoch']:
        print("No data to plot!")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    epochs = metrics['epoch']
    
    # Plot configurations
    plot_configs = [
        ('loss', 'Training Loss', 'Loss', 'red'),
        ('iou', 'IoU (Intersection over Union)', 'IoU', 'blue'),
        ('niou', 'Normalized IoU', 'nIoU', 'green'),
        ('pd', 'Probability of Detection', 'PD', 'orange'),
        ('fa', 'False Alarm Rate', 'FA', 'purple')
    ]
    
    for metric_name, title, ylabel, color in plot_configs:
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, metrics[metric_name], color=color, linewidth=2, marker='o', markersize=3)
        plt.title(f'{title} over Epochs', fontsize=16, fontweight='bold')
        plt.xlabel('Epoch', fontsize=14)
        plt.ylabel(ylabel, fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Add some statistics to the plot
        mean_val = np.mean(metrics[metric_name])
        max_val = np.max(metrics[metric_name])
        min_val = np.min(metrics[metric_name])
        
        # Add text box with statistics
        stats_text = f'Mean: {mean_val:.4f}\nMax: {max_val:.4f}\nMin: {min_val:.4f}'
        plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Save plot
        output_path = os.path.join(output_dir, f'{metric_name}_plot.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved {metric_name} plot to {output_path}")

def create_combined_plots(metrics: Dict[str, List[float]], output_dir: str) -> None:
    """
    Create combined plots showing multiple metrics.
    
    Args:
        metrics (Dict[str, List[float]]): Parsed metrics data
        output_dir (str): Directory to save plots
    """
    if not metrics or not metrics['epoch']:
        print("No data to plot!")
        return
    
    epochs = metrics['epoch']
    
    # Create a 2x3 subplot layout
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Training Metrics Overview', fontsize=20, fontweight='bold')
    
    # Plot configurations for subplots
    plot_configs = [
        (0, 0, 'loss', 'Training Loss', 'Loss', 'red'),
        (0, 1, 'iou', 'IoU', 'IoU', 'blue'),
        (0, 2, 'niou', 'Normalized IoU', 'nIoU', 'green'),
        (1, 0, 'pd', 'Probability of Detection', 'PD', 'orange'),
        (1, 1, 'fa', 'False Alarm Rate', 'FA', 'purple'),
    ]
    
    for row, col, metric_name, title, ylabel, color in plot_configs:
        ax = axes[row, col]
        ax.plot(epochs, metrics[metric_name], color=color, linewidth=2, marker='o', markersize=2)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Add statistics
        mean_val = np.mean(metrics[metric_name])
        ax.axhline(y=mean_val, color=color, linestyle='--', alpha=0.7, label=f'Mean: {mean_val:.4f}')
        ax.legend()
    
    # Hide the last subplot (2,2) since we only have 5 metrics
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    
    # Save combined plot
    output_path = os.path.join(output_dir, 'combined_metrics.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved combined metrics plot to {output_path}")

def create_comparison_plot(metrics: Dict[str, List[float]], output_dir: str) -> None:
    """
    Create a comparison plot showing IoU vs nIoU and PD vs FA.
    
    Args:
        metrics (Dict[str, List[float]]): Parsed metrics data
        output_dir (str): Directory to save plots
    """
    if not metrics or not metrics['epoch']:
        print("No data to plot!")
        return
    
    epochs = metrics['epoch']
    
    # Create comparison plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # IoU vs nIoU comparison
    ax1.plot(epochs, metrics['iou'], label='IoU', color='blue', linewidth=2, marker='o', markersize=2)
    ax1.plot(epochs, metrics['niou'], label='nIoU', color='green', linewidth=2, marker='s', markersize=2)
    ax1.set_title('IoU vs Normalized IoU', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Value', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # PD vs FA comparison
    ax2.plot(epochs, metrics['pd'], label='PD (Probability of Detection)', color='orange', linewidth=2, marker='o', markersize=2)
    ax2_twin = ax2.twinx()
    ax2_twin.plot(epochs, metrics['fa'], label='FA (False Alarm)', color='red', linewidth=2, marker='^', markersize=2)
    
    ax2.set_title('PD vs FA', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('PD', fontsize=12, color='orange')
    ax2_twin.set_ylabel('FA', fontsize=12, color='red')
    ax2.grid(True, alpha=0.3)
    
    # Combine legends
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    plt.tight_layout()
    
    # Save comparison plot
    output_path = os.path.join(output_dir, 'comparison_metrics.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved comparison metrics plot to {output_path}")

def create_all_in_one_plot(metrics: Dict[str, List[float]], output_dir: str) -> None:
    """
    Create a single plot with all metrics curves.
    
    Args:
        metrics (Dict[str, List[float]]): Parsed metrics data
        output_dir (str): Directory to save plots
    """
    if not metrics or not metrics['epoch']:
        print("No data to plot!")
        return
    
    epochs = metrics['epoch']
    
    # Create figure with dual y-axes since metrics have different scales
    fig, ax1 = plt.subplots(figsize=(14, 8))
    
    # Plot loss on primary y-axis (left)
    color1 = 'red'
    ax1.set_xlabel('Epoch', fontsize=14)
    ax1.set_ylabel('Loss', color=color1, fontsize=14)
    line1 = ax1.plot(epochs, metrics['loss'], color=color1, linewidth=2, marker='o', markersize=2, label='Loss')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, alpha=0.3)
    
    # Create secondary y-axis for other metrics
    ax2 = ax1.twinx()
    
    # Plot IoU, nIoU, PD on secondary y-axis (right)
    colors = ['blue', 'green', 'orange']
    markers = ['s', '^', 'D']
    metrics_to_plot = ['iou', 'niou', 'pd']
    labels = ['IoU', 'nIoU', 'PD']
    
    lines2 = []
    for i, (metric, label, color, marker) in enumerate(zip(metrics_to_plot, labels, colors, markers)):
        line = ax2.plot(epochs, metrics[metric], color=color, linewidth=2, marker=marker, 
                       markersize=2, label=label, alpha=0.8)
        lines2.extend(line)
    
    ax2.set_ylabel('IoU / nIoU / PD', fontsize=14)
    ax2.tick_params(axis='y')
    
    # Create third y-axis for FA (since it has very small values)
    ax3 = ax1.twinx()
    ax3.spines['right'].set_position(('outward', 60))  # Move the spine outward
    
    color3 = 'purple'
    line3 = ax3.plot(epochs, metrics['fa'], color=color3, linewidth=2, marker='v', 
                    markersize=2, label='FA', alpha=0.8)
    ax3.set_ylabel('FA (False Alarm)', color=color3, fontsize=14)
    ax3.tick_params(axis='y', labelcolor=color3)
    
    # Set title
    plt.title('All Training Metrics in One Plot', fontsize=16, fontweight='bold', pad=20)
    
    # Combine all legends
    all_lines = line1 + lines2 + line3
    all_labels = [l.get_label() for l in all_lines]
    ax1.legend(all_lines, all_labels, loc='upper left', bbox_to_anchor=(0, 1))
    
    # Add statistics text box
    stats_text = f'Final Values (Epoch {epochs[-1]}):\n'
    stats_text += f'Loss: {metrics["loss"][-1]:.4f}\n'
    stats_text += f'IoU: {metrics["iou"][-1]:.4f}\n'
    stats_text += f'nIoU: {metrics["niou"][-1]:.4f}\n'
    stats_text += f'PD: {metrics["pd"][-1]:.4f}\n'
    stats_text += f'FA: {metrics["fa"][-1]:.6f}'
    
    ax1.text(0.02, 0.02, stats_text, transform=ax1.transAxes, 
            verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    plt.tight_layout()
    
    # Save plot
    output_path = os.path.join(output_dir, 'all_metrics_in_one.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved all-in-one metrics plot to {output_path}")

def create_normalized_all_in_one_plot(metrics: Dict[str, List[float]], output_dir: str) -> None:
    """
    Create a single plot with all metrics curves normalized to [0,1] range.
    
    Args:
        metrics (Dict[str, List[float]]): Parsed metrics data
        output_dir (str): Directory to save plots
    """
    if not metrics or not metrics['epoch']:
        print("No data to plot!")
        return
    
    epochs = metrics['epoch']
    
    # Normalize all metrics to [0,1] range for better comparison
    normalized_metrics = {}
    for metric in ['loss', 'iou', 'niou', 'pd', 'fa']:
        values = np.array(metrics[metric])
        min_val = np.min(values)
        max_val = np.max(values)
        if max_val - min_val > 0:
            normalized_metrics[metric] = (values - min_val) / (max_val - min_val)
        else:
            normalized_metrics[metric] = values
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Plot configurations
    plot_configs = [
        ('loss', 'Loss (normalized)', 'red', 'o'),
        ('iou', 'IoU (normalized)', 'blue', 's'),
        ('niou', 'nIoU (normalized)', 'green', '^'),
        ('pd', 'PD (normalized)', 'orange', 'D'),
        ('fa', 'FA (normalized)', 'purple', 'v')
    ]
    
    for metric_name, label, color, marker in plot_configs:
        plt.plot(epochs, normalized_metrics[metric_name], color=color, linewidth=2, 
                marker=marker, markersize=3, label=label, alpha=0.8)
    
    plt.title('All Training Metrics (Normalized) in One Plot', fontsize=16, fontweight='bold')
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Normalized Value [0,1]', fontsize=14)
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    
    # Add normalization info
    norm_text = 'Normalization Info:\n'
    for metric in ['loss', 'iou', 'niou', 'pd', 'fa']:
        values = np.array(metrics[metric])
        min_val = np.min(values)
        max_val = np.max(values)
        norm_text += f'{metric.upper()}: [{min_val:.4f}, {max_val:.4f}]\n'
    
    plt.text(0.02, 0.98, norm_text, transform=plt.gca().transAxes, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    plt.tight_layout()
    
    # Save plot
    output_path = os.path.join(output_dir, 'all_metrics_normalized.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved normalized all-in-one metrics plot to {output_path}")

def create_loss_iou_plot(metrics: Dict[str, List[float]], output_dir: str) -> None:
    """
    Create a plot with only loss and IoU curves with smoothing.
    
    Args:
        metrics (Dict[str, List[float]]): Parsed metrics data
        output_dir (str): Directory to save plots
    """
    if not metrics or not metrics['epoch']:
        print("No data to plot!")
        return
    
    epochs = metrics['epoch']
    
    # Apply smoothing to loss and IoU
    loss_raw = np.array(metrics['loss'])
    iou_raw = np.array(metrics['iou'])
    
    loss_smooth = smooth_curve(loss_raw, method='savgol', window_size=11)
    iou_smooth = smooth_curve(iou_raw, method='savgol', window_size=11)
    
    # Create figure with dual y-axes since loss and IoU have different scales
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # Plot loss on primary y-axis (left)
    color1 = 'red'
    color1_light = 'lightcoral'
    ax1.set_xlabel('Epoch', fontsize=14)
    ax1.set_ylabel('Loss', color=color1, fontsize=14)
    
    # Plot raw data as light line
    ax1.plot(epochs, loss_raw, color=color1_light, linewidth=1, alpha=0.5, label='Loss (raw)')
    # Plot smoothed data as main line
    line1 = ax1.plot(epochs, loss_smooth, color=color1, linewidth=3, label='Loss (smoothed)')
    
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, alpha=0.3)
    
    # Create secondary y-axis for IoU
    ax2 = ax1.twinx()
    color2 = 'blue'
    color2_light = 'lightblue'
    ax2.set_ylabel('IoU', color=color2, fontsize=14)
    
    # Plot raw data as light line
    ax2.plot(epochs, iou_raw, color=color2_light, linewidth=1, alpha=0.5, label='IoU (raw)')
    # Plot smoothed data as main line
    line2 = ax2.plot(epochs, iou_smooth, color=color2, linewidth=3, label='IoU (smoothed)')
    
    ax2.tick_params(axis='y', labelcolor=color2)
    
    # Set title
    plt.title('Training Loss and IoU (Smoothed)', fontsize=16, fontweight='bold')
    
    # Create combined legend
    # Get all lines and labels from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right')
    
    # Add statistics text box using smoothed data
    loss_final_smooth = loss_smooth[-1]
    loss_initial_smooth = loss_smooth[0]
    loss_improvement = ((loss_initial_smooth - loss_final_smooth) / loss_initial_smooth) * 100
    
    iou_final_smooth = iou_smooth[-1]
    iou_initial_smooth = iou_smooth[0]
    iou_improvement = ((iou_final_smooth - iou_initial_smooth) / iou_initial_smooth) * 100
    
    stats_text = f'Final Results (Epoch {epochs[-1]}):\n'
    stats_text += f'Loss (smooth): {loss_final_smooth:.4f} ({loss_improvement:+.1f}%)\n'
    stats_text += f'IoU (smooth): {iou_final_smooth:.4f} ({iou_improvement:+.1f}%)\n\n'
    stats_text += f'Best Values (smoothed):\n'
    stats_text += f'Min Loss: {np.min(loss_smooth):.4f}\n'
    stats_text += f'Max IoU: {np.max(iou_smooth):.4f}\n\n'
    stats_text += f'Raw vs Smoothed:\n'
    stats_text += f'Loss: {metrics["loss"][-1]:.4f} → {loss_final_smooth:.4f}\n'
    stats_text += f'IoU: {metrics["iou"][-1]:.4f} → {iou_final_smooth:.4f}'
    
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    plt.tight_layout()
    
    # Save plot
    output_path = os.path.join(output_dir, 'loss_iou_plot.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved loss and IoU plot to {output_path}")

def print_statistics(metrics: Dict[str, List[float]]) -> None:
    """
    Print statistical summary of the metrics.
    
    Args:
        metrics (Dict[str, List[float]]): Parsed metrics data
    """
    if not metrics or not metrics['epoch']:
        print("No data available for statistics!")
        return
    
    print("\n" + "="*60)
    print("TRAINING METRICS STATISTICS")
    print("="*60)
    
    metric_names = ['loss', 'iou', 'niou', 'pd', 'fa']
    
    for metric in metric_names:
        values = np.array(metrics[metric])
        print(f"\n{metric.upper()}:")
        print(f"  Mean:     {np.mean(values):.6f}")
        print(f"  Std:      {np.std(values):.6f}")
        print(f"  Min:      {np.min(values):.6f} (Epoch {metrics['epoch'][np.argmin(values)]})")
        print(f"  Max:      {np.max(values):.6f} (Epoch {metrics['epoch'][np.argmax(values)]})")
        print(f"  Final:    {values[-1]:.6f} (Epoch {metrics['epoch'][-1]})")
        
        # Calculate improvement
        if len(values) > 1:
            if metric == 'loss' or metric == 'fa':  # Lower is better
                improvement = values[0] - values[-1]
                improvement_pct = (improvement / values[0]) * 100
            else:  # Higher is better
                improvement = values[-1] - values[0]
                improvement_pct = (improvement / values[0]) * 100
            print(f"  Change:   {improvement:+.6f} ({improvement_pct:+.2f}%)")

def main():
    """Main function to parse arguments and create plots."""
    parser = argparse.ArgumentParser(description='Plot training metrics from evaluation_metrics.txt')
    parser.add_argument('--input', '-i', type=str, required=True,
                       help='Path to the evaluation_metrics.txt file')
    parser.add_argument('--output', '-o', type=str, default='plots/',
                       help='Output directory for plots (default: plots/)')
    parser.add_argument('--stats', action='store_true',
                       help='Print detailed statistics')
    
    args = parser.parse_args()
    
    # Parse the metrics file
    print(f"Reading metrics from: {args.input}")
    metrics = parse_metrics_file(args.input)
    
    if not metrics or not metrics['epoch']:
        print("No valid data found in the metrics file!")
        return
    
    # Print statistics if requested
    if args.stats:
        print_statistics(metrics)
    
    # Create plots
    print(f"\nCreating plots in directory: {args.output}")
    create_individual_plots(metrics, args.output)
    create_combined_plots(metrics, args.output)
    create_comparison_plot(metrics, args.output)
    create_all_in_one_plot(metrics, args.output)
    create_normalized_all_in_one_plot(metrics, args.output)
    create_loss_iou_plot(metrics, args.output)
    
    print(f"\nAll plots have been saved to: {args.output}")
    print("Generated files:")
    print("  - loss_plot.png")
    print("  - iou_plot.png") 
    print("  - niou_plot.png")
    print("  - pd_plot.png")
    print("  - fa_plot.png")
    print("  - combined_metrics.png")
    print("  - comparison_metrics.png")
    print("  - all_metrics_in_one.png")
    print("  - all_metrics_normalized.png")
    print("  - loss_iou_plot.png")

def plot_metrics_example():
    """Example of how to use the plotting functions programmatically."""
    
    # Path to your evaluation metrics file
    metrics_file = "workdirs/fix_mask_cache/evaluation_metrics.txt"
    output_dir = "workdirs/fix_mask_cache/"
    
    # Check if file exists
    if not os.path.exists(metrics_file):
        print(f"Error: Metrics file not found: {metrics_file}")
        print("Please update the path to your evaluation_metrics.txt file")
        return
    
    print(f"Parsing metrics from: {metrics_file}")
    
    # Parse the metrics file
    metrics = parse_metrics_file(metrics_file)
    
    if not metrics or not metrics['epoch']:
        print("No valid data found!")
        return
    
    # Print statistics
    print_statistics(metrics)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create all types of plots
    print(f"\nCreating plots in: {output_dir}")
    create_individual_plots(metrics, output_dir)
    create_combined_plots(metrics, output_dir)
    create_comparison_plot(metrics, output_dir)
    create_all_in_one_plot(metrics, output_dir)
    create_normalized_all_in_one_plot(metrics, output_dir)
    create_loss_iou_plot(metrics, output_dir)
    
    print(f"\nPlots saved successfully to: {output_dir}")
    print("Generated files:")
    print("  - Individual plots: loss_plot.png, iou_plot.png, niou_plot.png, pd_plot.png, fa_plot.png")
    print("  - Combined view: combined_metrics.png")
    print("  - Comparison view: comparison_metrics.png")
    print("  - All-in-one view: all_metrics_in_one.png")
    print("  - Normalized all-in-one: all_metrics_normalized.png")
    print("  - Loss & IoU only: loss_iou_plot.png")

if __name__ == "__main__":
    plot_metrics_example()


