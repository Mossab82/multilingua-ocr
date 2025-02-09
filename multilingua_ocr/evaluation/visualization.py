import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, List, Optional, Tuple
import torch

class DegradationVisualizer:
    """
    Visualizes document degradation analysis as described in Section 3.1.
    """
    def plot_degradation_map(
        self,
        image: np.ndarray,
        degradation_map: np.ndarray,
        save_path: Optional[str] = None
    ):
        """Plot original image with degradation heatmap overlay."""
        plt.figure(figsize=(15, 5))
        
        # Original image
        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.title('Original Document')
        plt.axis('off')
        
        # Degradation heatmap
        plt.subplot(1, 2, 2)
        plt.imshow(degradation_map, cmap='YlOrRd')
        plt.colorbar(label='Degradation Level')
        plt.title('Degradation Analysis')
        plt.axis('off')
        
        if save_path:
            plt.savefig(save_path)
        plt.close()

class AttentionVisualizer:
    """
    Visualizes script-aware attention patterns as described in Section 3.2.
    """
    def plot_attention_weights(
        self,
        attention_weights: torch.Tensor,
        tokens: List[str],
        save_path: Optional[str] = None
    ):
        """Plot attention weight matrix with token labels."""
        plt.figure(figsize=(10, 10))
        
        sns.heatmap(
            attention_weights.cpu().numpy(),
            xticklabels=tokens,
            yticklabels=tokens,
            cmap='viridis'
        )
        
        plt.title('Script-Aware Attention Weights')
        plt.xlabel('Target Tokens')
        plt.ylabel('Source Tokens')
        
        if save_path:
            plt.savefig(save_path)
        plt.close()

class ScriptConfusionMatrix:
    """
    Visualizes script classification results as described in Section 5.2.
    """
    def plot_confusion_matrix(
        self,
        conf_matrix: np.ndarray,
        scripts: List[str],
        save_path: Optional[str] = None
    ):
        """Plot confusion matrix for script classification."""
        plt.figure(figsize=(10, 8))
        
        sns.heatmap(
            conf_matrix,
            annot=True,
            fmt='d',
            xticklabels=scripts,
            yticklabels=scripts,
            cmap='Blues'
        )
        
        plt.title('Script Classification Confusion Matrix')
        plt.xlabel('Predicted Script')
        plt.ylabel('True Script')
        
        if save_path:
            plt.savefig(save_path)
        plt.close()

class ResultsPlotter:
    """
    Generates performance visualization plots as described in Section 5.
    """
    def plot_metrics_over_time(
        self,
        metrics_history: Dict[str, List[float]],
        save_path: Optional[str] = None
    ):
        """Plot training metrics over time."""
        plt.figure(figsize=(12, 6))
        
        for metric_name, values in metrics_history.items():
            plt.plot(values, label=metric_name)
        
        plt.title('Training Metrics Over Time')
        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
        plt.close()
    
    def plot_language_pair_comparison(
        self,
        results: Dict[str, Dict[str, float]],
        metrics: List[str],
        save_path: Optional[str] = None
    ):
        """Plot performance comparison across language pairs."""
        plt.figure(figsize=(12, 6))
        
        x = np.arange(len(results))
        width = 0.8 / len(metrics)
        
        for i, metric in enumerate(metrics):
            values = [pair_results[metric] for pair_results in results.values()]
            plt.bar(x + i * width, values, width, label=metric)
        
        plt.title('Performance Across Language Pairs')
        plt.xlabel('Language Pair')
        plt.ylabel('Score')
        plt.xticks(x + width * (len(metrics) - 1) / 2, list(results.keys()))
        plt.legend()
        
        if save_path:
            plt.savefig(save_path)
        plt.close()
