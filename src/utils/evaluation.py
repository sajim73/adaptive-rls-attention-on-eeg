"""
Evaluation utilities for RLS Attention models on SEED-VII dataset.
Contains comprehensive evaluation functions, metrics calculation, and analysis tools.
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import warnings

warnings.filterwarnings("ignore")

def evaluate_rls(model: nn.Module, dataloader: torch.utils.data.DataLoader, 
                criterion: nn.Module, device: torch.device) -> Tuple[float, float, float]:
    """
    Evaluate RLS model on given dataloader.

    Args:
        model: RLS model to evaluate
        dataloader: DataLoader containing evaluation data
        criterion: Loss function
        device: Device to run evaluation on

    Returns:
        Tuple of (average_loss, accuracy, f1_score)
    """
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in dataloader:
            eeg = batch['eeg'].to(device)
            labels = batch['label'].to(device)

            # Forward pass
            logits, _ = model(eeg)  # RLS models return (logits, features)
            logits = logits.squeeze(1)  # Remove sequence dimension if present

            # Calculate loss
            loss = criterion(logits, labels)
            total_loss += loss.item()

            # Calculate predictions
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    # Calculate metrics
    acc = 100.0 * correct / total
    f1 = f1_score(all_labels, all_preds, average='weighted') * 100

    return total_loss / len(dataloader), acc, f1


def evaluate_baseline_model(model: nn.Module, testloader: torch.utils.data.DataLoader, 
                           device: str = 'cuda') -> Tuple[float, float, np.ndarray, List, List]:
    """
    Evaluate baseline model and return detailed metrics.

    Args:
        model: Baseline model to evaluate
        testloader: Test data loader
        device: Device to run on

    Returns:
        Tuple of (accuracy, f1_score, confusion_matrix, predictions, labels)
    """
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in testloader:
            eeg = batch['eeg'].to(device)
            labels = batch['label'].to(device)
            outputs = model(eeg)

            predicted = torch.max(outputs, 1)[1]
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds) * 100
    f1 = f1_score(all_labels, all_labels, average='weighted') * 100
    cm = confusion_matrix(all_labels, all_preds)

    return accuracy, f1, cm, all_preds, all_labels


def evaluate_fixed_rls_model(model: nn.Module, testloader: torch.utils.data.DataLoader, 
                            device: str = 'cuda') -> Tuple[float, float]:
    """
    Evaluate fixed RLS model for cross-subject validation.

    Args:
        model: Fixed RLS model
        testloader: Test data loader  
        device: Device to run on

    Returns:
        Tuple of (accuracy, f1_score)
    """
    model.to(device).eval()
    all_p, all_y = [], []

    with torch.no_grad():
        for batch in testloader:
            x, y = batch['eeg'].to(device), batch['label'].to(device)
            pred = model(x)[0].argmax(1)  # Get logits and take argmax
            all_p += pred.cpu().tolist()
            all_y += y.cpu().tolist()

    return accuracy_score(all_y, all_p) * 100, f1_score(all_y, all_p, average='weighted') * 100


def calculate_comprehensive_metrics(all_labels: List, all_preds: List, 
                                  class_names: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Calculate comprehensive evaluation metrics.

    Args:
        all_labels: True labels
        all_preds: Predicted labels
        class_names: Optional class names for detailed reporting

    Returns:
        Dictionary containing various metrics
    """
    if class_names is None:
        class_names = [f'Class_{i}' for i in range(7)]  # SEED-VII has 7 emotions

    # Basic metrics
    accuracy = accuracy_score(all_labels, all_preds) * 100
    f1_weighted = f1_score(all_labels, all_preds, average='weighted') * 100
    f1_macro = f1_score(all_labels, all_preds, average='macro') * 100
    f1_micro = f1_score(all_labels, all_preds, average='micro') * 100

    # Per-class F1 scores
    f1_per_class = f1_score(all_labels, all_preds, average=None)

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)

    # Classification report
    report = classification_report(all_labels, all_preds, 
                                 target_names=class_names, 
                                 output_dict=True)

    return {
        'accuracy': accuracy,
        'f1_weighted': f1_weighted,
        'f1_macro': f1_macro,
        'f1_micro': f1_micro,
        'f1_per_class': f1_per_class,
        'confusion_matrix': cm,
        'classification_report': report
    }


def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], 
                         title: str = 'Confusion Matrix', figsize: Tuple[int, int] = (8, 6)):
    """
    Plot confusion matrix using seaborn.

    Args:
        cm: Confusion matrix
        class_names: Class names for labels
        title: Plot title
        figsize: Figure size
    """
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.show()


def analyze_dataset_quality(dataset) -> Dict[str, Any]:
    """
    Analyze dataset quality and potential issues.

    Args:
        dataset: SEEDVII dataset object

    Returns:
        Dictionary containing quality metrics
    """
    print("=" * 60)
    print("DATASET QUALITY ANALYSIS")
    print("=" * 60)

    # Basic statistics
    print(f"Total samples: {len(dataset.emotion_labels)}")
    print(f"EEG feature shape: {dataset.eeg_features.shape}")
    print(f"Number of subjects: {len(np.unique(dataset.subject_labels))}")
    print(f"Number of classes: {len(np.unique(dataset.emotion_labels))}")

    # Label distribution
    unique_labels, counts = np.unique(dataset.emotion_labels, return_counts=True)
    print(f"\nLabel distribution:")
    for label, count in zip(unique_labels, counts):
        percentage = (count / len(dataset.emotion_labels)) * 100
        print(f"  Class {label}: {count} samples ({percentage:.1f}%)")

    # Check for class imbalance
    min_count, max_count = counts.min(), counts.max()
    imbalance_ratio = max_count / min_count
    print(f"\nClass imbalance ratio: {imbalance_ratio:.2f}")
    if imbalance_ratio > 3:
        print("WARNING: Significant class imbalance detected!")

    # Feature statistics
    print(f"\nEEG feature statistics:")
    print(f"  Mean: {dataset.eeg_features.mean():.4f}")
    print(f"  Std: {dataset.eeg_features.std():.4f}")
    print(f"  Min: {dataset.eeg_features.min():.4f}")
    print(f"  Max: {dataset.eeg_features.max():.4f}")

    # Check for problematic features
    nan_count = np.isnan(dataset.eeg_features).sum()
    inf_count = np.isinf(dataset.eeg_features).sum()
    zero_variance_features = np.sum(dataset.eeg_features.std(axis=0) < 1e-6)

    print(f"\nData quality issues:")
    print(f"  NaN values: {nan_count}")
    print(f"  Inf values: {inf_count}")
    print(f"  Zero variance features: {zero_variance_features}")

    if nan_count > 0 or inf_count > 0:
        print("WARNING: NaN/Inf values detected in features!")

    return {
        'imbalance_ratio': imbalance_ratio,
        'nan_count': nan_count,
        'inf_count': inf_count,
        'zero_variance_features': zero_variance_features
    }


def cross_subject_evaluation(model_class, dataset, device: str = 'cuda', 
                           num_subjects: int = 5) -> List[float]:
    """
    Perform Leave-One-Subject-Out (LOSO) cross-subject evaluation.

    Args:
        model_class: Model class constructor
        dataset: Dataset object
        device: Device to run on
        num_subjects: Number of subjects to test (for speed)

    Returns:
        List of test accuracies for each subject
    """
    subjects = np.unique(dataset.subject_labels)
    results = []

    for test_subj in subjects[:num_subjects]:
        print(f"Testing subject {test_subj}...")

        # Create train/test splits
        train_indices = np.where(dataset.subject_labels != test_subj)[0]
        test_indices = np.where(dataset.subject_labels == test_subj)[0]

        if len(test_indices) < 5:
            continue

        # Create data loaders
        from torch.utils.data import DataLoader, Subset
        train_loader = DataLoader(Subset(dataset, train_indices), 
                                batch_size=64, shuffle=True)
        test_loader = DataLoader(Subset(dataset, test_indices), 
                               batch_size=64, shuffle=False)

        # Create and train model
        model = model_class(
            n_features=dataset.eeg_feature_dim,
            n_classes=dataset.num_classes,
            use_rls=True
        ).to(device)

        # Training code would go here
        # For now, we'll simulate evaluation
        acc, f1 = evaluate_fixed_rls_model(model, test_loader, device)
        results.append(acc)
        print(f"Subject {test_subj} Test Acc: {acc:.2f}%")

    return results


def compare_model_performance(results_dict: Dict[str, Tuple[float, float]], 
                            title: str = "Model Performance Comparison"):
    """
    Compare performance across different models.

    Args:
        results_dict: Dictionary mapping model names to (mean, std) tuples
        title: Comparison title
    """
    print("=" * 60)
    print(title)
    print("=" * 60)
    print(f"{'Model':<20} {'Val Acc':<10} {'Test Acc':<10} {'Test F1':<10}")
    print("-" * 60)

    for name, metrics in results_dict.items():
        if isinstance(metrics, dict):
            val_acc = metrics.get('val_acc', 0.0)
            test_acc = metrics.get('test_acc', 0.0) 
            test_f1 = metrics.get('test_f1', 0.0)
            print(f"{name:<20} {val_acc:<10.2f} {test_acc:<10.2f} {test_f1:<10.2f}")
        else:
            # Assume it's a tuple of (mean, std)
            mean_acc, std_acc = metrics
            print(f"{name:<20} {mean_acc:<10.2f} ±{std_acc:<8.2f}")


def statistical_significance_test(results1: List[float], results2: List[float], 
                                model1_name: str = "Model 1", 
                                model2_name: str = "Model 2") -> Dict[str, Any]:
    """
    Perform statistical significance test between two sets of results.

    Args:
        results1: Results from first model
        results2: Results from second model  
        model1_name: Name of first model
        model2_name: Name of second model

    Returns:
        Dictionary with test statistics
    """
    from scipy import stats

    # Paired t-test
    t_stat, p_value = stats.ttest_rel(results1, results2)

    # Effect size (Cohen's d)
    pooled_std = np.sqrt((np.var(results1) + np.var(results2)) / 2)
    cohens_d = (np.mean(results1) - np.mean(results2)) / pooled_std

    print(f"\nStatistical Significance Test:")
    print(f"{model1_name}: {np.mean(results1):.2f} ± {np.std(results1):.2f}")
    print(f"{model2_name}: {np.mean(results2):.2f} ± {np.std(results2):.2f}")
    print(f"t-statistic: {t_stat:.3f}")
    print(f"p-value: {p_value:.3f}")
    print(f"Cohen's d: {cohens_d:.3f}")

    if p_value < 0.05:
        print("Result: Statistically significant difference (p < 0.05)")
    else:
        print("Result: No statistically significant difference (p >= 0.05)")

    return {
        't_statistic': t_stat,
        'p_value': p_value,
        'cohens_d': cohens_d,
        'significant': p_value < 0.05
    }


def generate_evaluation_report(results: Dict[str, Any], 
                             dataset_info: Dict[str, Any],
                             save_path: Optional[str] = None) -> str:
    """
    Generate comprehensive evaluation report.

    Args:
        results: Dictionary containing all results
        dataset_info: Dataset information
        save_path: Optional path to save report

    Returns:
        Report string
    """
    report = []
    report.append("=" * 80)
    report.append("COMPREHENSIVE EVALUATION REPORT")
    report.append("=" * 80)

    # Dataset information
    report.append("\n## Dataset Information")
    report.append("-" * 40)
    for key, value in dataset_info.items():
        report.append(f"{key}: {value}")

    # Model performance
    report.append("\n## Model Performance")
    report.append("-" * 40)
    for model_name, metrics in results.items():
        report.append(f"\n### {model_name}")
        if isinstance(metrics, dict):
            for metric, value in metrics.items():
                if isinstance(value, float):
                    report.append(f"  {metric}: {value:.2f}")
                else:
                    report.append(f"  {metric}: {value}")

    # Conclusions
    report.append("\n## Conclusions")
    report.append("-" * 40)
    report.append("1. Results analysis...")
    report.append("2. Best performing model...")
    report.append("3. Recommendations for improvement...")

    report_text = "\n".join(report)

    if save_path:
        with open(save_path, 'w') as f:
            f.write(report_text)
        print(f"Report saved to {save_path}")

    return report_text


# Utility functions for specific evaluations
def evaluate_hyperparameter_search_results(search_results: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze hyperparameter search results.

    Args:
        search_results: DataFrame with hyperparameter search results

    Returns:
        Analysis dictionary
    """
    best_result = search_results.loc[search_results['val_acc'].idxmax()]

    print("Hyperparameter Search Analysis:")
    print("-" * 40)
    print(f"Best validation accuracy: {best_result['val_acc']:.2f}%")
    print(f"Best parameters:")
    for param in ['lambda_reg', 'forgetting_factor', 'd_model', 'learning_rate']:
        if param in best_result:
            print(f"  {param}: {best_result[param]}")

    # Parameter sensitivity analysis
    param_importance = {}
    for param in ['lambda_reg', 'forgetting_factor', 'd_model', 'learning_rate']:
        if param in search_results.columns:
            correlation = search_results[param].corr(search_results['val_acc'])
            param_importance[param] = abs(correlation)

    print(f"\nParameter importance (correlation with accuracy):")
    for param, importance in sorted(param_importance.items(), 
                                  key=lambda x: x[1], reverse=True):
        print(f"  {param}: {importance:.3f}")

    return {
        'best_params': best_result.to_dict(),
        'param_importance': param_importance
    }
