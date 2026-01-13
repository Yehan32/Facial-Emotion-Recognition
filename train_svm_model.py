"""
SVM Model Training for Facial Emotion Recognition
Trains Support Vector Machine classifiers on HOG features
"""

import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    accuracy_score,
    precision_recall_fscore_support
)
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings('ignore')

# CONFIGURATION

# SVM Parameters (you can experiment with these!)
SVM_KERNEL = 'rbf'          # Try: 'rbf' or 'linear'
SVM_C = 1.0                # Regularization (try: 0.1, 1.0, 10)
SVM_GAMMA = 'scale'         # For RBF kernel (try: 'scale', 0.001, 0.01)

# Whether to use Grid Search for hyperparameter tuning
USE_GRID_SEARCH = True      # Set to True for automatic tuning

# LOAD FEATURES

def load_features(features_file):
    """
    Load extracted HOG features from pickle file
    
    Args:
        features_file: Path to .pkl file with features
        
    Returns:
        Dictionary with train/test features and labels
    """
    print(f"\nLoading features from: {features_file}")
    
    with open(features_file, 'rb') as f:
        data = pickle.load(f)
    
    print(f"  Training samples: {data['train_features'].shape}")
    print(f"  Test samples: {data['test_features'].shape}")
    print(f"  Emotion classes: {data['label_names']}")
    
    return data

# TRAIN SVM MODEL

def train_svm_with_grid_search(X_train, y_train):
    """
    Train SVM with Grid Search to find best parameters
    
    Args:
        X_train: Training features
        y_train: Training labels
        
    Returns:
        Best trained SVM model
    """
    print("\n" + "="*60)
    print("GRID SEARCH FOR BEST PARAMETERS")
    print("="*60)
    
    # Define parameter grid to search
    param_grid = {
        'C': [0.1, 1, 10, 100],              # Regularization
        'gamma': ['scale', 0.001, 0.01, 0.1], # Kernel coefficient
        'kernel': ['rbf', 'linear']           # Kernel type
    }
    
    print("\nSearching through parameter combinations:")
    print(f"  C values: {param_grid['C']}")
    print(f"  Gamma values: {param_grid['gamma']}")
    print(f"  Kernel types: {param_grid['kernel']}")
    print(f"\nTotal combinations: {len(param_grid['C']) * len(param_grid['gamma']) * len(param_grid['kernel'])}")
    
    # Create base SVM
    svm = SVC(random_state=42)
    
    # Grid Search with Cross-Validation
    print("\nPerforming Grid Search (this may take a few minutes)...")
    grid_search = GridSearchCV(
        svm,
        param_grid,
        cv=5,              # 5-fold cross-validation
        scoring='accuracy',
        verbose=1,
        n_jobs=-1          # Use all CPU cores
    )
    
    # Fit grid search
    grid_search.fit(X_train, y_train)
    
    # Print results
    print("\n" + "="*60)
    print("GRID SEARCH RESULTS")
    print("="*60)
    print(f"\nBest parameters found:")
    for param, value in grid_search.best_params_.items():
        print(f"  {param}: {value}")
    
    print(f"\nBest cross-validation accuracy: {grid_search.best_score_:.4f}")
    
    # Return best model
    return grid_search.best_estimator_, grid_search.best_params_


def train_svm_manual(X_train, y_train, kernel='rbf', C=1.0, gamma='scale'):
    """
    Train SVM with manually specified parameters
    
    Args:
        X_train: Training features
        y_train: Training labels
        kernel: Kernel type ('rbf' or 'linear')
        C: Regularization parameter
        gamma: Kernel coefficient
        
    Returns:
        Trained SVM model
    """
    print("\n" + "="*60)
    print("TRAINING SVM WITH MANUAL PARAMETERS")
    print("="*60)
    
    print(f"\nParameters:")
    print(f"  Kernel: {kernel}")
    print(f"  C: {C}")
    print(f"  Gamma: {gamma}")
    
    # Create and train SVM
    svm = SVC(
        kernel=kernel,
        C=C,
        gamma=gamma,
        random_state=42
    )
    
    print(f"\nTraining on {X_train.shape[0]} samples...")
    svm.fit(X_train, y_train)
    
    print("Training complete!")
    
    return svm, {'kernel': kernel, 'C': C, 'gamma': gamma}

# EVALUATE MODEL

def evaluate_model(model, X_train, y_train, X_test, y_test, label_names):
    """
    Evaluate trained model and print metrics
    
    Args:
        model: Trained SVM model
        X_train, y_train: Training data
        X_test, y_test: Test data
        label_names: List of emotion names
        
    Returns:
        Dictionary with all evaluation metrics
    """
    print("\n" + "="*60)
    print("MODEL EVALUATION")
    print("="*60)
    
    # Predict on training set
    y_train_pred = model.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    
    # Predict on test set
    y_test_pred = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    
    print(f"\n Accuracy Scores:")
    print(f"  Training Accuracy: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
    print(f"  Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    
    # Calculate per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, y_test_pred, average=None, labels=range(len(label_names))
    )
    
    print(f"\n Per-Class Performance on Test Set:")
    print("-" * 60)
    print(f"{'Emotion':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support'}")
    print("-" * 60)
    for i, emotion in enumerate(label_names):
        print(f"{emotion:<12} {precision[i]:<12.3f} {recall[i]:<12.3f} {f1[i]:<12.3f} {support[i]}")
    print("-" * 60)
    
    # Overall metrics
    avg_precision = np.mean(precision)
    avg_recall = np.mean(recall)
    avg_f1 = np.mean(f1)
    
    print(f"\n Average Metrics:")
    print(f"  Precision: {avg_precision:.4f}")
    print(f"  Recall: {avg_recall:.4f}")
    print(f"  F1-Score: {avg_f1:.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_test_pred)
    
    # Classification report
    class_report = classification_report(
        y_test, y_test_pred, 
        target_names=label_names,
        digits=3
    )
    
    results = {
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'y_test_pred': y_test_pred,
        'confusion_matrix': cm,
        'classification_report': class_report,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'support': support
    }
    
    return results

# VISUALIZATIONS

def plot_confusion_matrix(cm, label_names, dataset_name, save_path):
    """
    Create and save confusion matrix visualization
    
    Args:
        cm: Confusion matrix
        label_names: List of emotion names
        dataset_name: Name of dataset (for title)
        save_path: Where to save the plot
    """
    plt.figure(figsize=(10, 8))
    
    # Create heatmap
    sns.heatmap(
        cm, 
        annot=True,          # Show numbers
        fmt='d',             # Integer format
        cmap='Blues',        # Color scheme
        xticklabels=label_names,
        yticklabels=label_names,
        cbar_kws={'label': 'Count'}
    )
    
    plt.title(f'Confusion Matrix - {dataset_name}', fontsize=16, fontweight='bold')
    plt.ylabel('True Emotion', fontsize=12)
    plt.xlabel('Predicted Emotion', fontsize=12)
    
    # Add accuracy text
    accuracy = np.trace(cm) / np.sum(cm)
    plt.text(
        len(label_names)/2, -0.5, 
        f'Accuracy: {accuracy:.2%}',
        ha='center', fontsize=12, fontweight='bold'
    )
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f" Confusion matrix saved to: {save_path}")


def plot_per_class_metrics(precision, recall, f1, label_names, dataset_name, save_path):
    """
    Create bar chart showing per-class metrics
    
    Args:
        precision, recall, f1: Arrays of metrics per class
        label_names: List of emotion names
        dataset_name: Name of dataset
        save_path: Where to save the plot
    """
    x = np.arange(len(label_names))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Create bars
    bars1 = ax.bar(x - width, precision, width, label='Precision', color='skyblue')
    bars2 = ax.bar(x, recall, width, label='Recall', color='lightgreen')
    bars3 = ax.bar(x + width, f1, width, label='F1-Score', color='salmon')
    
    # Customize plot
    ax.set_xlabel('Emotion', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title(f'Per-Class Performance Metrics - {dataset_name}', 
                fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(label_names, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim([0, 1.1])
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    def add_value_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}',
                   ha='center', va='bottom', fontsize=8)
    
    add_value_labels(bars1)
    add_value_labels(bars2)
    add_value_labels(bars3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f" Metrics chart saved to: {save_path}")


def save_classification_report(report_text, dataset_name, save_path):
    """
    Save classification report to text file
    
    Args:
        report_text: Classification report string
        dataset_name: Name of dataset
        save_path: Where to save the file
    """
    with open(save_path, 'w') as f:
        f.write(f"Classification Report - {dataset_name}\n")
        f.write("="*60 + "\n\n")
        f.write(report_text)
    
    print(f" Classification report saved to: {save_path}")

# TRAIN AND EVALUATE COMPLETE PIPELINE

def train_and_evaluate_dataset(features_file, dataset_name, use_grid_search=True):
    """
    Complete training and evaluation pipeline for one dataset
    
    Args:
        features_file: Path to features .pkl file
        dataset_name: Name of dataset (e.g., 'CK' or 'JAFFE')
        use_grid_search: Whether to use grid search for tuning
        
    Returns:
        Trained model and results
    """
    print("\n" + "="*70)
    print(f"TRAINING AND EVALUATING: {dataset_name} DATASET")
    print("="*70)
    
    # Load features
    data = load_features(features_file)
    
    X_train = data['train_features']
    y_train = data['train_labels']
    X_test = data['test_features']
    y_test = data['test_labels']
    label_names = data['label_names']
    
    # Train model
    if use_grid_search:
        model, best_params = train_svm_with_grid_search(X_train, y_train)
    else:
        model, best_params = train_svm_manual(
            X_train, y_train, 
            kernel=SVM_KERNEL, 
            C=SVM_C, 
            gamma=SVM_GAMMA
        )
    
    # Evaluate model
    results = evaluate_model(model, X_train, y_train, X_test, y_test, label_names)
    
    # Create visualizations
    print("\n" + "="*60)
    print("CREATING VISUALIZATIONS")
    print("="*60)
    
    # Confusion matrix
    plot_confusion_matrix(
        results['confusion_matrix'],
        label_names,
        dataset_name,
        f'{dataset_name}_confusion_matrix.png'
    )
    
    # Per-class metrics
    plot_per_class_metrics(
        results['precision'],
        results['recall'],
        results['f1'],
        label_names,
        dataset_name,
        f'{dataset_name}_metrics.png'
    )
    
    # Save classification report
    save_classification_report(
        results['classification_report'],
        dataset_name,
        f'{dataset_name}_classification_report.txt'
    )
    
    # Save trained model
    model_file = f'{dataset_name}_svm_model.pkl'
    with open(model_file, 'wb') as f:
        pickle.dump({
            'model': model,
            'parameters': best_params,
            'label_names': label_names,
            'results': results
        }, f)
    
    print(f" Model saved to: {model_file}")
    
    return model, results


def compare_datasets(ck_results, jaffe_results):
    """
    Create comparison visualization between CK and JAFFE results
    
    Args:
        ck_results: Results dictionary for CK dataset
        jaffe_results: Results dictionary for JAFFE dataset
    """
    print("\n" + "="*60)
    print("CREATING DATASET COMPARISON")
    print("="*60)
    
    # Prepare data
    datasets = ['CK', 'JAFFE']
    train_acc = [ck_results['train_accuracy'], jaffe_results['train_accuracy']]
    test_acc = [ck_results['test_accuracy'], jaffe_results['test_accuracy']]
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Accuracy comparison
    x = np.arange(len(datasets))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, train_acc, width, label='Training', color='skyblue')
    bars2 = ax1.bar(x + width/2, test_acc, width, label='Testing', color='coral')
    
    ax1.set_xlabel('Dataset', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax1.set_title('Accuracy Comparison', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(datasets)
    ax1.legend()
    ax1.set_ylim([0, 1.1])
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontsize=10)
    
    # Sample counts comparison
    ck_train = ck_results['confusion_matrix'].sum()
    ck_test = len(ck_results['y_test_pred'])
    jaffe_train = jaffe_results['confusion_matrix'].sum()
    jaffe_test = len(jaffe_results['y_test_pred'])
    
    train_samples = [ck_train, jaffe_train]
    test_samples = [ck_test, jaffe_test]
    
    bars3 = ax2.bar(x - width/2, train_samples, width, label='Training', color='lightgreen')
    bars4 = ax2.bar(x + width/2, test_samples, width, label='Testing', color='plum')
    
    ax2.set_xlabel('Dataset', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Number of Samples', fontsize=12, fontweight='bold')
    ax2.set_title('Dataset Size Comparison', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(datasets)
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bars in [bars3, bars4]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('dataset_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(" Comparison saved to: dataset_comparison.png")

# MAIN EXECUTION

if __name__ == "__main__":
    print("\n" + "="*70)
    print("SVM MODEL TRAINING FOR FACIAL EMOTION RECOGNITION")
    print("="*70)
    
    # Train on CK dataset
    ck_model, ck_results = train_and_evaluate_dataset(
        features_file='CK_features.pkl',
        dataset_name='CK',
        use_grid_search=USE_GRID_SEARCH
    )
    
    # Train on JAFFE dataset
    jaffe_model, jaffe_results = train_and_evaluate_dataset(
        features_file='JAFFE_features.pkl',
        dataset_name='JAFFE',
        use_grid_search=USE_GRID_SEARCH
    )
    
    # Compare both datasets
    compare_datasets(ck_results, jaffe_results)
    
    # Final summary
    print("\n" + "="*70)
    print("Training complete!")
    print("="*70)
    
    print("\n Generated Files:")
    print("  Models:")
    print("    - CK_svm_model.pkl")
    print("    - JAFFE_svm_model.pkl")
    print("\n  Visualizations:")
    print("    - CK_confusion_matrix.png")
    print("    - JAFFE_confusion_matrix.png")
    print("    - CK_metrics.png")
    print("    - JAFFE_metrics.png")
    print("    - dataset_comparison.png")
    print("\n  Reports:")
    print("    - CK_classification_report.txt")
    print("    - JAFFE_classification_report.txt")
    
    print("\n Final Results Summary:")
    print("-" * 70)
    print(f"{'Dataset':<15} {'Train Acc':<15} {'Test Acc':<15}")
    print("-" * 70)
    print(f"{'CK':<15} {ck_results['train_accuracy']:.4f} ({ck_results['train_accuracy']*100:.2f}%){'':<3} {ck_results['test_accuracy']:.4f} ({ck_results['test_accuracy']*100:.2f}%)")
    print(f"{'JAFFE':<15} {jaffe_results['train_accuracy']:.4f} ({jaffe_results['train_accuracy']*100:.2f}%){'':<3} {jaffe_results['test_accuracy']:.4f} ({jaffe_results['test_accuracy']*100:.2f}%)")
    print("-" * 70)
    
    print("="*70)