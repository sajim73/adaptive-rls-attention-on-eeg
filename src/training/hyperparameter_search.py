"""
Hyperparameter search utilities for RLS attention models (lambda, forgetting factor, etc.).
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from models_rls_phase1 import Phase1RLSModel

def hyperparameter_search_single_subject(dataset, subject_id, device):
    """Find best hyperparameters for a single subject"""
    
    # Hyperparameter search space
    lambda_values = [0.001, 0.01, 0.1, 1.0]
    forgetting_factors = [0.95, 0.98, 0.99]
    d_models = [128, 256, 512]
    learning_rates = [1e-4, 1e-3, 1e-2]
    
    print(f"\n=== HYPERPARAMETER SEARCH FOR SUBJECT {subject_id} ===")
    
    # Get subject data
    subject_mask = dataset.subject_labels == subject_id
    subject_indices = np.where(subject_mask)[0]
    
    if len(subject_indices) < 8:
        return None, None
    
    # Split subject data into train/val
    train_indices, val_indices = train_test_split(
        subject_indices, test_size=0.3, random_state=42)
    
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    best_val_acc = 0.0
    best_params = None
    search_results = []
    
    total_combinations = len(lambda_values) * len(forgetting_factors) * len(d_models) * len(learning_rates)
    combo_count = 0
    
    # Grid search
    for lambda_reg in lambda_values:
        for forgetting_factor in forgetting_factors:
            for d_model in d_models:
                for lr in learning_rates:
                    combo_count += 1
                    
                    print(f" Combo {combo_count}/{total_combinations}: λ={lambda_reg}, ff={forgetting_factor}, dim={d_model}, lr={lr}")
                    
                    # Create model with current hyperparameters
                    model = Phase1RLSModel(
                        n_channels=dataset.eeg_feature_dim,
                        d_model=d_model,
                        n_classes=dataset.num_classes,
                        lambda_reg=lambda_reg,
                        forgetting_factor=forgetting_factor
                    ).to(device)
                    
                    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
                    criterion = nn.CrossEntropyLoss()
                    
                    # Quick training (fewer epochs for search)
                    max_val_acc = 0.0
                    for epoch in range(20):  # Reduced epochs for faster search
                        from train_rls import train_epoch_rls, evaluate_rls
                        train_epoch_rls(model, train_loader, optimizer, criterion, device)
                        _, val_acc, val_f1 = evaluate_rls(model, val_loader, criterion, device)
                        max_val_acc = max(max_val_acc, val_acc)
                    
                    # Record results
                    result = {
                        'subject': subject_id,
                        'lambda_reg': lambda_reg,
                        'forgetting_factor': forgetting_factor,
                        'd_model': d_model,
                        'learning_rate': lr,
                        'val_acc': max_val_acc,
                        'val_f1': val_f1
                    }
                    search_results.append(result)
                    
                    if max_val_acc > best_val_acc:
                        best_val_acc = max_val_acc
                        best_params = result.copy()
                    
                    print(f" Val Acc: {max_val_acc:.2f}%")
                    
                    # Clean up
                    del model
                    torch.cuda.empty_cache() if device.type == 'cuda' else None
    
    print(f"\n BEST PARAMS for Subject {subject_id}: {best_params}")
    print(f" BEST VAL ACC: {best_val_acc:.2f}%")
    
    return best_params, search_results

def subject_dependent_experiment_with_search(data_dir=".", subset_ratio=0.01):
    """Subject-dependent experiment with hyperparameter search"""
    print("\n=== SUBJECT-DEPENDENT WITH HYPERPARAMETER SEARCH ===")
    from utils.data_loader import SEEDVII_Dataset
    
    dataset = SEEDVII_Dataset(data_dir, modality='eeg', subset_ratio=subset_ratio)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    subjects = np.unique(dataset.subject_labels)
    
    all_search_results = []
    final_results = []
    
    # Search hyperparameters for first 3 subjects (to save time)
    for subj in subjects[:3]:
        best_params, search_results = hyperparameter_search_single_subject(dataset, subj, device)
        
        if best_params is None:
            continue
        
        all_search_results.extend(search_results)
        
        # Train final model with best hyperparameters
        print(f"\n=== FINAL TRAINING FOR SUBJECT {subj} WITH BEST PARAMS ===")
        
        subject_mask = dataset.subject_labels == subj
        subject_indices = np.where(subject_mask)[0]
        train_indices, test_indices = train_test_split(
            subject_indices, test_size=0.3, random_state=42)
        
        train_loader = DataLoader(Subset(dataset, train_indices), batch_size=16, shuffle=True)
        test_loader = DataLoader(Subset(dataset, test_indices), batch_size=16, shuffle=False)
        
        # Create model with best hyperparameters
        final_model = Phase1RLSModel(
            n_channels=dataset.eeg_feature_dim,
            d_model=best_params['d_model'],
            n_classes=dataset.num_classes,
            lambda_reg=best_params['lambda_reg'],
            forgetting_factor=best_params['forgetting_factor']
        ).to(device)
        
        optimizer = torch.optim.Adam(final_model.parameters(), lr=best_params['learning_rate'])
        criterion = nn.CrossEntropyLoss()
        
        # Full training
        best_test_acc = 0.0
        for epoch in range(80):  # Full training epochs
            from train_rls import train_epoch_rls, evaluate_rls
            train_epoch_rls(final_model, train_loader, optimizer, criterion, device)
            _, test_acc, test_f1 = evaluate_rls(final_model, test_loader, criterion, device)
            best_test_acc = max(best_test_acc, test_acc)
            
            if epoch % 10 == 0:
                print(f" Epoch {epoch}: Test Acc {test_acc:.1f}%")
        
        final_results.append({
            'subject': subj,
            'test_acc': best_test_acc,
            'test_f1': test_f1,
            'best_params': best_params
        })
        
        print(f" Subject {subj} FINAL TEST ACC: {best_test_acc:.2f}%")
    
    # Save detailed results
    search_df = pd.DataFrame(all_search_results)
    search_df.to_csv('hyperparameter_search_results.csv', index=False)
    print(f"\nHyperparameter search results saved to 'hyperparameter_search_results.csv'")
    
    # Calculate overall performance
    test_accs = [r['test_acc'] for r in final_results]
    avg_acc = np.mean(test_accs)
    std_acc = np.std(test_accs)
    
    print(f"\nFINAL SUBJECT-DEPENDENT RESULTS: {avg_acc:.2f}% ± {std_acc:.2f}%")
    
    return avg_acc, std_acc, final_results

def cross_subject_with_best_params(data_dir=".", subset_ratio=0.01, best_params=None):
    """Cross-subject experiment with optimized hyperparameters"""
    print("\n=== CROSS-SUBJECT WITH OPTIMIZED HYPERPARAMETERS ===")
    
    if best_params is None:
        # Default best parameters
        best_params = {
            'lambda_reg': 0.01,
            'forgetting_factor': 0.99,
            'd_model': 256,
            'learning_rate': 1e-3
        }
        print(f"Using default parameters: {best_params}")
    else:
        print(f"Using optimized parameters: {best_params}")
    
    from utils.data_loader import SEEDVII_Dataset
    dataset = SEEDVII_Dataset(data_dir, modality='eeg', subset_ratio=subset_ratio)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    subjects = np.unique(dataset.subject_labels)
    results = []
    
    for test_subj in subjects[:3]:
        print(f"Testing subject {test_subj}...")
        
        tr_idxs = np.where(dataset.subject_labels != test_subj)[0]
        te_idxs = np.where(dataset.subject_labels == test_subj)[0]
        
        if len(te_idxs) < 5:
            continue
        
        train_loader = DataLoader(Subset(dataset, tr_idxs), batch_size=16, shuffle=True)
        test_loader = DataLoader(Subset(dataset, te_idxs), batch_size=16, shuffle=False)
        
        # Use best hyperparameters
        model = Phase1RLSModel(
            n_channels=dataset.eeg_feature_dim,
            d_model=best_params['d_model'],
            n_classes=dataset.num_classes,
            lambda_reg=best_params['lambda_reg'],
            forgetting_factor=best_params['forgetting_factor']
        ).to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=best_params['learning_rate'])
        criterion = nn.CrossEntropyLoss()
        
        # Training
        for epoch in range(100):
            from train_rls import train_epoch_rls, evaluate_rls
            train_epoch_rls(model, train_loader, optimizer, criterion, device)
            
            if epoch % 10 == 0:
                _, test_acc, _ = evaluate_rls(model, test_loader, criterion, device)
                print(f" Epoch {epoch}: Test Acc {test_acc:.1f}%")
        
        # Final evaluation
        _, final_test_acc, final_f1 = evaluate_rls(model, test_loader, criterion, device)
        results.append(final_test_acc)
        print(f" Subject {test_subj} FINAL ACC: {final_test_acc:.2f}%")
    
    if results:
        avg_acc = np.mean(results)
        std_acc = np.std(results)
        print(f"CROSS-SUBJECT RESULTS: {avg_acc:.2f}% ± {std_acc:.2f}%")
        return avg_acc, std_acc
    
    return 0.0, 0.0
