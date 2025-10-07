"""
Training functions for RLS attention models
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

def train_epoch_rls(model, dataloader, optimizer, criterion, device):
    """Training epoch for RLS models"""
    model.train()
    total_loss, correct, total = 0., 0, 0
    
    for batch in dataloader:
        eeg = batch['eeg'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        logits, _ = model(eeg)  # (B, 1, 7)
        logits = logits.squeeze(1)  # (B, 7)

        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / len(dataloader), 100. * correct / total

def evaluate_rls(model, dataloader, criterion, device):
    """Evaluation function for RLS models"""
    model.eval()
    total_loss, correct, total = 0., 0, 0
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for batch in dataloader:
            eeg = batch['eeg'].to(device)
            labels = batch['label'].to(device)
            logits, _ = model(eeg)
            logits = logits.squeeze(1)
            loss = criterion(logits, labels)
            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    
    acc = 100.*correct/total
    f1 = f1_score(all_labels, all_preds, average='weighted')*100
    return total_loss/len(dataloader), acc, f1

def subject_dependent_experiment_rls(data_dir=".", subset_ratio=0.01):
    """Subject-dependent RLS experiment"""
    from utils.data_loader import SEEDVII_Dataset
    from models.rls_attention import Phase1RLSModel
    
    print("\n=== SUBJECT-DEPENDENT RLS EXPERIMENT ===")
    dataset = SEEDVII_Dataset(data_dir, modality='eeg', subset_ratio=subset_ratio)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    subjects = np.unique(dataset.subject_labels)
    subject_results = []
    
    for subj in subjects[:5]:  # Test first 5 subjects
        print(f"\nTraining subject {subj}...")
        
        # Get subject data
        subject_mask = dataset.subject_labels == subj
        subject_indices = np.where(subject_mask)[0]
        
        if len(subject_indices) < 20:
            continue
        
        # Split into train/test
        train_indices, test_indices = train_test_split(
            subject_indices, test_size=0.3, random_state=42)
        
        train_loader = DataLoader(Subset(dataset, train_indices), batch_size=16, shuffle=True)
        test_loader = DataLoader(Subset(dataset, test_indices), batch_size=16, shuffle=False)
        
        # Create and train model
        model = Phase1RLSModel(
            n_channels=dataset.eeg_feature_dim,
            d_model=256,
            n_classes=dataset.num_classes
        ).to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        
        # Training loop
        best_test_acc = 0.0
        for epoch in range(100):
            train_epoch_rls(model, train_loader, optimizer, criterion, device)
            _, test_acc, test_f1 = evaluate_rls(model, test_loader, criterion, device)
            best_test_acc = max(best_test_acc, test_acc)
            
            if epoch % 20 == 0:
                print(f" Epoch {epoch}: Test Acc {test_acc:.1f}%")
        
        subject_results.append(best_test_acc)
        print(f" Subject {subj} FINAL: {best_test_acc:.2f}%")
    
    if subject_results:
        avg_acc = np.mean(subject_results)
        std_acc = np.std(subject_results)
        print(f"\nSUBJECT-DEPENDENT RESULTS: {avg_acc:.2f}% ± {std_acc:.2f}%")
        return avg_acc, std_acc
    
    return 0.0, 0.0

def cross_subject_experiment_rls(data_dir=".", subset_ratio=0.01):
    """Cross-subject RLS experiment"""
    from utils.data_loader import SEEDVII_Dataset
    from models.rls_attention import Phase1RLSModel
    
    print("\n=== CROSS-SUBJECT RLS EXPERIMENT ===")
    dataset = SEEDVII_Dataset(data_dir, modality='eeg', subset_ratio=subset_ratio)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    subjects = np.unique(dataset.subject_labels)
    cross_subject_results = []
    
    for test_subj in subjects[:5]:  # Test first 5 subjects
        print(f"\nTesting on subject {test_subj}...")
        
        # Split train and test subjects
        train_indices = np.where(dataset.subject_labels != test_subj)[0]
        test_indices = np.where(dataset.subject_labels == test_subj)[0]
        
        if len(test_indices) < 10:
            continue
        
        train_loader = DataLoader(Subset(dataset, train_indices), batch_size=16, shuffle=True)
        test_loader = DataLoader(Subset(dataset, test_indices), batch_size=16, shuffle=False)
        
        # Create and train model
        model = Phase1RLSModel(
            n_channels=dataset.eeg_feature_dim,
            d_model=256,
            n_classes=dataset.num_classes
        ).to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        
        # Training loop
        for epoch in range(150):
            train_epoch_rls(model, train_loader, optimizer, criterion, device)
            
            if epoch % 30 == 0:
                _, test_acc, _ = evaluate_rls(model, test_loader, criterion, device)
                print(f" Epoch {epoch}: Test Acc {test_acc:.1f}%")
        
        # Final evaluation
        _, final_test_acc, final_f1 = evaluate_rls(model, test_loader, criterion, device)
        cross_subject_results.append(final_test_acc)
        print(f" Subject {test_subj} FINAL: {final_test_acc:.2f}%")
    
    if cross_subject_results:
        avg_acc = np.mean(cross_subject_results)
        std_acc = np.std(cross_subject_results)
        print(f"\nCROSS-SUBJECT RESULTS: {avg_acc:.2f}% ± {std_acc:.2f}%")
        return avg_acc, std_acc
    
    return 0.0, 0.0
