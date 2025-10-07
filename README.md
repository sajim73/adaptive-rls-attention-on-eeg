# Adaptive Hierarchical Least Squares Attention Networks

**Mathematical Framework for Multi-Scale EEG Emotion Recognition**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Work%20in%20Progress-yellow.svg)]()

> **Revolutionary approach to attention mechanisms**: Replacing traditional softmax-based attention with adaptive hierarchical least squares optimization for enhanced EEG emotion recognition.

---

## 🧠 **Abstract**

This repository implements **Adaptive Hierarchical Least Squares Attention (AHLSA)**, a novel mathematical framework that reformulates attention mechanisms as least squares optimization problems. Unlike traditional transformer attention which forces artificial probability constraints, our approach uses honest expert scoring to dynamically focus on truly important brain signal patterns for emotion recognition.

## 🔬 **The Core Innovation**

### **Traditional Attention Problems**
- **Attention Dispersion**: As sequence length increases, important signals get diluted
- **Artificial Constraints**: Forced probability distributions (sum to 1) ignore signal importance
- **No Uncertainty Quantification**: Fake probabilities with no confidence measures

### **Our Solution: Three-Expert System**

Instead of one overwhelmed "reader," we create specialized expert teams:

1. **🕐 Temporal Expert** (RLS-based): Tracks brain wave patterns over time
2. **🗺️ Spatial Expert** (Weighted LS): Analyzes electrode relationships  
3. **🔗 Integration Expert** (LASSO): Selects emotion-relevant features

## 📊 **Mathematical Framework**

### **Level 1: Temporal RLS Attention**
```math
min_{θt} ||A_t θ_t - b_t||^2 + λ_1||θ_t||^2
```
*Updates attention weights recursively as new brain data arrives*

### **Level 2: Spatial Weighted LS**  
```math
min_{φk} ||W_k S_k φ_k - r_k||^2
```
*Accounts for electrode reliability and brain connectivity*

### **Level 3: Cross-Scale LASSO Integration**
```math
min_Ψ ||HΨ - y||^2 + λ_2||Ψ||_1
```
*Selects only emotion-relevant patterns, ignoring noise*

## 🚀 **Key Advantages**

- **⚡ Real-time Processing**: O(C²) complexity vs O(T×C²) for standard attention  
- **📈 Honest Scoring**: No artificial percentage constraints - signals get attention they deserve
- **🎯 Uncertainty Quantification**: Actual confidence levels, not fake probabilities
- **🧩 Multi-scale Learning**: Specialized experts for different brain signal aspects
- **💡 Interpretable**: Mathematical foundation provides clear explanations

## 📁 **Repository Structure**

```
adaptive-rls-attention-eeg/
├── src/
│   ├── models/
│   │   ├── rls_attention.py          # Core AHLSA implementation
│   │   └── baseline_models.py        # Comparison baselines
│   ├── training/
│   │   ├── train_rls.py             # Training pipelines
│   │   └── hyperparameter_search.py # Parameter optimization
│   └── utils/
│       ├── data_loader.py           # SEED-VII dataset handling
│       └── evaluation.py            # Metrics and visualization
├── experiments/
│   ├── parameter_optimization.ipynb  # Hyperparameter experiments
│   └── single_subject_experiments.ipynb # Subject-dependent analysis
├── configs/
│   └── default_config.yaml          # Model configurations
├── results/                         # Experimental results
└── docs/                           # Documentation
```

## 🛠️ **Installation & Setup**

### **Prerequisites**
```bash
# Create environment
conda create -n rls-attention python=3.11
conda activate rls-attention

# Install dependencies
pip install -r requirements.txt
```

### **Quick Start**
```python
from src.models.rls_attention import Phase1RLSModel

# Initialize model
model = Phase1RLSModel(d_model=128, num_classes=7)

# Process EEG data (B, T, D)
eeg_data = torch.randn(32, 128, 128)  
emotions = model(eeg_data)
```

## 📚 **Datasets**

### **SEED-VII Dataset**
- **Subjects**: 15 participants
- **Emotions**: 7 emotional states  
- **Channels**: 62 EEG electrodes
- **Sampling**: High-resolution temporal data

*Prior experience with SEED-IV/VII datasets accelerates research protocols*

## ⚖️ **Baseline Comparisons**

- **Standard Transformer Attention** (Vaswani et al.)
- **CNN-LSTM Networks** 
- **MAET Implementation** (Multimodal Attention)
- **Graph Neural Networks for EEG**

## 📈 **Evaluation Metrics**

- **Accuracy**: Emotion classification performance
- **Computational Efficiency**: FLOPs, memory usage, inference time  
- **Interpretability**: Attention visualization, confidence intervals
- **Ablation Studies**: Each least squares component contribution

## 🔬 **Expected Contributions**

### **Mathematical Innovations**
- First attention mechanism reformulated as adaptive least squares
- Hierarchical regularization theory with convergence guarantees
- Novel multi-scale learning framework

### **Practical Impact** 
- Real-time EEG processing capabilities
- Uncertainty quantification for clinical applications
- Interpretable AI with mathematical grounding
- Robustness to EEG artifacts and missing channels

## 🎯 **Why This Approach Works**

### **Traditional Softmax Analogy**
*Like being forced to split pizza equally among friends regardless of hunger levels*

### **Our Least Squares Approach** 
*Like having expert judges give honest scores - someone scoring 95/100 gets more attention than someone scoring 3/100*

**Result**: AI focuses on what actually matters for emotion recognition, not artificial percentage splits.

## 🔄 **Development Branches**

- **`main`**: Stable implementation
- **`development`**: Active development  
- **`experiments/parameter-tuning`**: Hyperparameter optimization
- **`experiments/cross-subject`**: Cross-subject validation

## 📖 **Literature Foundation**

**Building on Recent Work (2024-2025):**
- Least Squares-Attention mathematical equivalence (Goulet Coulombe, 2025)
- Hierarchical attention mechanisms (various authors)
- EEG emotion recognition advances
- Recursive least squares improvements

**Novel Contributions:**
- First practical superior implementation
- EEG-specific hierarchical design
- Real-time adaptive capabilities  
- Uncertainty quantification integration

## 🚧 **Current Status**

**Phase 1**: ✅ Mathematical framework development  
**Phase 2**: 🔄 Implementation and testing (current)  
**Phase 3**: ⏳ Experimental validation  
**Phase 4**: ⏳ Clinical applications

## 🤝 **Contributing**

This is an active research project. Contributions welcome for:
- Algorithm optimizations
- Additional baseline implementations  
- Experimental protocols
- Documentation improvements

## 📄 **Citation**

```bibtex
@misc{ahmed2025ahlsa,
    title={Adaptive Hierarchical Least Squares Attention Networks: Mathematical Framework for Multi-Scale EEG Emotion Recognition},
    author={Sajim Ahmed},
    year={2025},
    note={Work in Progress}
}
```

## 📧 **Contact**

**Sajim Ahmed**  
Computer Science Student | AI Research  
[GitHub](https://github.com/your-username) | [Email](mailto:your.email@example.com)

---

**🔬 Research Philosophy**: *"Why force every signal to get attention when we can honestly score what matters?"*

This research represents the convergence of classical optimization theory with modern deep learning, specifically targeting EEG-based emotion recognition challenges while opening new avenues for interpretable, efficient AI systems.
