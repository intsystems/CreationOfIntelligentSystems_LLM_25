# Testing the Manifold Hypothesis in the Embedding Subspaces of Large Language Models

A comprehensive research project investigating the manifold hypothesis in the embedding subspaces of large language models, with implications for understanding model behavior and improving interpretability.

## About This Project

This research explores whether embedding subspaces of large language models exhibit manifold structure. Understanding the geometric properties of these embeddings can provide insights into model behavior, generalization patterns, and potential improvements for model architecture and training procedures.

| | |
|---|---|
| **Research Topic** | Testing the Manifold Hypothesis in the Embedding Subspaces of Large Language Models |
| **Research Type** | Creation of Intelligent Systems (CoIS) |
| **Authors** | Ivan Papai, Vladislav Minashkin, Fedor Sobolevsky, Vladislav Meshkov |
| **Supervisor** | Andrey Grabovoy |

## Features

- **Manifold Hypothesis Testing**: Statistical analysis of embedding subspace geometry
- **Multi-layer Analysis**: Investigation across different transformer layers
- **Dimensionality Reduction**: Advanced techniques for high-dimensional visualization
- **Synthetic Validation**: Testing framework with controlled manifold structures
- **Comprehensive Visualization**: Interactive tools for embedding space exploration
- **Statistical Framework**: Robust hypothesis testing with multiple validation methods

## Quick Start

### Prerequisites

- Python 3.9 or higher
- [uv](https://github.com/astral-sh/uv) - Modern Python package manager
- Git for version control

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/intsystems/CreationOfIntelligentSystems_LLM_25.git
   cd CreationOfIntelligentSystems_LLM_25
   ```

2. **Install uv** (if not already installed)
   ```bash
   # On macOS/Linux
   curl -LsSf https://astral.sh/uv/install.sh | sh

   # On Windows
   powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```

3. **Set up the project**
   ```bash
   # Create virtual environment and install dependencies
   uv venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate

   # Install all project dependencies
   uv sync

   # Alternative: Use uv run to execute commands directly
   uv run python your_script.py
   ```

4. **Verify installation**
   ```bash
   uv run python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
   uv run python -c "import transformers; print(f'Transformers version: {transformers.__version__}')"
   ```

## Project Structure

```
CreationOfIntelligentSystems_LLM_25/
├── README.md                   # Project documentation
├── pyproject.toml             # Project configuration and dependencies
├── uv.lock                    # Locked dependency versions
├── linkreview.md              # Literature review and related work
├── src/                       # Main source code directory
│   ├── activation_processor.py  # Process neural network activations
│   ├── calc_stats.py            # Statistical calculations
│   ├── create_topic_configs.py  # Configuration management
│   ├── dataset.py               # Dataset handling
│   ├── extract_act_main.py      # Main activation extraction script
│   ├── hooks.py                 # Model hooks for activation capture
│   ├── model.py                 # Model definitions
│   ├── parse_mrpc.py            # MRPC dataset parsing
│   ├── parse_topics.py          # Topic analysis utilities
│   ├── save_stats_from_topics.py # Save statistical results
│   └── utils.py                 # Utility functions
├── tests/                     # Test suite
├── notebooks/                 # Jupyter notebooks for experiments
│   ├── exploratory_analysis.ipynb
│   ├── manifold_testing.ipynb
│   └── visualization_demo.ipynb
├── data/                      # Datasets and data files
├── paper/                     # LaTeX source for research paper
├── slides/                    # Presentation materials
├── configs/                   # Configuration files
├── all_stats/                 # Statistical analysis results
├── cache/                     # Cached computational results
├── doc/                       # Documentation
└── activation_saved/          # Saved neural network activations
```

## Usage Examples

### Extract Hidden States
```bash

uv run python src/extract_act_main.py --config-name="bert-base_mrpc.yaml"

```

## Calculate Statistics

```bash

uv run python src/calc_stats.py --config-name="calc_stats"

```

### Working with Jupyter Notebooks
```bash
# Start Jupyter Lab with project environment
uv run jupyter lab

# Navigate to notebooks/ and run analyze_instrinsic.ipynb
# This notebook contains intrinsic dimensionality analysis of LLM embeddings
```

## Research Methodology

This project employs a comprehensive approach to testing the manifold hypothesis:

### 1. Embedding Extraction
- Collect embeddings from multiple transformer layers
- Support for various LLM architectures (GPT, BERT, T5, etc.)
- Proper handling of different embedding dimensions

### 2. Dimensionality Analysis
- Principal Component Analysis (PCA)
- t-SNE and UMAP for non-linear dimensionality reduction
- Intrinsic dimensionality estimation

### 3. Manifold Testing
- **Intrinsic Tests**: Measures based on local geometry
- **Extrinsic Tests**: Global embedding space analysis
- **Statistical Validation**: Hypothesis testing with significance levels
- **Cross-validation**: Bootstrap methods for robust estimation

### 4. Visualization
- Interactive 3D embedding visualizations
- Manifold structure representation
- Statistical distribution plots
- Comparative analysis across layers

### 5. Synthetic Validation
- Generated manifolds with known properties
- Controlled experiments with varying complexity
- Baseline comparisons for method validation

## Performance Characteristics

- **Embedding Processing**: Efficient handling of large-scale embeddings
- **Memory Usage**: Optimized for high-dimensional data analysis
- **Computational Complexity**: Scalable to various embedding sizes
- **Support for Multiple Models**: GPT, BERT, T5, and other transformer variants

### Research Extensions
- **Temporal Analysis**: Manifold evolution during training
- **Cross-linguistic Studies**: Manifold properties across languages
- **Task-specific Analysis**: Manifold structure for different downstream tasks
- **Theoretical Framework**: Mathematical foundations for observed phenomena

## Citation

If you use this research in your work, please cite:

```bibtex
@software{manifold_llm_embeddings_2025,
  title={Testing the Manifold Hypothesis in the Embedding Subspaces of Large Language Models},
  authors={Papai, Ivan and Minashkin, Vladislav and Sobolevsky, Fedor and Meshkov, Vladislav},
  year={2024},
  url={https://github.com/your-repo/CreationOfIntelligentSystems_LLM_25}
}
```
