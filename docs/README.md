# NucleicBERT

NucleicBERT is a BERT-based language model designed for RNA struture and function prediction tasks. This project implements a transformer architecture specifically adapted for nucleotide sequences and provides multiple downstream applications including secondary structure prediction, contact map/distance map prediction, splice site prediction, fitness prediction and shuffled-sequence classification.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Datasets](#datasets)
- [Training](#training)
- [Downstream Tasks](#downstream-tasks)
- [Analysis Tools](#analysis-tools)
- [Configuration](#configuration)

## Overview

NucleicBERT is a transformer-based language model pre-trained on RNA sequences from the [MARS](https://doi.org/10.1093/gpbjnl/qzae018) database. The model learns contextual representations of RNA sequences that can be fine-tuned for various downstream tasks. The architecture is based on BERT but adapted for nucleotide sequences with specialized tokenization and positional encoding.

## Features

- **Pre-training**: Masked language modeling on large-scale RNA datasets
- **Secondary Structure Prediction**: Predict RNA secondary structure from sequence
- **Contact Map/Distance Map Prediction**: Predict tertiary interactions in the form of contacts
- **Splice Site Prediction**: Identify splice acceptor and donor sites
- **Shuffled-Sequence Classification**: Detect if an RNA sequence is authentic or not
- **Fitness Prediction**: Predict RNA fitness landscapes
- **Attention Analysis**: Analyze learned attention patterns
- **Saliency Analysis**: Generate saliency maps for model interpretability

## Installation

### Prerequisites

- Python >= 3.6
- PyTorch
- PyTorch Lightning
- Transformers (Hugging Face)
- NumPy
- Pandas
- Matplotlib
- Seaborn

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Setup

```bash
git clone <repository-url>
cd NucleicBERT
pip install -e .
```

## Project Structure

```
NucleicBERT/
├── nucleicbert/
│   ├── models/           # Model architectures
│   │   ├── bert.py       # BERT model implementation
│   │   ├── resnet.py     # ResNet for Contact Maps
│   │   └── ballquerymodel.py
│   ├── pretrain/         # Pre-training components
│   │   ├── train.py      # Pre-training script
│   │   ├── pretrainingmodule.py
│   │   ├── pretrainingdataset.py
│   │   └── utils.py
│   ├── downstream/       # Downstream task implementations
│   │   ├── secstr*       # Secondary structure prediction
│   │   ├── closeness*    # Contact map/distance map prediction
│   │   ├── splicesite*   # Splice site prediction
│   │   ├── shuffle*      # Shuffled-sequence classification
│   │   └── fitness*      # Fitness prediction
│   ├── inference/        # Inference and analysis tools
│   │   ├── inference.py
│   │   └── saliency/     # Saliency analysis
│   ├── analysis/         # Analysis tools
│   │   ├── attention/    # Attention analysis
│   │   └── properties/   # Attention properties
│   ├── tokenizers/       # Tokenization utilities
│   └── utils/            # Utility functions
├── configs/              # Configuration files
├── docs/                 # Documentation
├── setup.py              # Package setup
└── requirements.txt      # Dependencies
```

## Usage

### Pre-training

Train NucleicBERT on RNA sequences:

```bash
python nucleicbert/pretrain/train.py -y configs/pretrain.yml --save True --logging True
```

### Downstream Tasks

#### Secondary Structure Prediction

```bash
python nucleicbert/downstream/secstrtrain.py -y configs/secstr.yml --save True --logging True
```

#### Contact Map Prediction

```bash
python nucleicbert/downstream/closenesstrain.py -y configs/contactmap.yml --save True --logging True
```

#### Splice Site Prediction

```bash
python nucleicbert/downstream/splicesitetrain.py -y configs/splicesite.yml --save True --logging True
```
#### Shuffled Sequence Detection

```bash
python nucleicbert/downstream/shuffletrain.py -y configs/shuffle.yml --save True --logging True
```

#### Fitness Prediction

```bash
python nucleicbert/downstream/fitnesstrain.py -y configs/fitness.yml --save True --logging True
```

### Saliency Analysis

Generate predictions on new data:

```bash
python nucleicbert/inference/saliency/integrate_analyze_saliency.py --model_path <path_to_model> --data_path <path_to_data>
```

#### Attention Analysis

Analyze attention patterns learned by the model:

```bash
python nucleicbert/analysis/run_attention_analysis.py --model_path <path_to_model>
```

## Model Architecture

NucleicBERT uses a transformer architecture with the following components:

- **Embedding Layer**: Token and positional embeddings for RNA sequences
- **Transformer Encoder**: Multi-layer transformer with multi-head attention
- **Masked Language Model Head**: For pre-training objective
- **Task-specific Heads**: Various heads for downstream tasks

### Default Configuration

- Hidden size: 1024
- Number of layers: 32
- Number of attention heads: 32
- Maximum sequence length: 1024
- Vocabulary size: 25 (RNA nucleotides + special tokens)

## Datasets

### Pre-training Dataset

The model is pre-trained on RNA sequences from the MARS database:
- Total sequences: ~1.7 billion
- Non-Coding RNA sequences used: ~30 million

### Downstream Task Datasets

- **Secondary Structure**: RNAStralign, TR0, ArchiveII
- **Contact Maps/Distance Maps**: BGSU, NucleoSeeker
- **Splice Sites**: Spliceator benchmark datasets
- **Fitness**: CPEB3 Mutation Dataset

## Training

### Pre-training

The model is pre-trained using masked language modeling:

1. **Tokenization**: RNA sequences are tokenized using nucleotide-level tokens
2. **Masking**: 15% of tokens are masked for prediction
3. **Training**: Model learns to predict masked tokens from context

### Fine-tuning

For downstream tasks:

1. Initialize with pre-trained weights
2. Add task-specific head
3. Fine-tune on task-specific data
4. Use appropriate loss functions (CrossEntropy, Focal Loss, etc.)

## Downstream Tasks

### Secondary Structure Prediction

Predicts RNA secondary structure elements:
- Stems
- Loops
- Bulges
- Hairpins

### Contact Map Prediction

Predicts nucleotide-nucleotide contacts:
- Short-range contacts (8 to 16 nucleotides apart)
- Medium-range contacts (16 to 24 nucleotides apart)
- Long-range contacts (>24 nucleotides apart)

### Splice Site Prediction

Identifies splice sites in RNA sequences:
- Acceptor sites
- Donor sites

### Sequence Classification

Various classification tasks:
- Natural vs. shuffled sequences
- RNA type classification
- Functional annotation

## Analysis Tools

### Attention Analysis

- Visualize attention patterns across layers and heads
- Analyze attention to different RNA motifs
- Compare attention patterns between tasks

### Saliency Analysis

- Generate input attribution maps
- Identify important nucleotides for predictions
- Visualize model decision-making process

## Configuration

Configuration files are stored in the `configs/` directory:

- `pretrain.yml`: Pre-training configuration
- `secstr.yml`: Secondary structure prediction
- `contactmap.yml`: Contact map prediction
- `splicesite.yml`: Splice site prediction
- `shuffle.yml`: Shuffle detection
- `fitness.yml`: Fitness prediction

The weights of the pretrained model and downstream tasks can be downloaded from [Zenodo](https://doi.org/10.5281/zenodo.16989562).