# ”SHERLOCK” (Structured Hierarchical Encoding and Reasoning for Looking,Observing, and Comprehending Knowledge)

This repository contains an implementation of a Visual Question Answering (VQA) system that leverages CLIP embeddings with cross-modal attention mechanisms. The project includes both a baseline model using vanilla CLIP and an enhanced model with multi-level feature processing and attention mechanisms.

## Project Overview

The system is designed to answer natural language questions about images. We implement two approaches:
1. **Baseline Model**: Direct VQA using CLIP embeddings
2. **Enhanced Model**: Multi-branch architecture with cross-modal attention

### Key Features

- Zero-shot question answering capabilities using CLIP
- Multi-level feature processing (global, object, and relation levels)
- Cross-modal attention mechanisms
- Feature caching for improved efficiency
- Comprehensive evaluation framework
- COCO-QA dataset support

## Architecture

### Baseline Model
- Uses CLIP embeddings directly
- Simple comparison between image-question features and answer candidates
- Implemented in `modified_clip_test.py`

### Enhanced Model
The enhanced model consists of several key components:

1. **CLIP Encoder** (`clip_encoder.py`)
   - Handles image and text encoding using CLIP
   - Includes feature caching mechanism
   - Outputs normalized embeddings

2. **Feature Generator** (`feature_generator.py`)
   - WhatBranch: Object detection
   - WhereBranch: Spatial analysis
   - AttributeBranch: Property detection
   - HowBranch: Relationship analysis

3. **Cross-Modal Attention** (`cross_modal_attention.py`, `CMA_object_branch.py`)
   - Multi-head attention mechanisms
   - Feature fusion at multiple levels
   - Object-level and relation-level processing

4. **Main Model** (`model.py`)
   - Integrates all components
   - Handles end-to-end processing
   - Includes training and evaluation logic

## Usage

### Running the Baseline Model

```python
python modified_clip_test.py
```

### Running the Enhanced Model

```python
# Run the complete model
python model.py
```

### Data Preparation

1. Download the COCO-QA dataset
2. Update the data paths

## Model Configuration

Key configuration variables can be found in each component file:

```python
# Configuration variables
CLIP_DIM = 512          # CLIP output dimension
FEATURE_DIM = 128       # Feature dimension from generator
HIDDEN_DIM = 256        # Hidden dimension for attention
DROPOUT_RATE = 0.1
NUM_HEADS = 8
```

