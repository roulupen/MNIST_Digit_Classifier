# MNIST Digit Classifier

CNN implementation for MNIST digit classification. Achieves 97.56% test accuracy in one epoch using 24,418 parameters.

## Results

- Test accuracy: 97.56%
- Training accuracy: 81.66%
- Parameters: 24,418
- Test loss: 0.0755
- Training loss: 0.5454

## Features

- Single epoch training
- Cross-platform support (CUDA, MPS, CPU)
- Reproducible results with seed management
- Data augmentation pipeline

## Model Architecture

### SmallMNISTNet4Block

4-block CNN with progressive feature extraction and parameter optimization:

```
Input (1×28×28) 
↓ Block 1: Feature extraction (1→8→16→32→16 channels) + MaxPool + Dropout
↓ Block 2: Pattern recognition (16→32→8 channels) + MaxPool + Dropout  
↓ Block 3: High-level features (8→16 channels) + MaxPool + Dropout
↓ Block 4: Final refinement (16→32 channels)
↓ Global Average Pooling (32×1×1)
↓ Linear Classification (32→10)
↓ Output (10 classes)
```

### Architecture Details

| Layer | Operation | Input Shape | Output Shape | RF | Parameters | Notes |
|-------|-----------|-------------|--------------|----|-----------| -----|
| INPUT | - | [B, 1, 28, 28] | [B, 1, 28, 28] | 1 | 0 | MNIST input |
| **BLOCK 1** | | | | | | |
| conv1_1 | Conv2d(3×3,p=1) | [B, 1, 28, 28] | [B, 8, 28, 28] | 3 | 80 | Initial features |
| conv1_2 | Conv2d(3×3,p=1) | [B, 8, 28, 28] | [B, 16, 28, 28] | 5 | 1,168 | Edge detection |
| conv1_3 | Conv2d(3×3,p=1) | [B, 16, 28, 28] | [B, 32, 28, 28] | 7 | 4,640 | Pattern detection |
| trans1 | Conv2d(1×1) | [B, 32, 28, 28] | [B, 16, 28, 28] | 7 | 528 | Channel reduction |
| maxpool | MaxPool2d(2×2) | [B, 16, 28, 28] | [B, 16, 14, 14] | 8 | 0 | Downsampling |
| dropout1 | Dropout2d(0.1) | [B, 16, 14, 14] | [B, 16, 14, 14] | 8 | 0 | Regularization |
| **BLOCK 2** | | | | | | |
| conv2_1 | Conv2d(3×3,p=1) | [B, 16, 14, 14] | [B, 16, 14, 14] | 12 | 2,320 | Pattern refinement |
| conv2_2 | Conv2d(3×3,p=1) | [B, 16, 14, 14] | [B, 32, 14, 14] | 16 | 4,640 | Feature enhancement |
| trans2 | Conv2d(1×1) | [B, 32, 14, 14] | [B, 8, 14, 14] | 16 | 264 | Channel reduction |
| maxpool | MaxPool2d(2×2) | [B, 8, 14, 14] | [B, 8, 7, 7] | 18 | 0 | Downsampling |
| dropout2 | Dropout2d(0.1) | [B, 8, 7, 7] | [B, 8, 7, 7] | 18 | 0 | Regularization |
| **BLOCK 3** | | | | | | |
| conv3_1 | Conv2d(3×3,p=1) | [B, 8, 7, 7] | [B, 16, 7, 7] | 26 | 1,168 | High-level features |
| conv3_2 | Conv2d(3×3,p=1) | [B, 16, 7, 7] | [B, 16, 7, 7] | 34 | 2,320 | Feature consolidation |
| maxpool | MaxPool2d(2×2) | [B, 16, 7, 7] | [B, 16, 3, 3] | 38 | 0 | Downsampling |
| dropout3 | Dropout2d(0.1) | [B, 16, 3, 3] | [B, 16, 3, 3] | 38 | 0 | Regularization |
| **BLOCK 4** | | | | | | |
| conv4_1 | Conv2d(3×3,p=1) | [B, 16, 3, 3] | [B, 16, 3, 3] | 54 | 2,320 | Final refinement |
| conv4_2 | Conv2d(3×3,p=1) | [B, 16, 3, 3] | [B, 32, 3, 3] | 70 | 4,640 | Output preparation |
| **CLASSIFICATION** | | | | | | |
| GAP | AdaptiveAvgPool2d(1) | [B, 32, 3, 3] | [B, 32, 1, 1] | 70 | 0 | Global pooling |
| flatten | view(-1) | [B, 32, 1, 1] | [B, 32] | 70 | 0 | Flatten |
| fc | Linear(32→10) | [B, 32] | [B, 10] | 70 | 330 | Classification |
| log_softmax | Log probability | [B, 10] | [B, 10] | 70 | 0 | Output |

### Architecture Notes

#### Receptive Field Coverage
- Final RF = 70 pixels (larger than 28×28 MNIST images)
- Provides complete global context for digit classification
- Each output neuron sees the entire digit

#### Channel Management
- Progressive expansion/reduction: 1→8→16→32→16→32→8→16→16→32
- 1×1 convolutions for efficient parameter reduction
- Each block contributes meaningfully to final result

#### Spatial Processing
- Progressive downsampling: 28→14→7→3→1
- Global Average Pooling replaces fully connected layers
- Dropout applied at transition points for regularization

## Training Configuration

### Setup

Training configuration optimized for single-epoch convergence with data augmentation:

#### Data Augmentation
```python
train_transform = transforms.Compose([
    transforms.RandomRotation(5),           # ±5° rotation
    transforms.RandomAffine(                 # Geometric transforms
        degrees=0, translate=(0.05, 0.05), 
        scale=(0.9, 1.1), shear=5
    ),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),  # Illumination
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST normalization
])
```

**Effects:**
- RandomRotation(5°): Handles digit rotation variations
- RandomAffine: Translation, scaling, and shearing transforms
- ColorJitter: Brightness/contrast variations
- MNIST Normalization: Dataset-specific statistics (μ=0.1307, σ=0.3081)

#### Optimizer Configuration
```python
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()
```

**Settings:**
- Adam optimizer with adaptive learning rates
- Learning rate: 0.001
- Weight decay: 1e-4 (L2 regularization)
- CrossEntropyLoss for multi-class classification

#### Reproducibility Setup
```python
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For CUDA
    torch.mps.manual_seed(seed)       # For Apple Silicon
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)
```

**Features:**
- Deterministic results across runs
- Cross-platform consistency (CUDA, MPS, CPU)
- Reproducible experiments

#### Data Loading
```python
train_loader = DataLoader(
    train_dataset, 
    batch_size=32, 
    shuffle=True,
    worker_init_fn=worker_init_fn,
    generator=torch.Generator().manual_seed(SEED)
)
```

**Configuration:**
- Batch size: 32
- Shuffled training data
- Reproducible data loading
- Seeded generator for consistent shuffling

### Training Analysis

#### Feature Learning
Data augmentation enables robust digit recognition:
- Rotation invariance: Handles digit rotation variations
- Scale robustness: Manages different digit sizes and positions
- Illumination adaptation: Works under various lighting conditions

#### Convergence
Optimizer configuration for efficient learning:
- Adam's adaptive learning rates per parameter
- Weight decay prevents overfitting
- CrossEntropyLoss provides stable gradients

#### Regularization
Training setup promotes generalization:
- Data augmentation as implicit regularization
- Dropout layers (0.1 probability) for additional regularization

#### Learning Process
Configuration enables rapid convergence:
- MNIST normalization ensures optimal gradient flow
- Batch size 32 provides stable gradients


### Results Analysis

#### Generalization Performance
- Test accuracy (97.56%) > Training accuracy (81.66%): Inverted gap indicates good regularization
- Single epoch training prevents overfitting
- High test accuracy with minimal computational resources

#### 4-Layer Architecture
- Receptive field coverage: 70×70 RF > 28×28 images
- Progressive feature abstraction:
  - Block 1: Basic edges and corners (RF: 3-8 pixels)
  - Block 2: Pattern recognition (RF: 12-18 pixels)
  - Block 3: High-level features (RF: 26-38 pixels)
  - Block 4: Final representations (RF: 54-70 pixels)
- Parameter allocation: 24,418 parameters distributed across layers
- Global Average Pooling replaces fully connected layers

#### Training Data Transformation
- Augmentation strategy: RandomRotation(5°), RandomAffine, ColorJitter
- Implicit regularization: Augmentation makes training data harder than test data
- Invariance learning: Model ignores irrelevant variations
- Generalization: Features applicable to unseen test data

#### Training Configuration
- Regularization: Weight decay (1e-4) + dropout (0.1) + augmentation
- Optimizer: Adam with learning rate 0.001
- Single epoch training prevents memorization
- Reproducible results with seed management

#### MNIST Dataset Characteristics
- Well-defined problem: Clear digit boundaries and consistent 28×28 format
- Ideal for CNNs: Spatial patterns that CNNs excel at detecting
- Sufficient training data: 60,000 samples provide good coverage
- Problem-model match: Difficulty matches 24,418-parameter model capacity

#### Resource Efficiency
- Model size: 0.09 MB
- Training time: ~30 seconds on modern hardware
- Memory usage: Minimal GPU/CPU requirements
- Test loss: 0.0755
- Stable performance across runs

#### Summary
This implementation demonstrates efficient CNN design for MNIST classification. Key factors:

1. 4-layer architecture with complete receptive field coverage
2. Training data transformation for robust feature recognition
3. Balanced regularization strategy
4. Single epoch training for generalizable learning
5. CNN architecture well-suited for spatial patterns

Result: 97.56% test accuracy in one epoch using 24,418 parameters.
