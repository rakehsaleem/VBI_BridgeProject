# Domain Adaptation Approach for Bridge Damage Classification

## Problem Setup

**Current State:**
- **Source Domains (3 bridges)**: Fully labeled with all 5 damage conditions
  - Bridge 11m (Simulation01): DC0, DC1, DC2, DC3, DC4 (complete labels)
  - Bridge 13m (Simulation02): DC0, DC1, DC2, DC3, DC4 (complete labels)
  - Bridge 17m (Simulation04): DC0, DC1, DC2, DC3, DC4 (complete labels)
  
- **Target Domain (1 bridge)**: Partially labeled (only 1 damage condition)
  - Bridge 15m (Simulation03): Only DC0 labeled (or one other condition)

**Challenge:**
- Transfer knowledge from 3 fully labeled bridges to 1 partially labeled bridge
- Bridge characteristics differ (length: 11m, 13m, 15m, 17m)
- Domain shift exists between bridges
- Limited labeled data on target bridge

## Proposed Architecture: Bridge-Adaptive CNN (BACNN)

Inspired by **Bridged-GNN** and domain adaptation principles, we propose a multi-stage approach:

### Phase 1: Knowledge Bridge Learning (KBL)

**Concept:**
Instead of domain-level transfer, perform **sample-wise knowledge transfer** by constructing bridges between similar samples across domains.

**Step 1: Build Knowledge Graph (Bridged-Graph)**
```
For each target sample (from Bridge 15m):
   1. Find K nearest neighbors from:
      - Other samples in Bridge 15m (intra-domain)
      - Samples from Bridge 11m, 13m, 17m (inter-domain)
   2. Create edges: target_sample ← source_samples
   3. Result: Directed graph connecting useful knowledge
```

**Step 2: Adaptive Knowledge Retrieval (AKR)**
- Use distance metrics (Euclidean, cosine similarity) in feature space
- Weight connections by similarity
- Filter noisy connections

**Step 3: Graph Knowledge Transfer (GKT)**
- Use graph attention mechanism (like GAT) to aggregate information
- Learn enhanced representation: P_T(Y|X, K(X))
- Where K(X) = knowledge from bridged samples

### Phase 2: Domain Alignment

**Adversarial Domain Adaptation:**
```
Feature Extractor (CNN)
    ↓
    ├─→ Classification Head (for labeled data)
    └─→ Domain Discriminator (to align source & target)
```

**Principles:**
- Learn domain-invariant features
- Bridge length differences handled implicitly
- Multiple source domains → single target

### Phase 3: Semi-Supervised Learning

**Label Propagation:**
- Use limited labels from target bridge
- Propagate labels through knowledge graph
- Self-training on high-confidence predictions

## Implementation Strategy

### Architecture Design

```
┌─────────────────────────────────────────────────────────┐
│           Multi-Bridge Domain Adaptation Model          │
└─────────────────────────────────────────────────────────┘

Input: (samples, 250, 2) from multiple bridges
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│ 1. SHARED FEATURE EXTRACTOR (CNN Encoder)                │
│    Conv1D blocks → Extract bridge-invariant features      │
│    Output: Feature embeddings (samples, feature_dim)     │
└─────────────────────────────────────────────────────────┘
    │
    ├──────────────────────────────────────┐
    ▼                                        ▼
┌─────────────────────┐          ┌──────────────────────┐
│ 2. BRIDGED-GRAPH    │          │ 3. DOMAIN DISCRIMINATOR│
│   Construction       │          │   Classify domain:     │
│   - Find K-NN        │          │   (11m, 13m, 17m, 15m) │
│   - Build edges      │          │   → Align features     │
└─────────────────────┘          └──────────────────────┘
    │                                        │
    ▼                                        ▼
┌─────────────────────────────────────────────────────────┐
│ 4. KNOWLEDGE AGGREGATION (Graph Attention Network)       │
│    - Aggregate features from bridged samples             │
│    - Learn knowledge-enhanced representations            │
│    Output: Enhanced features                             │
└─────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│ 5. CLASSIFICATION HEAD                                   │
│    - Predict damage level (DC0-DC4)                     │
│    - Multi-task: source loss + target loss              │
└─────────────────────────────────────────────────────────┘
```

### Key Components

**1. Shared CNN Encoder**
```python
# Feature extractor trained on all bridges
# Learns general bridge damage features
Input: (batch, 250, 2)
Output: (batch, feature_dim)
```

**2. Bridged-Graph Construction**
```python
def build_bridged_graph(target_features, source_features, k=10):
    """
    For each target sample, find K nearest neighbors in:
    - Target domain (same bridge)
    - Source domains (other bridges)
    """
    # Calculate pairwise distances
    # Return: edge indices, edge weights
```

**3. Graph Knowledge Transfer (GAT-based)**
```python
class GraphKnowledgeTransfer(tf.keras.layers.Layer):
    """
    Aggregates features from connected samples
    Uses attention mechanism to weight contributions
    """
    def __call__(self, features, graph_edges, edge_weights):
        # Multi-head attention aggregation
        # Output: knowledge-enhanced features
```

**4. Domain Discriminator**
```python
# Adversarial component to align domains
# 4-way classifier: 11m, 13m, 15m, 17m
# Trained to confuse domain identity
```

**5. Classification Head**
```python
# 5-way classifier for damage levels
# Separate losses for:
# - Source domains (supervised)
# - Target domain (semi-supervised)
```

### Training Strategy

**Stage 1: Pre-training on Source Domains**
```
- Train CNN encoder on 3 fully labeled bridges
- Learn general damage classification features
- Baseline model
```

**Stage 2: Domain Adaptation**
```
- Build Bridged-Graph connecting source → target
- Train with:
  * Classification loss (source + target labeled data)
  * Domain adversarial loss (align features)
  * Graph aggregation loss (knowledge transfer)
```

**Stage 3: Fine-tuning on Target**
```
- Self-training on high-confidence target predictions
- Label propagation through graph
- Iterative refinement
```

### Loss Functions

**Total Loss:**
```
L_total = α·L_classification + β·L_domain + γ·L_graph + δ·L_target_semi

Where:
- L_classification: Standard cross-entropy (source labeled)
- L_domain: Adversarial domain confusion loss
- L_graph: Graph consistency loss (similar nodes → similar predictions)
- L_target_semi: Semi-supervised loss on target (labeled + pseudo-labeled)
```

### Advantages of This Approach

1. **Sample-wise Transfer**: More flexible than domain-level transfer
2. **No Strong Assumptions**: Doesn't assume same distribution across bridges
3. **Noise Filtering**: Bridged-Graph filters irrelevant source samples
4. **Interpretability**: Can visualize which source samples help which target samples
5. **Multi-source**: Leverages all 3 source bridges simultaneously

## Implementation Plan

### Step 1: Modify Data Loader
- Separate source domains (3 bridges) from target domain (1 bridge)
- Handle partial labels in target domain
- Track bridge ID for each sample

### Step 2: Build Graph Construction Module
- Implement K-nearest neighbor search
- Calculate similarity metrics
- Build Bridged-Graph structure

### Step 3: Implement Graph Attention Layer
- Multi-head attention for knowledge aggregation
- Weight edges by similarity

### Step 4: Domain Discriminator
- 4-way classifier (bridge ID prediction)
- Adversarial training to confuse domains

### Step 5: Combined Model
- Integrate all components
- Multi-task learning with weighted losses

### Step 6: Semi-Supervised Learning
- Pseudo-labeling on target unlabeled data
- Self-training loop

## Expected Benefits

1. **Better target performance**: Leverage knowledge from 3 bridges
2. **Robust to domain shift**: Graph mechanism handles bridge differences
3. **Handles limited labels**: Semi-supervised component uses unlabeled target data
4. **Interpretable**: Can see which source samples influence which predictions

## Questions to Consider

1. Which damage condition is labeled in target bridge? (DC0 only?)
2. How many labeled samples in target bridge? (all DC0 samples or subset?)
3. Should we use all unlabeled target data or only a subset?
4. Prefer simpler implementation first, or full Bridged-GNN from start?

## Next Steps

I can implement:
1. **Simple version**: Domain adversarial + feature alignment (faster to implement)
2. **Full Bridged-GNN**: Complete knowledge bridge learning (more powerful, complex)

Which approach would you prefer to start with?

