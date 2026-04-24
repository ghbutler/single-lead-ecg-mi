# Development Process

## Phase 1: Replication (MVP)
Built a 1D ResNet-18 to replicate Obermeyer et al. (2026) — detecting myocardial infarction from single-lead ECG (Lead I), which is traditionally blind to inferior MI. The "anatomical discovery" task: if the model can detect MI from the wrong lead, it has found biological signal invisible to human clinicians.

- Pre-trained on PTB-XL (Germany, 21,799 records, 25% MI prevalence)
- Evaluated cross-population on Shaoxing (China, 45,152 records, 0.27% MI prevalence)
- Initial MVP used 50-batch training limit and 5K Shaoxing subset
- Results were unstable (0.63-0.97 AUC) due to only ~12 MI cases in the subset

**Notebooks:** `mvp_prototype.ipynb`, `mvp_prototype_executed.ipynb`, `mvp_prototype_shaoxing_executed.ipynb`

## Phase 2: On-Device Deployment (ResNet-18)
Extended the replication with a mobile deployment feasibility analysis — an angle the original paper does not address. Motivated by the paper's premise: cheap handheld ECGs in rural LMICs need on-device inference.

- ONNX export with INT8 quantization (ONNX Runtime, QDQ format)
- Qualcomm AI Hub: compile and profile on real devices
  - Samsung Galaxy S24 (Snapdragon 8 Gen 3, flagship)
  - Xiaomi Redmi Note 10 5G (Snapdragon 678, budget LMIC device)
- Encountered and resolved: external ONNX weights, dynamic shapes, TFLite compilation failures on quantized 1D conv, transformer SequenceEmpty ops

**Notebook:** `quantization_analysis.ipynb`

## Phase 3: Training Improvements
Applied training-only improvements without changing the ResNet-18 architecture. None of these affect model size or inference latency.

- Removed 50-batch limit, used full PTB-XL and Shaoxing datasets
- Class weighting (pos_weight) to address MI imbalance
- Focal loss for better handling of rare positives
- Data augmentation: Gaussian noise, amplitude scaling, random time shift
- Cosine annealing LR scheduler
- Stratified 40/60 train/test split (matching Obermeyer methodology)
- **Result: ResNet-18 AUC improved from ~0.63 to 0.85**
- Most gain came from using full data and addressing class imbalance

**Notebook:** `training_improvements.ipynb`

## Phase 4: Architecture Experiments
Compared 7 architectures under identical training conditions (focal loss, CutMix, augmentation, cosine LR, full data).

| Architecture | Params | Size | Fine-tuned AUC |
|---|---|---|---|
| Mamba-ECG | 104K | 0.42 MB | 0.863 |
| ResNet-18 | 3.8M | 14.7 MB | 0.852 |
| CNN-Transformer | 131K | 0.52 MB | 0.851 |
| ResNet-34 | 7.2M | 27.7 MB | 0.848 |
| ResNet-18+CBAM | 3.9M | 15.1 MB | 0.844 |
| ResNet-50 | 16M | 61.2 MB | 0.805 |
| Tiny Transformer | 4.4K | 0.02 MB | 0.729 |

**Key findings:**
- Larger models (34, 50, CBAM) did not help — too many parameters for 49 MI fine-tuning cases
- Tiny Transformer underfitted — too few parameters for the task
- Top 3 (Mamba, ResNet-18, CNN-Transformer) within ~1 AUC point of each other
- On-device profiling: CNN-Transformer fastest on both devices (161us S24, 1.9ms Redmi)

**Notebook:** `architecture_experiments.ipynb`

## Phase 5: CNN-Transformer Hyperparameter Tuning
Selected CNN-Transformer as the deployment candidate based on best latency/size/AUC trade-off. Used Optuna to search over learning rate, warmup epochs, and CNN freezing.

- 30 total trials across MedianPruner and HyperbandPruner
- Search space: lr (1e-5 to 1e-2), warmup (0-5 epochs), freeze_cnn (True/False)
- Up to 100 pretrain + 50 finetune epochs per trial with early stopping
- **Best result: 0.853 AUC** (lr=0.000356, warmup=5, freeze=False)
- Tuning did not meaningfully improve over default hyperparameters (~0.85 ceiling)
- Conclusion: data bottleneck (49 MI cases), not hyperparameters, is the limiting factor

**Notebook:** `catnet_tuning.ipynb`

## Phase 6: CNN-Transformer Quantization & Final Profiling
Quantized the final CNN-Transformer model and profiled on target devices.

- INT8 quantization preserves AUC (99.9% retention)
- Model size: 0.63 MB FP32 → 0.19 MB INT8
- **FP32 is faster than INT8 on-device** — model too small for quantization to help latency
- INT8 QNN compilation fails on S24 (exit code 14); FP32 works fine
- **Deployment recommendation: ship FP32**

| Model | AUC | Size | S24 | Redmi |
|---|---|---|---|---|
| CNN-Transformer FP32 | 0.855 | 0.63 MB | 152 us | 1,960 us |
| CNN-Transformer INT8 | 0.855 | 0.19 MB | FAILED | 4,158 us |

**Notebook:** `catnet_quantization.ipynb`

## Final Model
- **Architecture:** CNN-Transformer hybrid (131K params)
- **Checkpoint:** `checkpoints/cnn_transformer_final_finetuned_best.pt`
- **ONNX for deployment:** `models/cnn_transformer_final.onnx` (0.63 MB)
- **AUC:** 0.855 on Shaoxing (cross-population)
- **Latency:** 1.96ms on Xiaomi Redmi Note 10 5G (~$100 phone)
- **Format:** FP32 (quantization not beneficial at this model size)

## Infrastructure
- **Experiment tracking:** DuckDB (`experiment_results.duckdb`)
- **On-device profiling:** Qualcomm AI Hub (cached in `cache/`)
- **Checkpointing:** PyTorch state dicts in `checkpoints/`

## Pending
- Docker containerization for reproducibility
- Project poster
