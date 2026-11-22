# Detecting Myocardial Infarction from a Single ECG Lead — with on-device deployment

Final project for **CS 5782 — Introduction to Deep Learning** (Cornell, Spring 2026).

## 1. Introduction

This repository re-implements and extends:

> Wang, Schubert, Kanakamedala, McKelway, Messac, Reginald, Schilbach, Selvavinayagam, Vaidyanathan, Duflo, Obermeyer (2026).
> *AI-Enhanced Handheld ECGs to Screen for Prior Myocardial Infarction in Rural India.*
> NBER Working Paper No. 34690.

The paper trains a 1D ResNet-18 on Lead I ECG signals (pretrained on a Swedish EHR cohort, fine-tuned on a new dataset of 5,301 adults from rural Tamil Nadu, India) to predict prior MI via regional wall motion abnormality on echocardiography, using a $60 AliveCor KardiaMobile single-lead device — a proof-of-concept that low-cost handheld ECGs paired with deep learning can enable MI screening in resource-limited settings.

**Our re-implementation mirrors the paper's cross-population transfer structure with publicly available datasets (Germany → China)** and adds an on-device deployment angle the paper does not address.

## 2. Chosen Result

We reproduce two of the paper's headline findings:

- **Cross-population transfer** of single-lead-ECG MI detection (their Sweden → India; ours Germany → China)
- **Risk-stratification utility:** the model's top-percentile risk scores enrich for MI relative to the population base rate

Original paper headline: AUC 0.77 on the held-out 60% Tamil Nadu cohort, with the top 2.5% of risk scores showing ~6× enrichment over the 1.5% base rate.

## 3. GitHub Contents

```
single-lead-ecg-mi/
├── README.md               ← you are here
├── LICENSE                 ← MIT
├── .gitignore
├── DEVELOPMENT.md          ← phase-by-phase project narrative
├── demo.html               ← interactive single-file demo (section 5.4)
├── project_proposal.md     ← original project proposal
├── code/                   ← Jupyter notebooks (training, tuning, quantization, deployment)
├── data/                   ← see data/README.md for download instructions (waveforms not committed)
├── docs/                   ← project proposal (PDF), poster outline, base-rate notes
├── models/                 ← only the final deployment ONNX is committed
└── poster/                 ← poster.tex/pdf and TikZ architecture diagrams
```

## 4. Re-implementation Details

**Datasets.** Pretrain: PTB-XL (Wagner et al., PhysioNet) — 21,799 12-lead ECG records from Germany. Fine-tune: Zheng et al. (2020) — 45,152 12-lead ECG records from Ningbo First Hospital and Shaoxing People's Hospital, China. We use **Lead I only** in both stages.

**Model.** Final architecture is a small **CNN-Transformer hybrid** (3-layer 1D CNN feature extractor → learned positional embedding → 2-layer Transformer encoder → global average pool → linear head). 131K parameters, 0.63 MB ONNX. Selected from a sweep of seven architectures (ResNet-{18, 34, 50}, ResNet-18+CBAM, Tiny Transformer, Mamba-ECG, CNN-Transformer) trained under identical conditions; chosen for the best AUC-per-MB trade-off.

**Training pipeline.**
1. **Pretrain** on PTB-XL (PhysioNet patient-level folds 1–8 / 9 / 10 → train / val / test): 17,418 / 2,183 / 2,198 records, ~25% MI prevalence. Focal loss (α=0.25, γ=2.0), Adam 1e-3, 5-epoch warmup + cosine annealing, augmentation (Gaussian noise, amplitude scale, random time shift).
2. **Zero-shot evaluate** the pretrained model on the full Chinese set (no Chinese labels seen).
3. **Fine-tune** on Ningbo+Shaoxing (40 / 20 / 40 split, stratified by MI label at record level): 18,060 (49 MI) / 9,030 (25 MI) / 18,062 (49 MI). Focal loss, Adam 1e-4.
4. **Evaluate** on the held-out Chinese test split.

**Extension: on-device deployment.**
- ONNX export (FP32, 0.63 MB) and INT8 post-training static quantization via ONNX Runtime QDQ format (0.19 MB)
- Real-device latency profiling via Qualcomm AI Hub on a flagship (Samsung Galaxy S24) and an LMIC-tier phone (Xiaomi Redmi Note 10 5G, ~$100)
- Result: 152 µs (S24) / 1.96 ms (Redmi) for FP32. INT8 shrinks the model 3.3× with 99.9 % AUC retention but does not improve on-device latency at this size; FP32 is the recommended deployment.

## 5. Reproduction Steps

### 5.1 Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch torchvision numpy pandas scikit-learn duckdb \
    onnx onnxruntime onnxscript qai-hub coremltools jupyter
```

CPU works for inference; pretraining and fine-tuning use GPU (we used ~1× consumer GPU; ~100 epochs PTB-XL pretrain + ~50 epochs fine-tune).

### 5.2 Data

Download instructions in [`data/README.md`](data/README.md). Both datasets are open-access on PhysioNet.

### 5.3 Notebooks

Open `code/` in Jupyter. Recommended order:

1. `mvp_prototype.ipynb` — original MVP (smaller-scale baseline)
2. `training_improvements.ipynb` — full-data training with focal loss, augmentation, cosine LR
3. `architecture_experiments.ipynb` — 7-architecture sweep
4. `catnet_tuning.ipynb` — Optuna hyperparameter search on the CNN-Transformer
5. `catnet_final.ipynb` — final CNN-Transformer training, evaluation, ONNX export
6. `catnet_quantization.ipynb` — INT8 quantization and Qualcomm AI Hub on-device profiling

Final artifacts (committed): `models/cnn_transformer_final.onnx`, `models/cnn_transformer_final_thresholds.json`, `models/cnn_transformer_final_calibration.json`.

### 5.4 Interactive Demo

A self-contained, single-file interactive demo lives at [`demo.html`](demo.html) at the repo root. Open it in any modern browser — no server, no build step.

**What's in it (three tabs):**

- **The Challenge** — draw a Lead I ECG sample, guess the patient's MI risk yourself, then see the model's calibrated risk score on the same trace.
- **Risk Stratification** — run the model on all 349 demo samples and see enrichment ratios at each top-percentile cutoff.
- **Results Explorer** — full architecture-sweep table and on-device latency comparison.

**Demo data.** All 349 ECG traces are drawn from the held-out 40/20/40 test split of the Ningbo+Shaoxing cohort (the model has never seen them during training or validation). The set is deliberately enriched: all 49 MI cases plus 300 randomly sampled non-MI cases, so the apparent MI rate (~14%) does not reflect any real-world prevalence — it's chosen so the demo has enough positives to be informative.

**Regenerate the demo data.** If you retrain or re-export the model, regenerate `demo.html` with:

```bash
cd code
python build_demo.py
```

This re-scores the full 18,062-record test split, samples the enriched demo set, and rewrites `demo.html` in place (with a `.bak` of the previous version). It needs `models/cnn_transformer_final.onnx`, the calibration JSON, and `data/shaoxing_final_splits.csv` (committed) — plus the actual Ningbo+Shaoxing waveforms (downloaded per `data/README.md`).

## 6. Results / Insights

| Metric                                     | Value                          |
|--------------------------------------------|--------------------------------|
| Held-out test AUC (Chinese, fine-tuned)    | **0.81**                       |
| Validation AUC (Chinese)                   | 0.86                           |
| Zero-shot AUC (no Chinese fine-tune)       | 0.81                           |
| Top-2.5% enrichment over dataset base rate | 7.3×                           |
| Final model size (FP32 ONNX)               | 0.63 MB                        |
| On-device latency (Galaxy S24)             | 152 µs                         |
| On-device latency (Redmi Note 10 5G)       | 1.96 ms                        |

Single-lead MI detection transfers across populations (Europe → Asia), mirroring the original paper's Sweden → India result. A 131K-parameter CNN-Transformer matches a 3.8M-parameter ResNet-18's AUC at 1/29th the size, with sub-2-millisecond inference on a $100 phone. INT8 quantization is not worth it at this scale: FP32 is already faster and avoids mobile-toolchain compatibility issues.

## 7. Conclusion

Re-implementing the Obermeyer et al. (2026) cross-population framework with publicly available data confirms its central claim — Lead-I MI detection generalizes across populations — and shows that the model can be small enough and fast enough for the kind of LMIC handheld deployment the original paper motivates but does not implement.

## 8. References

- Wang, Schubert, Kanakamedala, et al. (2026). *AI-Enhanced Handheld ECGs to Screen for Prior Myocardial Infarction in Rural India.* NBER Working Paper No. 34690.
- Wagner, Strodthoff, Bousseljot, et al. *PTB-XL, a large publicly available electrocardiography dataset.* PhysioNet.
- Zheng, J. et al. (2020). *A large-scale 12-lead electrocardiogram database for arrhythmia study.* Ningbo First Hospital + Shaoxing People's Hospital, PhysioNet.
- Qualcomm AI Hub (on-device profiling). https://aihub.qualcomm.com

## 9. Acknowledgements

Completed as coursework for **CS 5782 — Introduction to Deep Learning**, Cornell University, Spring 2026.
