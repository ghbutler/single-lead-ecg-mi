# Poster Outline

36" x 24", landscape. Roughly 3 columns.

---

## Title Bar (full width)

**Title:** Detecting Inferior Myocardial Infarction from a Single ECG Lead
(or similar, TBD)

**Author:** Gabriel Butler, Cornell University
**Course:** CS 5782, Spring 2026

---

## Column 1: Problem & Motivation

### Background: ECG and Leads
- An electrocardiogram (ECG) records the electrical activity of the heart over time, used to screen for arrhythmias, ischemia, and myocardial infarction (MI, "heart attack")
- A "lead" is a specific viewing angle of the heart's electrical activity, defined by a pair of electrodes on the body. A standard clinical ECG uses 12 leads; each captures the same heartbeat from a different direction
- **Lead I** runs horizontally across the chest (left arm minus right arm). It is the only lead captured by most cheap, handheld, single-lead ECG devices (e.g. Apple Watch, KardiaMobile)
- Inferior MI damages the bottom wall of the heart, whose electrical signal projects primarily onto the *inferior* leads (II, III, aVF) — Lead I is nearly orthogonal to that direction, so inferior MI is traditionally considered invisible to Lead I

### The Problem
- Obermeyer et al. (2026) showed a deep learning model can detect inferior MI from Lead I anyway: an "anatomical discovery" — biological signal present in a lead clinicians consider blind to the pathology
- Implication: MI screening becomes possible in resource-limited settings using ~$60 single-lead devices (the price point discussed in the paper)

### Replication Approach
- Obermeyer et al. trained on data from Sweden, fine-tuned/tested on data from India
- We replicate this cross-population transfer framework: pretrain on PTB-XL (Germany), fine-tune/test on the Zheng et al. (2020) Chinese dataset (Ningbo First Hospital + Shaoxing People's Hospital, referred to as "Ningbo+Shaoxing" hereafter)
- Key idea: train in a developed country, deploy in a developing one
- Proxy datasets chosen to be faithful to the original paper's methodology while using publicly available data

### Visual ideas
- Sample Lead I ECG trace (MI vs normal)
- Maybe a simple map/diagram showing the cross-population transfer (Germany -> China, mirroring Sweden -> India)
- Lead-placement diagram showing why Lead I is geometrically blind to inferior MI

---

## Column 2: Method

### Architecture
- Original paper uses 1D ResNet-18
- We replicate the ResNet-18 baseline, then propose CNN-Transformer (CNN + Transformer hybrid) as our extension
- **Diagram: ResNet-18 architecture** (the replication)
- **Diagram: CNN-Transformer architecture** (the extension)

### Architecture Comparison (table)
- 7 architectures tested under identical conditions
- Table with: Architecture, Params, Size, AUC
- Highlight ResNet-18 (original) and CNN-Transformer (ours) rows
- Others (Mamba-ECG, ResNet-34, ResNet-50, ResNet-18+CBAM, Tiny Transformer) shown for completeness

### Training Pipeline
- **Pretrain** on PTB-XL (21,799 ECGs, 12-lead, Germany) using Lead I only
- **Zero-shot** evaluate the pretrained model on the full Chinese dataset (cross-population transfer, no Chinese labels seen)
- **Fine-tune** on Ningbo+Shaoxing (45,152 ECGs, Lead I only, China)
- **Evaluate** on a held-out Chinese test split
- Focal loss (α=0.25, γ=2.0), Adam, 5-epoch linear warmup + cosine annealing, data augmentation (Gaussian noise, amplitude scaling, random time shift)

### Data Splits

**PTB-XL (pretrain):** PhysioNet's built-in 10-fold `strat_fold` column (patient-level, stratified by label)
- Train: folds 1–8 (17,418 records)
- Val: fold 9 (2,183 records)
- Test: fold 10 (2,198 records)
- MI prevalence: 25%

**Ningbo+Shaoxing (fine-tune):** 40/20/40 stratified by MI label at record level
- Train: 18,060 records (49 MI)
- Val: 9,030 records (25 MI)
- Test: 18,062 records (49 MI)
- MI prevalence: 0.27%

- Key constraint: only 49 MI cases in the fine-tune training set — data is the bottleneck, not architecture

---

## Column 3: Results & Deployment

### Risk Stratification
- Model outputs risk scores, not binary diagnoses (following Obermeyer's framing)
- Enrichment table: top percentile tiers vs. base rate
- Top 2.5% shows 7.3x enrichment over dataset base rate

### What is Quantization?
- Neural networks are normally stored and run in 32-bit floating point (FP32). "Quantization" replaces those weights and activations with lower-precision integers (here, 8-bit integers — INT8)
- Why it matters for deployment: smaller model (~4x shrink), lower memory bandwidth, and on many mobile NPUs, faster inference
- Trade-off: precision loss can degrade accuracy. We use **post-training static quantization** via ONNX Runtime's QDQ (Quantize-DeQuantize) format, calibrating activation ranges on a small sample of Chinese ECGs

### On-Device Deployment
- ONNX export → INT8 static quantization → profile on real devices via Qualcomm AI Hub
- Target devices:
  - Samsung Galaxy S24 (flagship): 152 us
  - Xiaomi Redmi Note 10 5G (~$100 LMIC-tier device): 1.96 ms
- INT8 shrinks model (0.63 MB -> 0.19 MB) at essentially zero AUC cost (99.9% retention)
- BUT: INT8 does *not* improve latency here — the model is already so small that FP32 is fast enough, and INT8 QNN compilation fails on the S24
- **Recommendation: ship FP32** for this model size

### Comparison to Obermeyer et al.
- Table or side-by-side: their AUC vs ours, their enrichment vs ours
- Note differences in data, population, and model

---

## Bottom Strip (full width)

### Takeaways
- Single-lead MI detection transfers across populations (Europe -> Asia, mirroring Sweden -> India in the original paper)
- CNN-Transformer achieves comparable AUC to ResNet-18 at 1/29th the size and faster inference
- Model runs in <2ms on a ~$100 phone, viable for real-world LMIC screening with a ~$60 single-lead ECG device

### References
- Obermeyer et al. (2026), NBER Working Paper 34690
- PTB-XL (Wagner et al., PhysioNet)
- Zheng et al. (2020), Ningbo+Shaoxing 12-lead ECG database (PhysioNet)
- Qualcomm AI Hub

### Acknowledgements
- CS 5782, Cornell University
