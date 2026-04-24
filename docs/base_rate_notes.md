# Base Rate & Selection Bias Notes

## Two base rates to report

### 1. Population-level incidence: ~0.088% (88 per 100,000)
- **Source:** 2024 Annual Report on Cardiovascular Health and Diseases in China
- **Published in:** JACC (Journal of the American College of Cardiology)
- **Citation:** "The Burden of Cardiovascular Disease in China: Incidence Estimates From National Surveillance (2023)," JACC, 2025. https://www.jacc.org/doi/10.1016/j.jacc.2025.08.060
- **Detail:** Crude incidence of acute myocardial infarction (AMI) among Chinese adults aged ≥18, from national cardiovascular surveillance data.
- **Interpretation:** This is the best available estimate of how common MI is in the general adult population of China — the denominator is everyone, not just people who got an ECG.

### 2. Shaoxing dataset prevalence: ~0.27% (123 MI / 45,152 total records)
- **Source:** Shaoxing People's Hospital ECG database (PhysioNet)
- **Detail:** Proportion of ECG records in the dataset labeled as inferior myocardial infarction.
- **Interpretation:** This reflects hospital ECG traffic — patients who were already referred for an ECG, meaning they were more likely to have a cardiac concern than the general population.

## Selection / enrichment layers

There are three layers of selection that inflate MI prevalence relative to the general population:

1. **Hospital ECG referral (population → Shaoxing dataset):**
   The Shaoxing dataset contains ECGs from patients referred for electrocardiography at a hospital. These patients are already pre-selected — they had symptoms, risk factors, or clinical suspicion that prompted an ECG. MI prevalence in this group (0.27%) is ~3x the population rate (0.088%).

2. **Train/test split (Shaoxing dataset → test set):**
   The test set contains 74 MI out of 27,092 records (0.27%), preserving the dataset-level prevalence. No additional enrichment here.

3. **Demo sampling (test set → 374 demo records):**
   The demo includes all 74 MI cases + 300 randomly sampled non-MI cases, giving an apparent MI rate of ~19.8% (74/374). This is deliberately enriched to ensure the demo has enough MI cases to be informative. The demo clearly labels this as enriched sampling.

## How to present in the demo

- Report the population rate (0.088%) as the epidemiological context for real-world screening.
- Report the Shaoxing dataset rate (0.27%) as the prevalence in the evaluation data (hospital ECG traffic).
- Explain that enrichment ratios can be computed against either baseline depending on the deployment scenario:
  - Against 0.088% → "If screening the general population"
  - Against 0.27% → "If screening patients already referred for ECG"
- Note that the 374 demo samples are enriched (all MI cases included) and do not reflect either base rate.
