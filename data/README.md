# Datasets

The two datasets used in this project are open-access on PhysioNet but too large to commit to git (≈ 8.6 GB combined). Re-create the directory layout below before re-running the notebooks.

## Layout

```
data/
├── ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/
│   └── ...      ← unpack PTB-XL release here
└── a-large-scale-12-lead-electrocardiogram-database-for-arrhythmia-study-1.0.0/
    └── ...      ← unpack Zheng et al. 2020 (Ningbo + Shaoxing) here
```

The folder names match the PhysioNet release archives so the existing notebook paths work without modification.

## PTB-XL (pretrain)

- **Source:** Wagner P, Strodthoff N, Bousseljot RD, et al. *PTB-XL, a large publicly available electrocardiography dataset.* PhysioNet, version 1.0.3.
- **URL:** https://physionet.org/content/ptb-xl/1.0.3/
- **Size:** ~1.8 GB
- **What we use:** `ptbxl_database.csv` for labels and `strat_fold` (PhysioNet's built-in 10-fold patient-level split), and the `records100/` waveforms (100 Hz). Lead I only.

## Zheng et al. 2020 — Ningbo + Shaoxing (fine-tune)

- **Source:** Zheng J, Cui H, Struppa D, et al. *A large-scale 12-lead electrocardiogram database for arrhythmia study.* PhysioNet, version 1.0.0.
- **URL:** https://physionet.org/content/ecg-arrhythmia/1.0.0/
- **Size:** ~6.8 GB
- **Origin:** Combined cohort from Shaoxing People's Hospital and Ningbo First Hospital.
- **What we use:** the `WFDBRecords/` waveforms and the diagnosis CSVs to derive a binary MI label. Lead I only.

## Download

You will need a free PhysioNet account to download. The simplest way is the PhysioNet command-line client:

```bash
pip install wfdb requests

# PTB-XL (pretrain)
wget -r -N -c -np https://physionet.org/files/ptb-xl/1.0.3/ -P data/

# Zheng et al. (fine-tune)
wget -r -N -c -np https://physionet.org/files/ecg-arrhythmia/1.0.0/ -P data/
```

After download, move the unpacked folders so the layout matches the section above (the `wget -r` form creates an extra `physionet.org/files/...` prefix you can flatten).

## Licenses

Both datasets are released under the **PhysioNet Credentialed Health Data License** / **ODC-BY**. See each dataset's PhysioNet page for the exact terms.
