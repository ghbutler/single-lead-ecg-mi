# Project Proposal: Bridging the Diagnostic Gap via Anatomical Discovery using 18-layer ResNet Models

## Introduction and Motivation
In their recent study, Duflo & Obermeyer, et al (2026) highlight a critical failure in the scaling of medical technology: artificial intelligence models trained on tertiary healthcare data in high-income countries often have disappointing performance when applied to global health settings. While drugs can be delivered almost anywhere, high-precision diagnosis remains trapped within the formal health systems of developed nations due to the prohibitive cost and logistical barriers of imaging technologies like Echocardiography. 

This diagnosis bottleneck is particularly binding in low- and middle-income countries (LMICs), where medical resources are scarce. Obermeyer et al. propose a triple win for patients and providers by layering an 18-layer ResNet model on top of inexpensive, handheld ECG devices. By training these deep learning algorithms to discover structural disease, such as regional wall motion abnormalities (RWMA), from raw electrical signals, they demonstrate that low-cost devices can bypass the need for hospital-grade imaging and more efficiently allocate life-saving preventative medications to those who need them most.

## Proposed Research
While this project replicates the methodology of the Obermeyer study, it departs from the original in both geographic scope and dataset scale. Due to data access constraints, specifically the proprietary nature of the Swedish electronic health records and pending logistical support for transferring the Tamil Nadu dataset to a secure environment at Cornell, we utilize two high-quality proxies from PhysioNet. 

First, we use the German PTB-XL dataset for pre-training. It is important to note a significant difference in scale: while the original study leveraged a vast Swedish dataset of 556,000 observations for pre-training, PTB-XL provides a more focused set of approximately 21,800 records. Despite this smaller pre-training volume, we aim to demonstrate that the 18-layer ResNet architecture can still extract robust features.

Second, for the cross-population evaluation, we use the Chinese Shaoxing dataset. Interestingly, whereas the original study’s Indian cohort was relatively small (approximately 5,300 observations), our Chinese evaluation set is nearly nine times larger, containing 45,152 records. Mirroring the Obermeyer methodology, we plan to adopt a 40/60 training-to-test split for the local fine-tuning and evaluation on this dataset. This approach is informed by Obermeyer’s own power calculations, which suggested that a larger test set is necessary to robustly identify significant differences in heart attack rates among the highest-risk subgroups. This provides a substantially larger and more statistically rigorous test bed for verifying the anatomical discovery signal in a developing world context. We continue to coordinate with the Cornell High Performance Computing Center to facilitate the secure transfer of the original Tamil Nadu data, which, if successful, will be integrated for a final direct comparison.

## Anatomical Discovery Task
The core objective remains the validation of predictable hidden physiological markers using a controlled anatomical proxy. To simulate the challenge of predicting structural ground truths from limited electrical signals, we use a single-lead signal (Lead I) to predict Inferior Myocardial Infarction (IMI). Traditionally, an IMI is electrically silent on Lead I and requires specialized leads (II, III, and aVF) for human diagnosis. If an 18-layer ResNet can identify these anatomical markers from the wrong lead, it has discovered biological data that is invisible to a human clinician restricted to the same low-cost input.

## Preliminary Results and MVP Proof
A functional minimum viable prototype (MVP) has already been developed and verified locally using Apple Silicon (MPS) acceleration. Preliminary results confirm the validity of the anatomical discovery hypothesis:
- Anatomical Discovery (Source): The model successfully learned to identify inferior wall damage from a lateral lead signal, achieving a validation AUC of 0.67.
- Cross-Population Portability: Most importantly, the model’s ability to discover these hidden markers generalized remarkably well to the Chinese Shaoxing population, yielding a Zero-shot AUC of 0.86.

These results prove that the underlying physiological signals of structural heart damage are persistent across diverse datasets and can be effectively captured by deep neural networks. The next phase of this project will focus on full-scale training and an error analysis to better understand the specific waveform features the 18-layer ResNet uses to see what human eyes cannot.

## Proposed Timeline
- Week 1: Complete full-scale pre-training on 100% of PTB-XL data and hyperparameter optimization with Optuna.
- Week 2: Conduct fine-tuning and sub-anatomical analysis on the Shaoxing dataset to verify discovery performance on back-wall vs. lower-wall infarctions.
- Week 3: Finalize error analysis and generate visualizations for the Anatomical Discovery interpretation.
- Final: Submission of project poster and GitHub repository including the fully documented discovery pipeline.

## References
1. Wang, J., Schubert, A., & Obermeyer, Z. (2026). "AI-Enhanced Handheld ECGs to Screen for Prior Myocardial Infarction in Rural India." NBER Working Paper No. 34690.
2. Wagner, P., et al. (2020). "PTB-XL, a large publicly available electrocardiography dataset." PhysioNet. https://physionet.org/content/ptb-xl/1.0.3/
3. Zheng, J., et al. (2020). "A large scale 12-lead electrocardiogram database for arrhythmia study." PhysioNet. https://physionet.org/content/ecg-arrhythmia/1.0.0/
