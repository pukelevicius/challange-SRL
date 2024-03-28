# challange-SRL
This is the repository for Advanced NLP course final take-home exam.

Data folder contains: 
1) Universal Propositions Bank 1.0 English data file which were used to train DistilBERT models for SRL task.
2) two challange dataset - challange_mft.csv (contains tests for MFT test type) and challange_inv.csv (contains tests for INV test type)

Code folder contains several files:
1) data_utils.py and feature_utils.py in which data processing and feature engineering functions are stored.
2) Aurgument_Classification_Transformers.ipynb notebook for training to DistilBERT model variants for SRL task.
3) evaluate_challange_dataset.ipynb notebook for evaluating trained models with challange datasets. To run this notebook, one must download both trained models and store to models/ folder
the link for downloading models - https://drive.google.com/drive/folders/1wsJXgd4Z4RcvuXZKaKuoDz5v2WXAlY-A?usp=sharing

training results folder contains results from training DistilBERT model, this include classification metrics (Precision, recall, F1) for each semantic role label

evaluation results folder stores the challenge datasets with appended predictions (baseline and advanced model), and failure rate metrics for each tested capability for used model.



