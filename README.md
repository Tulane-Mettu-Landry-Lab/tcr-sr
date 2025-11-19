# TCR-SREM: Structure-Regularized Explainable TCR-pMHC Binding Prediction
T cells play a central role in adaptive immunity and are fundamental to vaccine and immunotherapy development. Accurate prediction of T cell receptor-peptide-major histocompatibility complex (TCR-pMHC) binding is therefore essential for understanding T cell response mechanisms. Existing deep learning approaches are predominantly sequence-based or structure-based, yet both struggle to generalize to unseen epitopes and have limited explainability. Here, we propose a hybrid explainable model, structure-regularized explainable model (TCR-SREM), that uses a contact prototype to model TCR-pMHC binding and integrates large-scale sequence data with a limited number of crystallographic structures. Our model uses structural regularization during training to incorporate TCR-pMHC contact patterns, while requiring only sequence inputs for prediction. This design effectively transfers structure information to sequence-based models and yields high-quality contact-level explanations and achieves state-of-the-art generalization performance on unseen epitopes.

## Requirements
Please install the packages in `pip install -r requirements.txt`.
To run proteinbert_embed, please download ProteinBERT official repository and install all their requirements.


## Generate Embedding
1. To generate ESM embeddings, please run `python esm_embed.py`.
2. To generate ProteinBERT embedings, please place its folder `proteinbert` in their official reporitory in current folder. Then, please run `python proteinbert_embed.py`.

The embeddings will be generated to folder `./embeddings`.

## Train and Test Models
Please use this command to train and test model:
```bash
bash run.sh
```
or
```bash
python run.py \
    --config "configs/contactareareg/esm2_t6_8M_UR50D.json" \
    --batchsize 512 \
    --device "cuda:0" \
    --epoch 150 \
    --numworkers 8
```

Please change the config file to train and test different models.

The trained models and test results will be stored to `./experiments` folder.

## Datasets

### Train and Test Datasets
- Train dataset: `data/train.csv`
- Test dataset: `data/test.csv`

### TCRXAI Benchmark

- Full benchmark:
    - Seqeunce Table: `data/TCRXAI/tcrxai.csv`
    - Distance Data: `data/TCRXAI/distance.json`

- Regularization and Evaluation Split:
    - Regularization: 
        - Sequence Table: `data/TCRXAI/55/train.csv`
        - Distance Data: `data/TCRXAI/55/train_distance.json`
    - Evaluation: 
        - Sequence Table: `data/TCRXAI/55/test.csv`
        - Distance Data: `data/TCRXAI/55/test_distance.json`


## Visualize Contact Prototype

Please following the jupyter notebook: `contact_prototype_vis.ipynb` to visualize contact prototype.