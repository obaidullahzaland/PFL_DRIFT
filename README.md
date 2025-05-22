# PFL-DRIFT
This repository hosts the supplementary code for our paper titled "Concept Drift Aware Hierarchical Aggregation for Personalised Federated Learning". 

We propose PFL-DRIFT, a novel framework that integrates FedALA and FedNN to adaptively tackle key statistical heterogeneity challenges via a two-level hierarchical aggregation scheme, while automatically selecting the most appropriate global aggregation function.

# Requirements

    pip install torch torchvision
    
# Training Script Example
As an example, if we want to run PFL-DRIFT with the PACS dataset across 200 communication rounds, we can use the following command:

    python main.py \
        --dataset PACS \
        --num_classes 7 \
        --n_client 4 \
        --com_amount 200 \
        --seed 50 \
        --model LeNet_fednn \ #Adopting FedNN to address concept drift
        --personalize True \ #Turns on local personalised aggregation
        --mixed True \ #Turns on global aggregate function selector
        --warmup 25 \ #Warm up rounds set to 25 (only effective when mixed is True)
        > results/PACS/PFL-DRFIT_200_50_25.txt
        


# Acknowledgements

(1) ALA module from [FedALA](https://github.com/TsingZ0/FedALA)

(2) Concept drift solution from [FedNN](https://github.com/myeongkyunkang/FedNN/blob/main/main.py)

(3) Personalised global aggregate functions from [FedProx](https://github.com/litian96/FedProx) and [SCAFFOLD](https://github.com/KarhouTam/SCAFFOLD-PyTorch)

(4) Datasets used in the paper
* [CIFAR-10C](https://github.com/hendrycks/robustness)
* [DIGITS-5](https://github.com/med-air/FedBN)
* [PACS from](https://paperswithcode.com/dataset/pacs)
* [OfficeHome](https://github.com/hemanthdv/da-hash)
* [VLCS](https://github.com/belaalb/G2DM#download-vlcs)
* [FairFace](https://github.com/joojs/fairface)

