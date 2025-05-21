# PFL-DRIFT
This repository hosts the supplementary code for our paper titled "Concept Drift Aware Hierarchical Aggregation for Personalised Federated Learning".

# Requirements

    pip install torch torchvision
    
# Train

    # PFL-DRIFT
    python main.py \
        --method fedavg \
        --data_dir ./data \
        --dataset_name DIGIT \
        --model_name LeNet_fednn \
        --result_path results

    # native FedAvg
    python main.py \
        --method fedavg \
        --data_dir ./data \
        --dataset_name DIGIT \
        --model_name LeNet \
        --result_path results



# Acknowledgements

(1) ALA module from [FedALA](https://github.com/TsingZ0/FedALA)

(2) Concept drift solution from [FedNN](https://github.com/myeongkyunkang/FedNN/blob/main/main.py)

(3) Global aggregate functions from [FedProx](https://github.com/litian96/FedProx) and [SCAFFOLD](https://github.com/KarhouTam/SCAFFOLD-PyTorch)

(4) Pre-processed digit datasets from [FedBN](https://github.com/med-air/FedBN)
