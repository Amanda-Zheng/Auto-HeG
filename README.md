# Auto-HeG
This is the official Pytorch implementation of "Auto-HeG: Automated Graph Neural Network on Heterophilic Graphs" in WWW'23

# Introduction

This paper proposes a novel automated graph neural network search framework (Auto-HeG) for heterophilic graphs. 
Specifically, this work incorporates heterophily into three stages:
(1) heterophilic GNN search space design;
(2) progressive supernet training based on layer-wise heterophily variation;
(3) heterophily-aware distance guided architecture selection.

Following is the main framework of the proposed Auto-HeG:
![Auto-HeG_3](https://user-images.githubusercontent.com/61812981/177067417-62743c6f-6f35-43b4-a674-e28127de49bf.png)

# Implementation

Step1: Run the shell in ```script/1_arch_selection_shrink.sh``` for first search, containing the progressive supernet training and heterophily-aware architecture selection, for example:

```
python main_search.py  --data texas \
--learning_rate 0.005 --learning_rate_min 0.005 --weight_decay 3e-4 \
--arch_learning_rate 0.0005 --arch_weight_decay 5e-3 \
--epochs_srk 200 --epochs_nspa 1000 \
--ensem_param 0.7 --num_to_keep 15 12 9 --num_to_drop 3 3 3 --eps_no_archs 0 0 0 \
--space_ver v1 --edge_index mixhop --search_num 5 \
--tau_max 8 --tau_min 4 --device cuda:0
```

Step2: Run the shell in ```script/2_scrach_train_srk.sh``` to finetune the searched architectures, for example:

```
python fine_tune.py --data texas --edge_index mixhop \
--arch_opt proj_hetro_arch  --epochs 1000 --hyper_epoch 20 --device cuda:0 --fix_last --weight_loss \
--arch_filename ./EXP_search/Arch-mixhop-texas-v1-20220507-170030-664845/VLOSS_2452.txt
```
*The code of Auto-HeG is partially refered the work [SANE: Search to aggregate neighborhood for Graph Neural Networks, ICDE,2021.](https://github.com/AutoML-Research/SANE)*

# Discussion & Cite

Please feel free to connect xin.zheng@monash.edu for any questions and issues of this work. 

You are welcome to kindly cite our paper:
```
@inproceedings{autoheg_zheng2023,
  title={Auto-HeG: Automated Graph Neural Network on Heterophilic Graphs},
  author={Zheng, Xin and Zhang, Miao and Chen, Chunyang and Zhang, Qin and Zhou, Chuan, and Pan, Shirui},
  booktitle={Proceedings of the ACM Web Conference (WWW)},
  year={2023}
}
```

