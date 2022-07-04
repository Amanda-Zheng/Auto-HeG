CUDA_VISIBLE_DEVICES=3 python main_search_wo_srk.py  --data texas \
--learning_rate 0.005 --learning_rate_min 0.005 --weight_decay 3e-4 \
--arch_learning_rate 0.0005 --arch_weight_decay 5e-3 --fix_last \
--epochs_nspa 1000 --space_ver v1 --edge_index mixhop --search_num 5 \
--tau_max 8 --tau_min 4 --device cuda:0 |tee 2>&1 log_texas_search_srk_0515.txt

CUDA_VISIBLE_DEVICES=3 python main_search_wo_srk.py  --data wisconsin \
--learning_rate 0.005 --learning_rate_min 0.005 --weight_decay 3e-4 \
--arch_learning_rate 0.0005 --arch_weight_decay 5e-3 --fix_last \
--epochs_nspa 1000 --space_ver v1 --edge_index mixhop --search_num 5 \
--tau_max 8 --tau_min 4 --device cuda:0 |tee 2>&1 log_wisconsin_search_srk_0515.txt

CUDA_VISIBLE_DEVICES=3 python main_search_wo_srk.py  --data film \
--learning_rate 0.005 --learning_rate_min 0.005 --weight_decay 3e-4 \
--arch_learning_rate 0.0005 --arch_weight_decay 5e-3 --fix_last \
--epochs_nspa 1000 --space_ver v1 --edge_index mixhop --search_num 5 \
--tau_max 8 --tau_min 4 --device cuda:0 |tee 2>&1 log_film_search_srk_0515.txt

CUDA_VISIBLE_DEVICES=3 python main_search_wo_srk.py  --data cornell \
--learning_rate 0.005 --learning_rate_min 0.005 --weight_decay 3e-4 \
--arch_learning_rate 0.0005 --arch_weight_decay 5e-3 --fix_last \
--epochs_nspa 1000 --space_ver v1 --edge_index mixhop --search_num 5 \
--tau_max 8 --tau_min 4 --device cuda:0 |tee 2>&1 log_cornell_search_srk_0515.txt

CUDA_VISIBLE_DEVICES=3 python main_search_wo_srk.py  --data cora \
--learning_rate 0.005 --learning_rate_min 0.005 --weight_decay 3e-4 \
--arch_learning_rate 0.0005 --arch_weight_decay 5e-3 --fix_last \
--epochs_nspa 1000 --space_ver v1 --edge_index mixhop --search_num 5 \
--tau_max 8 --tau_min 4 --device cuda:0 |tee 2>&1 log_cora_search_srk_0515.txt

CUDA_VISIBLE_DEVICES=3 python main_search_wo_srk.py  --data citeseer \
--learning_rate 0.005 --learning_rate_min 0.005 --weight_decay 3e-4 \
--arch_learning_rate 0.0005 --arch_weight_decay 5e-3 --fix_last \
--epochs_nspa 1000 --space_ver v1 --edge_index mixhop --search_num 5 \
--tau_max 8 --tau_min 4 --device cuda:0 |tee 2>&1 log_citeseer_search_srk_0515.txt

CUDA_VISIBLE_DEVICES=3 python main_search_wo_srk.py  --data pubmed \
--learning_rate 0.005 --learning_rate_min 0.005 --weight_decay 3e-4 \
--arch_learning_rate 0.0005 --arch_weight_decay 5e-3 --fix_last \
--epochs_nspa 1000 --space_ver v1 --edge_index mixhop --search_num 5 \
--tau_max 8 --tau_min 4 --device cuda:0 |tee 2>&1 log_pubmed_search_srk_0515.txt

CUDA_VISIBLE_DEVICES=3 python train_search_pt_wo_shrink_time.py  --data cornell \
--learning_rate 0.005 --learning_rate_min 0.005 --weight_decay 3e-4 \
--arch_learning_rate 0.0005 --arch_weight_decay 5e-3 --fix_last \
--epochs_nspa 2 --space_ver v1 --edge_index mixhop --search_num 1 \
--tau_max 8 --tau_min 4 --device cuda:0 |tee 2>&1 log_cornell_search_srk_0526.txt



