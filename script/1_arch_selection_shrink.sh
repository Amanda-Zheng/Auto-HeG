CUDA_VISIBLE_DEVICES=3 python main_search.py  --data texas \
--learning_rate 0.005 --learning_rate_min 0.005 --weight_decay 3e-4 \
--arch_learning_rate 0.0005 --arch_weight_decay 5e-3 \
--epochs_srk 200 --epochs_nspa 1000 \
--ensem_param 0.7 --num_to_keep 15 12 9 --num_to_drop 3 3 3 --eps_no_archs 0 0 0 \
--space_ver v1 --edge_index mixhop --search_num 5 \
--tau_max 8 --tau_min 4 --device cuda:0

CUDA_VISIBLE_DEVICES=3 python main_search.py  --data wisconsin \
--learning_rate 0.005 --learning_rate_min 0.005 --weight_decay 3e-4 \
--arch_learning_rate 0.0005 --arch_weight_decay 5e-3 \
--epochs_srk 200 --epochs_nspa 1000 \
--ensem_param 0.7 --num_to_keep 15 12 9 --num_to_drop 3 3 3 --eps_no_archs 0 0 0 \
--space_ver v1 --edge_index mixhop --search_num 5 \
--tau_max 8 --tau_min 4 --device cuda:0

CUDA_VISIBLE_DEVICES=3 python main_search.py  --data film \
--learning_rate 0.005 --learning_rate_min 0.005 --weight_decay 3e-4 \
--arch_learning_rate 0.0005 --arch_weight_decay 5e-3 \
--epochs_srk 200 --epochs_nspa 1000 \
--ensem_param 0.7 --num_to_keep 15 12 9 --num_to_drop 3 3 3 --eps_no_archs 0 0 0 \
--space_ver v1 --edge_index mixhop --search_num 5 \
--tau_max 8 --tau_min 4 --device cuda:0

CUDA_VISIBLE_DEVICES=3 python main_search.py  --data cornell \
--learning_rate 0.005 --learning_rate_min 0.005 --weight_decay 3e-4 \
--arch_learning_rate 0.0005 --arch_weight_decay 5e-3 \
--epochs_srk 200 --epochs_nspa 1000 \
--ensem_param 0.7 --num_to_keep 15 12 9 --num_to_drop 3 3 3 --eps_no_archs 0 0 0 \
--space_ver v1 --edge_index mixhop --search_num 5 \
--tau_max 8 --tau_min 4 --device cuda:0

CUDA_VISIBLE_DEVICES=3 python main_search.py  --data citeseer \
--learning_rate 0.005 --learning_rate_min 0.005 --weight_decay 3e-4 \
--arch_learning_rate 0.0005 --arch_weight_decay 5e-3 \
--epochs_srk 200 --epochs_nspa 1000 \
--ensem_param 0.7 --num_to_keep 15 12 9 --num_to_drop 3 3 3 --eps_no_archs 0 0 0 \
--space_ver v1 --edge_index mixhop --search_num 5 \
--tau_max 8 --tau_min 4 --device cuda:0

CUDA_VISIBLE_DEVICES=3 python main_search.py  --data pubmed \
--learning_rate 0.005 --learning_rate_min 0.005 --weight_decay 3e-4 \
--arch_learning_rate 0.0005 --arch_weight_decay 5e-3 \
--epochs_srk 200 --epochs_nspa 1000 \
--ensem_param 0.7 --num_to_keep 15 12 9 --num_to_drop 3 3 3 --eps_no_archs 0 0 0 \
--space_ver v1 --edge_index mixhop --search_num 5 \
--tau_max 8 --tau_min 4 --device cuda:0

CUDA_VISIBLE_DEVICES=3 python main_search.py  --data cora \
--learning_rate 0.005 --learning_rate_min 0.005 --weight_decay 3e-4 \
--arch_learning_rate 0.0005 --arch_weight_decay 5e-3 \
--epochs_srk 200 --epochs_nspa 1000 \
--ensem_param 0.7 --num_to_keep 15 12 9 --num_to_drop 3 3 3 --eps_no_archs 0 0 0 \
--space_ver v1 --edge_index mixhop --search_num 5 \
--tau_max 8 --tau_min 4 --device cuda:0



