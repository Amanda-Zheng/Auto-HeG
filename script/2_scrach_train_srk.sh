CUDA_VISIBLE_DEVICES=3 python fine_tune.py --data cornell --edge_index mixhop \
--arch_opt proj_hetro_arch  --epochs 800 --hyper_epoch 100 --device cuda:0 --weight_loss \
--arch_filename ./EXP_search/Arch-mixhop-cornell-v1-20220425-215315-396177/VLOSS_seed_4821.txt \

CUDA_VISIBLE_DEVICES=3 python fine_tune.py --data film --edge_index mixhop \
--arch_opt proj_hetro_arch  --epochs 2000 --hyper_epoch 50 --weight_loss --device cuda:0 \
--arch_filename ./EXP_search/Arch-mixhop-film-v1-20220425-171904-650863/film-searched_20220425-215300_res_best_valid_loss_arch.txt \

CUDA_VISIBLE_DEVICES=1 python fine_tune.py --data texas --edge_index mixhop \
--arch_opt proj_loss_arch  --epochs 1000 --hyper_epoch 20 --device cuda:0 --fix_last --weight_loss \
--arch_filename ./EXP_search/Arch-mixhop-texas-v1-20220507-170030-664845/VLOSS_2452.txt

CUDA_VISIBLE_DEVICES=1 python fine_tune.py --data wisconsin --edge_index mixhop \
--arch_opt proj_hetro_arch  --epochs 1000 --hyper_epoch 100 --device cuda:0 --fix_last \
--arch_filename ./EXP_search/Arch-mixhop-wisconsin-v1-20220430-171718-260888/VLOSS_seed_5317.txt

CUDA_VISIBLE_DEVICES=1 python fine_tune.py --data cora --edge_index mixhop \
--arch_opt proj_hetro_arch  --epochs 1200 --hyper_epoch 200 --fix_last --seed 1234 --device cuda:0 \
--arch_filename ./EXP_search/Arch-mixhop-cora-v1-20220504-090528-245859/VLOSS_8156.txt

CUDA_VISIBLE_DEVICES=0 python fine_tune.py --data citeseer --edge_index mixhop \
--arch_opt proj_hetro_arch  --epochs 1000 --hyper_epoch 100 --device cuda:0 --fix_last \
--arch_filename ./EXP_search/Arch-mixhop-citeseer-v1-20220509-192006-384391/citeseer-searched_20220509-210702_res_best_valid_loss_arch.txt

CUDA_VISIBLE_DEVICES=3 python fine_tune.py --data pubmed --edge_index mixhop \
--arch_opt proj_hetro_arch  --epochs 3500 --hyper_epoch 50 --device cuda:0 --fix_last \
--arch_filename ''