CUDA_VISIBLE_DEVICES=3 python fine_tune.py --data texas --edge_index mixhop \
--arch_opt proj_hetro_arch  --epochs 1000 --hyper_epoch 20 --device cuda:0 --fix_last --weight_loss \
--arch_filename ./EXP_search/Arch-mixhop-texas-v1-20220515-173113-203056/texas-searched_20220515-180942_res_best_valid_loss_arch.txt \
|tee 2>&1 Logs/ft/log_texas_wo_srk_0516.txt

CUDA_VISIBLE_DEVICES=1 python fine_tune.py --data wisconsin --edge_index mixhop \
--arch_opt proj_hetro_arch  --epochs 1000 --hyper_epoch 20 --device cuda:0 --fix_last \
--arch_filename ./EXP_search/Arch-mixhop-wisconsin-v1-20220515-180948-524869/wisconsin-searched_20220515-191548_res_best_valid_loss_arch.txt \
|tee 2>&1 Logs/ft/log_wisconsin_wo_srk_0516.txt

CUDA_VISIBLE_DEVICES=1 python fine_tune.py --data film --edge_index mixhop \
--arch_opt proj_hetro_arch  --epochs 2000 --hyper_epoch 20 --weight_loss --device cuda:0 --fix_last \
--arch_filename ./EXP_search/Arch-mixhop-film-v1-20220515-191555-817436/film-searched_20220515-214723_res_best_valid_loss_arch.txt \
|tee 2>&1 Logs/ft/log_film_wo_srk_0516.txt

CUDA_VISIBLE_DEVICES=3 python fine_tune.py --data cornell --edge_index mixhop \
--arch_opt proj_hetro_arch  --epochs 800 --hyper_epoch 20 --device cuda:0 --weight_loss --fix_last\
--arch_filename ./EXP_search/Arch-mixhop-cornell-v1-20220515-214728-857123/cornell-searched_20220515-222058_res_best_valid_loss_arch.txt \
|tee 2>&1 Logs/ft/log_cornell_wo_srk_0516.txt

CUDA_VISIBLE_DEVICES=1 python fine_tune.py --data cora --edge_index mixhop \
--arch_opt proj_hetro_arch  --epochs 800 --hyper_epoch 20 --fix_last --device cuda:0 \
--arch_filename ./EXP_search/Arch-mixhop-cora-v1-20220515-222103-462131/cora-searched_20220515-233436_res_best_valid_loss_arch.txt \
|tee 2>&1 Logs/ft/log_cora_wo_srk_0516.txt

CUDA_VISIBLE_DEVICES=0 python fine_tune.py --data citeseer --edge_index mixhop \
--arch_opt proj_hetro_arch  --epochs 1000 --hyper_epoch 20 --device cuda:0 --fix_last \
--arch_filename ./EXP_search/Arch-mixhop-citeseer-v1-20220515-233449-880375/citeseer-searched_20220516-004412_res_best_valid_loss_arch.txt \
|tee 2>&1 Logs/ft/log_citeseer_wo_srk_0516.txt

CUDA_VISIBLE_DEVICES=3 python fine_tune.py --data pubmed --edge_index mixhop \
--arch_opt proj_hetro_arch --epochs 2000 --hyper_epoch 20 --device cuda:0 --fix_last \
--arch_filename ./EXP_search/Arch-mixhop-pubmed-v1-20220516-004420-630896/pubmed-searched_20220516-034302_res_best_valid_loss_arch.txt \
|tee 2>&1 Logs/ft/log_pubmed_wo_srk_0516.txt
