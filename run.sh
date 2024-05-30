cd code

## train uneditied models
python training/two_stage_training/exp_runner_unedited_models.py --conf confs_sg/default_two_stage_training.conf --data_split_dir ../example_data/shiny/ball/train --expname ball --nepoch 2000 --max_niter 200001 --gamma 1.0 --geometry .pth

python training/two_stage_training/exp_runner_unedited_models.py --conf confs_sg/default_two_stage_training.conf --data_split_dir ../example_data/kitty/train --expname kitty --nepoch 2000 --max_niter 200001 --gamma 1.0

## color editing & texture editing
python evaluation/eval.py --conf confs_sg/default.conf --data_split_dir ../example_data/kitty/train --expname physg_synthetic/kitty --gamma 2.2  --exps_folder exps/0_unedited_models

python diffuse_finetune/envmap_finetune.py --conf confs_sg/dual_mlp_cdist.conf --data_split_dir ../example_data/kitty/edit1/data --expname physg_synthetic/kitty --exps_folder exps --gamma 1.0 --resolution 256 --edited_image ../example_data/kitty/edit1/edited_diffuse.png --n_epochs 2000 --mask_image ../example_data/kitty/edit1/scribble_mask.png --task 2_texture_editing --flag 0_unrelight_finetune --lr 1e-3

python diffuse_finetune/eval_dual_mlp_cdist.py --conf confs_sg/dual_mlp_cdist.conf --data_split_dir ../example_data/kitty/train --expname physg_synthetic/kitty --exps_folder exps --gamma 2.2 --resolution 256 --model_params_dir ../iccv23/exps/2_texture_editing/kitty/0_unrelight_finetune --threshold 1e-1 --task 2_texture_editing --flag 2_unrelight_thres_1e-1

