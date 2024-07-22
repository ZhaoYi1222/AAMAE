CUDA_VISIBLE_DEVICES=0,1 srun -p rtx3090 python -m torch.distributed.launch --nproc_per_node=2 main_pretrain.py \
	--wandb satmae_project \
	--batch_size 32 --accum_iter 32 --blr 0.0001 \
	--epochs 200 --warmup_epochs 20 --num_workers 16 \
	--input_size 96 --patch_size 8 \
	--mask_ratio 0.75 \
	--model_type group_c \
	--dataset_type sentinel --dropped_bands 0 9 10 \
	--grouped_bands 0 1 2 6 --grouped_bands 3 4 5 7 --grouped_bands 8 9 --start_epoch 50 \
	--resume /nfs-data1/zlx/zlx/big_model/fMoW/fmow_spectral/experiments/pretrain/checkpoint-50.pth \
	--train_path /nfs-data1/zlx/zlx/big_model/fMoW/fmow_spectral/filtered_csv/train.csv \
	--output_dir /nfs-data1/zlx/zlx/big_model/fMoW/fmow_spectral/experiments/pretrain \
	--log_dir /nfs-data1/zlx/zlx/big_model/fMoW/fmow_spectral/experiments/pretrain

