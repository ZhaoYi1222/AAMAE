export OMP_NUM_THREADS=56
export MKL_NUM_THREADS=32
export MPLCONFIGDIR=/share/home/thuzjx/matplotlib_dir

# export MASTER_ADDR=5500-node03:29400
# export MASTER_ADDR=5500-node04:29400

srun -p gpu_th -c 7 --gres=gpu:1 --job-name zy_fp32_gf torchrun --nproc_per_node=1 \
	main_pretrain_e100to130.py \
	--batch_size 64 --accum_iter 2 --blr 1e-5 --weight_decay 0.05 \
	--epochs 130 --num_workers 8   \
	--input_size 224 --patch_size 8 \
	--mask_ratio 0.75 --norm_pix_loss \
	--model_type anchor_aware  \
	--dataset_type lmdb_list \
	--grouped_bands 0 1 2 \
	--isAnchor --isGeoembeded --isScale \
	--lmdb_path /share/home/thuzjx/data/LMDB/ \
	--resume pth_save/checkpoint-99.pth \
	--output_dir pth_save \
	--log_dir log | tee log/e100to130.log 2>&1


