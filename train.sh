MASTER_ADDR="localhost" MASTER_PORT="12343" NODE_RANK=0 WORLD_SIZE=7 \
exec python ./sscd/train.py --nodes=1 --gpus=7 \
                            --train_dataset_path /data/DISC/references \
			    --val_dataset_path /nfs_shared_/vsc2022_data_frame/val \
			    --infonce_temperature 0.05 \
                            --entropy_weight 0 \
                            --base_learning_rate 1e-4 \
                            --optim "adamw" \
			    --batch_size 3584 \
			    --output_path ./results/512_1e-4_ew30_temp005/ \
			    --mae \
			    --epochs 10

