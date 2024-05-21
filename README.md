# defensemae
## Todo List
    - pretraining

## Todo Pretraining Follow The Recipe From The Paper
```bash
torchrun --standalone --nnode 1 --nproc-per-node 4 main_pretrain.py --batch_size 128 --epochs 33 --model_name 'mae_vit_base_vit_base_patch16' --lr 0.0002 --weight_decay 0.0001 --warmup_epochs 3 --save_every_epoch 4 --output_dir './pretrain_vit_base_vit_base_patch16_checkpoints' --log_dir './pretrain_vit_base_vit_base_patch16_checkpoints/logs' --device 'cuda' --num_workers 10 --world_size 4 --dataset "audioset" --gpu 2 3 4 5
```

## Todo Finetuning
```bash
torchrun --standalone --nnode 1 --nproc-per-node 4 main_finetuning.py --batch_size $BATCH_SIZE --epochs $EPOCHS --model_name 'mae_vit_base_vit_base_patch16' --lr 0.0002 --weight_decay 0.0001 --warmup_epochs 3 --save_every_epoch 4 --output_dir './pretrain_vit_base_vit_base_patch16_checkpoints' --log_dir './pretrain_vit_base_vit_base_patch16_checkpoints/logs' --device 'cuda' --num_workers 10 --world_size 4 --dataset "speech_commands" --gpu 2 3 4 5
```