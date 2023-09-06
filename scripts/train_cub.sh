python main_contrast.py --num_classes 50 --num_cpt 20 \
    --base_model resnet18 --lr 0.0003 --epoch 100 --lr_drop 40 \
    --dataset CUB200 --dataset_dir /local/scratch/cv_datasets/CUB_200_2011 \
    --weak_supervision_bias 0.1 --quantity_bias 0.1 \
    --distinctiveness_bias 0.05 --consistence_bias 0.01