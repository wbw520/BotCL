python main_contrast.py --num_classes 50 --num_cpt 20 \
    --base_model resnet18 --lr 0.0001 --epoch 60 --lr_drop 40 \
    --pre_train True --dataset CUB200 \
    --dataset_dir /local/scratch/cv_datasets/CUB_200_2011