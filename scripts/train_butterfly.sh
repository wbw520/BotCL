python main_contrast.py --num_classes 18 --num_cpt 12 \
    --base_model resnet50 --lr 0.0005 --epoch 3000 --lr_drop 500 \
    --dataset butterfly --dataset_dir ../datasets/cuthill_curated/ \
    --weak_supervision_bias 0.1 --quantity_bias 0.1 \
    --distinctiveness_bias 0.05 --consistence_bias 0.01 --batch_size 256