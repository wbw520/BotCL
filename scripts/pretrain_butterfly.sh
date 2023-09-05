python main_contrast.py --num_classes 18 --num_cpt 30 \
    --base_model resnet50 --lr 0.0001 --epoch 60 --lr_drop 40 \
    --pre_train True --dataset butterfly \
    --dataset_dir "../datasets/cuthill_curated/" --batch_size 64