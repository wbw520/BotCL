## Usage

##### Data Set
Download CUB and set them into direction of your "dataset_dir".

##### Training for MNIST
Using the following command
```
python main_recon.py --num_classes 10 --num_cpt 20 --lr 0.001 --epoch 50 --lr_drop 30
```

##### Demo for MNIST
You can change index to select different input samples.
You will get a vis folder for index sample and vis_pp folder for all concepts demonstration.
```
python vis_recon.py --num_classes 10 --num_cpt 20 --index 0 --top_sample 20
```

##### Training for CUB and ImageNet
We first pre-train the backbone and then train the whole model. For ImageNet, just change the name for dataset
```
python main_retri.py --num_classes 50 --num_cpt 20 --lr 0.0005 --epoch 80 --lr_drop 60 --pre_train True --dataset CUB200
python main_retri.py --num_classes 50 --num_cpt 20 --lr 0.0005 --epoch 80 --lr_drop 60 --pre_train False --dataset CUB200
```

##### Demo for MNIST and ImageNet
```
python vis_recon.py --num_classes 50 --num_cpt 50 --index 0 --top_sample 20
```