## Usage

#### Data Set
Download CUB or ImageNet and set them into direction of your "dataset_dir". You can also make your own dataset with the structure similar to ImageNet and name it as Custom.

#### Usage for MNIST
Using the following command for training
```
python main_recon.py --num_classes 10 --num_cpt 20 --lr 0.001 --epoch 50 --lr_drop 30
```
Use the following command for the inference of a sample. You can change the index to select different input samples. Change deactivate (deactivate one concept, 1 to num_class) and see the changes of reconstruction. Visualization for the input sample and all concepts are shown at folder "vis" and "vis_pp", respectively. 
```
python vis_recon.py --num_classes 10 --num_cpt 20 --index 0 --top_sample 20 --top_sample 20 --deactivate -1
```

#### Usage for CUB200, ImageNet and Custom
We first pre-train the backbone and then train the whole model. For ImageNet, just change the name for dataset. The generated concept is different upon to the training. Thus, your can train several times to get the satisfying concepts.
```
Pre-traing of backbone:
python main_retri.py --num_classes 50 --num_cpt 20 --lr 0.0005 --epoch 80 --lr_drop 60 --pre_train True --dataset CUB200 --dataset_dir "your dir"

Traing for BotCL:
python main_retri.py --num_classes 50 --num_cpt 20 --lr 0.0005 --epoch 80 --lr_drop 60 --pre_train False --dataset CUB200 --dataset_dir "your dir"
```

#### Demo for MNIST and ImageNet
```
python process.py
python vis_recon.py --num_classes 50 --num_cpt 50 --index 0 --top_sample 20
```