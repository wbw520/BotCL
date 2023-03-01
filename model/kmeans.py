import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from configs import parser
import matplotlib.pyplot as plt
import os
import sklearn.cluster as cluster
import torch
import json
from sklearn.manifold import TSNE
from utils.quantitative_eval import make_statistic
from loaders.ImageNet import get_name
from sklearn.decomposition import PCA
from model.retrieval.model_main import MainModel
import cv2


def load_image_from_file(filename, shape):
    img = np.array(Image.open(filename).resize(shape, Image.BILINEAR))
    # Normalize pixel values to between 0 and 1.
    img = np.float32(img) / 255

    return img


class ConceptDiscovery(object):
    def __init__(self, model):
        self.model = model
        self.image_shape = (224, 224)
        self.data_root = "loaders/"

    def load_concept_imgs(self):
        record = []
        load_npy_y = np.load(self.data_root + "y_data.npy")

        for i in range(load_npy_y.shape[0]):
            img_root = self.data_root + "matplob/raw/" + str(i) + ".jpg"
            record.append(img_root)

        train, test = train_test_split(record, train_size=0.9, random_state=1)
        print("get test image, number:", len(test))
        img_list = []
        name_list = []
        for i in range(len(test)):
            img = load_image_from_file(test[i], self.image_shape)
            img_list.append(img)
            name_list.append(test[i])
        return img_list, name_list

    @torch.no_grad()
    def get_activation(self, imgs, bs=64):
        output = []
        for i in range(int(imgs.shape[0] / bs) + 1):
            print(i)
            _, features = self.model(
                torch.from_numpy(np.array(imgs[i * bs:(i + 1) * bs])).permute([0, 3, 1, 2]).to(device).float())
            features = features.permute([0, 2, 3, 1]).reshape(-1, 49, 512)
            output.append(features.cpu().detach().numpy())
        output = np.concatenate(output, 0)
        return output

    def cluster(self, acts):
        n_clusters = cluster_number
        centers = None
        km = cluster.KMeans(n_clusters, random_state=2)
        d = km.fit(acts)
        centers = km.cluster_centers_
        d = np.linalg.norm(np.expand_dims(acts, 1) - np.expand_dims(centers, 0), ord=2, axis=-1)
        asg, cost = np.argmin(d, -1), np.min(d, -1)

        return asg, cost, centers

    def cluster2(self, acts):
        n_clusters = cluster_number
        axis = None
        pca = PCA(n_components=n_clusters)
        x_ = pca.fit(acts)
        axis = pca.components_

        acts = torch.from_numpy(acts)
        l, c = acts.shape
        mean = torch.mean(acts, dim=0).expand(750, 49, c)
        var = torch.std(acts, dim=0).expand(750, 49, c)
        acts = acts.reshape(750, 49, -1)
        f = (acts - mean) / var
        f = f.reshape(l, c)
        f = f.numpy()

        d = abs((np.expand_dims(f, 1) * np.expand_dims(axis, 0)).sum(-1))
        asg, cost = np.argmax(d, -1), np.max(d, -1)

        return asg, cost, axis

    def cluster3(self, acts):
        n_clusters = cluster_number
        axis = None
        ts = TSNE(n_components=2, init="pca", random_state=0)
        results = ts.fit_transform(acts)
        return results

    def inference(self):
        imgs, names = self.load_concept_imgs()
        out_f = self.get_activation(np.array(imgs))
        out_f = out_f.reshape(-1, 512)
        asg, cost, centers = self.cluster(out_f)
        asg = np.array(asg)
        asg = np.array(np.split(asg, len(names), axis=0)).reshape(-1, 7, 7)

        for i in range(len(names)):
            current_img_name = names[i]
            name_label = current_img_name.split("/")[-1].split(".")[0]
            sample_root = "loaders/matplob/label/" + name_label
            cpt_sample = get_name(sample_root, mode_folder=False)
            cluster_map = cv2.resize(asg[i], (224, 224), interpolation=cv2.INTER_NEAREST)

            if cpt_sample is None:
                # print("not exist folder " + sample_root)
                continue

            for j in range(cluster_number):
                current_map = cluster_map == j
                current_map = current_map.astype(int)

                for s in range(len(cpt_sample)):
                    sample_index = int(cpt_sample[s].split(".")[0])
                    current_sample = cv2.imread(sample_root + "/" + cpt_sample[s], 0)
                    current_sample[current_sample != 255] = 1
                    current_sample[current_sample == 255] = 0

                    overlap = current_map + current_sample
                    overlap_sum = (overlap == 2).sum()
                    union_sum = current_sample.sum()

                    if overlap_sum / union_sum > thresh_overlap:
                        statistic[j][sample_index] += 1

        with open("cpt_save_km.json", "w") as write_file:
            json.dump({"files_km": statistic}, write_file)

    def inference2(self):
        imgs, names = self.load_concept_imgs()
        out_f = self.get_activation(np.array(imgs))
        n, l, d = out_f.shape

        record = np.zeros(n*l)

        for i in range(len(names)):
            current_img_name = names[i]
            name_label = current_img_name.split("/")[-1].split(".")[0]
            sample_root = "loaders/matplob/label/" + name_label
            cpt_sample = get_name(sample_root, mode_folder=False)

            if cpt_sample is None:
                # print("not exist folder " + sample_root)
                continue

            for j in range(l):
                tt = np.zeros(l)
                tt[j] += 1
                tt = tt.reshape((7, 7))
                tt = cv2.resize(tt, (224, 224), interpolation=cv2.INTER_NEAREST)
                for s in range(len(cpt_sample)):
                    sample_index = int(cpt_sample[s].split(".")[0])
                    current_sample = cv2.imread(sample_root + "/" + cpt_sample[s], 0)
                    current_sample[current_sample != 255] = 1
                    current_sample[current_sample == 255] = 0

                    overlap = tt + current_sample
                    overlap_sum = (overlap == 2).sum()
                    union_sum = current_sample.sum()

                    if overlap_sum / union_sum > 0.1:
                        record[i * l + j] = sample_index + 1

        tsn_results = self.cluster3(out_f.reshape(-1, 512))
        print(tsn_results.shape)
        print(record.shape)
        draw(tsn_results, record)


def draw(data, labels):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)
    fig = plt.figure()
    ax = plt.subplot(111)

    color = ["red", "green", "black", "purple", "orange"]
    ll = ["s1", "s2", "s3", "s4", "s5"]

    for i in range(len(data)):
        label = int(labels[i])
        if label == 0:
            continue
        ax.scatter(data[i][0], data[i][1], c=color[label-1], alpha=0.5, label=ll[label-1])
    plt.xticks()
    plt.yticks()
    plt.savefig("t-SNE.png")
    plt.show()


if __name__ == '__main__':
    args = parser.parse_args()
    args.pre_train = True
    cluster_number = 15
    thresh_overlap = 0.9
    args.num_classes = 15
    model_ = MainModel(args)
    args.dataset = "matplot"
    args.device = "cuda:2"
    device = torch.device(args.device)
    model_.to(device)
    args.output_dir = "../saved_model"
    checkpoint = torch.load(os.path.join(args.output_dir,
                                                 f"{args.dataset}_{args.base_model}_cls{args.num_classes}_cpt_no_slot.pt"),
                                    map_location=device)
    model_.load_state_dict(checkpoint, strict=True)
    model_.eval()
    statistic = make_statistic(cluster_number)
    ConceptDiscovery(model_).inference()

