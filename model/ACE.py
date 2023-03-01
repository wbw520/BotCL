import numpy as np
from PIL import Image
import skimage.segmentation as segmentation
from sklearn.model_selection import train_test_split
import cv2
from loaders.ImageNet import get_name
import torch.nn.functional as F
import sklearn.cluster as cluster
import sklearn.metrics.pairwise as metrics
from configs import parser
import os
import shutil
# from draws.draw_synthetic import draw_syn
import torch
from utils.quantitative_eval import make_statistic
import json
from model.retrieval.model_main import MainModel


def load_image_from_file(filename, shape):
    img = np.array(Image.open(filename).resize(shape, Image.BILINEAR))
    # Normalize pixel values to between 0 and 1.
    img = np.float32(img) / 255
    return img


class ConceptDiscovery(object):
    def __init__(self, model, average_image_value=117):
        self.model = model
        self.average_image_value = average_image_value
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
        for i in range(len(test)):
            img = load_image_from_file(test[i], self.image_shape)
            img_list.append([test[i], img])
        return img_list

    def create_patches(self):
        discovery_images = self.load_concept_imgs()
        dataset, image_numbers, patches, masks = [], [], [], []

        for id, (fn, img) in enumerate(discovery_images):
            if id % 10 == 0:
                print("processed " + str(id) + " image for patches")
            image_superpixels, image_patches, image_masks = self.return_superpixels(img)
            for superpixel, patch, mask in zip(image_superpixels, image_patches, image_masks):
                # dataset.append(superpixel)
                patches.append(patch)
                image_numbers.append(fn)
                masks.append(mask)

        return dataset, image_numbers, patches, masks

    def return_superpixels(self, img, param_dict=None):
        """Returns all patches for one image.
        Given an image, calculates superpixels for each of the parameter lists in
        param_dict and returns a set of unique superpixels by
        removing duplicates. If two patches have Jaccard similarity more than 0.5,
        they are concidered duplicates.
        Args:
          img: The input image
          method: superpixel method, one of slic, watershed, quichsift, or
            felzenszwalb
          param_dict: Contains parameters of the superpixel method used in the form
                    of {'param1':[a,b,...], 'param2':[z,y,x,...], ...}. For instance
                    {'n_segments':[15,50,80], 'compactness':[10,10,10]} for slic
                    method.
        """
        if param_dict is None:
            param_dict = {}

        n_segmentss = param_dict.pop('n_segments', [15, 50, 80])
        n_params = len(n_segmentss)
        compactnesses = param_dict.pop('compactness', [20] * n_params)
        sigmas = param_dict.pop('sigma', [1.] * n_params)

        unique_masks = []
        for i in range(n_params):
            param_masks = []
            segments = segmentation.slic(img, n_segments=n_segmentss[i], compactness=compactnesses[i], sigma=sigmas[i])

            for s in range(segments.max()):
                mask = (segments == s).astype(float)
                if np.mean(mask) > 0.001:
                    unique = True
                    for seen_mask in unique_masks:
                        jaccard = np.sum(seen_mask * mask) / np.sum((seen_mask + mask) > 0)
                        if jaccard > 0.5:
                            unique = False
                            break
                    if unique:
                        param_masks.append(mask)
            unique_masks.extend(param_masks)
        superpixels, patches, mask = [], [], []

        # f = 0
        while unique_masks:
            current_mask = unique_masks.pop()
            superpixel, patch = self._extract_patch(img, current_mask)
            superpixels.append(superpixel)
            patches.append(patch)
            mask.append(current_mask)

            # cv2.imwrite(f'img_demo/super_{f}.jpg', np.array(superpixel))
            # cv2.imwrite(f'img_demo/patch_{f}.jpg', np.array(patch))
            # cv2.imwrite(f'img_demo/mask_{f}.jpg', np.array(current_mask) * 255)
            # f += 1

        return superpixels, patches, mask

    def _extract_patch(self, image, mask):
        """Extracts a patch out of an image.
        Args:
          image: The original image
          mask: The binary mask of the patch area
        Returns:
          image_resized: The resized patch such that its boundaries touches the
            image boundaries
          patch: The original patch. Rest of the image is padded with average value
        """
        mask_expanded = np.expand_dims(mask, -1)
        patch = (mask_expanded * image + (1 - mask_expanded) * float(self.average_image_value) / 255)
        ones = np.where(mask == 1)
        h1, h2, w1, w2 = ones[0].min(), ones[0].max(), ones[1].min(), ones[1].max()
        image = Image.fromarray((patch[h1:h2, w1:w2] * 255).astype(np.uint8))
        image_resized = np.array(image.resize(self.image_shape, Image.BICUBIC)).astype(float)
        return image_resized, patch * 255

    @torch.no_grad()
    def get_activation(self, imgs, bs=64):
        output = []
        for i in range(int(imgs.shape[0] / bs) + 1):
            print(i)
            _, features = self.model(torch.from_numpy(np.array(imgs[i * bs:(i + 1) * bs])).permute([0, 3, 1, 2]).to(device).float())
            features = F.adaptive_max_pool2d(features, 1).squeeze(-1).squeeze(-1)
            output.append(features.cpu().detach().numpy())
        output = np.concatenate(output, 0)
        return output

    def cluster(self, acts):
        n_clusters = cluster_number
        centers = None
        km = cluster.KMeans(n_clusters, random_state=10)
        d = km.fit(acts)
        centers = km.cluster_centers_
        d = np.linalg.norm(np.expand_dims(acts, 1) - np.expand_dims(centers, 0), ord=2, axis=-1)
        asg, cost = np.argmin(d, -1), np.min(d, -1)

        if centers is None:  ## If clustering returned cluster centers, use medoids
            centers = np.zeros((asg.max() + 1, acts.shape[1]))
            cost = np.zeros(len(acts))
            for cluster_label in range(asg.max() + 1):
                cluster_idxs = np.where(asg == cluster_label)[0]
                cluster_points = acts[cluster_idxs]
                pw_distances = metrics.euclidean_distances(cluster_points)
                centers[cluster_label] = cluster_points[np.argmin(
                    np.sum(pw_distances, -1))]
                cost[cluster_idxs] = np.linalg.norm(
                    acts[cluster_idxs] - np.expand_dims(centers[cluster_label], 0),
                    ord=2,
                    axis=-1)

        return asg, cost, centers

    def BGR_to_RGB(self, cvimg):
        pilimg = cvimg.copy()
        pilimg[:, :, 0] = cvimg[:, :, 2]
        pilimg[:, :, 2] = cvimg[:, :, 0]
        return pilimg

    def cal_ace(self):
        print("extract patches by superpixel")
        dataset, image_numbers, patches, masks = self.create_patches()
        print("the patch number is:", len(patches))
        print("get activation")
        activations = self.get_activation(np.array(patches))
        print("clustering")
        asg, cost, centers = self.cluster(activations)

        # for s in range(len(asg)):
        #     save = self.BGR_to_RGB(np.array(patches[s]))
        #     cv2.imwrite('img_demo/' + str(asg[s]) + f'/patch_{s}.jpg', save)

        print("calculate ace discovery")
        current_img_name = "start"
        cpt_sample = None
        detect_record = []
        statistic_sample = [0, 0, 0, 0, 0]

        for k in range(len(image_numbers)):
            image_name = image_numbers[k]
            cpt_index = asg[k]
            mask_current = masks[k] * 255
            mask_current[mask_current != 0] = 1
            mask_current[mask_current == 0] = 0

            if current_img_name != image_name:
                current_img_name = image_name
                name_label = current_img_name.split("/")[-1].split(".")[0]
                sample_root = "loaders/matplob/label/" + name_label
                cpt_sample = get_name(sample_root, mode_folder=False)

                if cpt_sample is not None:
                    for h in range(len(cpt_sample)):
                        sample_indexs = int(cpt_sample[h].split(".")[0])
                        statistic_sample[sample_indexs] += 1

            if cpt_sample is None:
                # print("not exist folder " + sample_root)
                continue

            for s in range(len(cpt_sample)):
                sample_index = int(cpt_sample[s].split(".")[0])

                current_sample = cv2.imread(sample_root + "/" + cpt_sample[s], 0)
                current_sample[current_sample != 255] = 1
                current_sample[current_sample == 255] = 0

                overlap = mask_current + current_sample
                overlap_sum = (overlap == 2).sum()
                union_sum = (overlap > 0).sum()

                if overlap_sum / union_sum > thresh_overlap:
                    # if cpt_index not in detect_record:
                    #     detect_record.update({cpt_index: [sample_index]})
                    # elif sample_index not in detect_record[cpt_index]:
                    #     detect_record[cpt_index].append(sample_index)
                    # else:
                    #     continue
                    current_d_name = image_name + "_" + str(cpt_index) + str(sample_index)
                    if current_d_name in detect_record:
                        continue
                    else:
                        detect_record.append(current_d_name)

                    statistic[cpt_index][sample_index] += 1

        # for ll in range(len(statistic)):
        #     print("cpt ", ll)
        #     print(statistic[ll])
        #
        # print(statistic_sample)
        #
        with open("cpt_save_ace.json", "w") as write_file:
            json.dump({"files_ace": statistic}, write_file)
        # print(statistic_sample)
        # draw_syn(statistic, statistic_sample)


if __name__ == '__main__':
    args = parser.parse_args()
    args.pre_train = True
    cluster_number = 15
    thresh_overlap = 0.1
    model_ = MainModel(args)
    args.device = "cuda:1"
    device = torch.device(args.device)
    model_.to(device)
    args.output_dir = "../saved_model"
    checkpoint = torch.load(os.path.join(args.output_dir,
            f"{args.dataset}_{args.base_model}_cls{args.num_classes}_cpt_no_slot.pt"), map_location=device)
    model_.load_state_dict(checkpoint, strict=True)
    model_.eval()

    shutil.rmtree('img_demo/', ignore_errors=True)
    os.makedirs('img_demo/', exist_ok=True)

    for i in range(cluster_number):
        os.makedirs('img_demo/' + str(i) + "/", exist_ok=True)

    statistic = make_statistic(cluster_number)

    ConceptDiscovery(model_).cal_ace()