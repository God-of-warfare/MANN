import numpy as np
import os
import random
import torch
from torch.utils.data import IterableDataset
import time
import imageio


def get_images(paths, labels, num_samples=None):

    if num_samples is not None:
        sampler = lambda x: random.sample(x, num_samples)
    else:
        sampler = lambda x: x
    labels_and_images = [
        (i, os.path.join(path, image))
        for i, path in zip(labels, paths)
        for image in sampler(os.listdir(path))
    ]

    return labels_and_images


class DataGenerator(IterableDataset):
    def __init__(
            self,
            num_classes,
            num_samples_per_class,
            batch_type,
            config={},
            cache=True,
    ):

        self.num_samples_per_class = num_samples_per_class
        self.num_classes = num_classes

        self.data_folder = config.get("data_folder", "./omniglot_resized")
        self.img_size = config.get("img_size", (28, 28))

        self.dim_input = np.prod(self.img_size)
        self.dim_output = self.num_classes

        character_folders = [
            os.path.join(self.data_folder, family, character)
            for family in os.listdir(self.data_folder)
            if os.path.isdir(os.path.join(self.data_folder, family))
            for character in os.listdir(os.path.join(self.data_folder, family))
            if os.path.isdir(os.path.join(self.data_folder, family, character))
        ]

        random.seed(1)
        random.shuffle(character_folders)
        num_val = 100
        num_train = 1100
        self.metatrain_character_folders = character_folders[:num_train]
        self.metaval_character_folders = character_folders[num_train: num_train + num_val]
        self.metatest_character_folders = character_folders[num_train + num_val:]
        self.image_caching = cache
        self.stored_images = {}

        if batch_type == "train":
            self.folders = self.metatrain_character_folders
        elif batch_type == "val":
            self.folders = self.metaval_character_folders
        else:
            self.folders = self.metatest_character_folders

    def image_file_to_array(self, filename, dim_input):

        if self.image_caching and (filename in self.stored_images):
            return self.stored_images[filename]
        image = imageio.imread(filename)  # misc.imread(filename)
        image = image.reshape([dim_input])
        image = image.astype(np.float32) / image.max()
        image = 1.0 - image
        if self.image_caching:
            self.stored_images[filename] = image
        return image

    def _sample(self):

        # Sample N (self.num_classes in our case) different family folders
        sampled_character_folders = random.sample(self.folders, self.num_classes)

        # Sample and load K + 1 (self.num_samples_per_class in our case) images per character together with their labels preserving the order!
        labels = np.eye(self.num_classes, dtype=int)
        image_paths = get_images(sampled_character_folders, labels, num_samples=self.num_samples_per_class)  #returns [(label,imagepath).....]




        # Iterate over the sampled files and create the image and label batches
        img_size = np.prod((28, 28))

        images = []  #(K+1 * N , 784)
        image_labels = []  #(one-hot vector of K+1 * N)
        for label, image_path in image_paths:
            image = self.image_file_to_array(image_path, self.dim_input)
            images.append(image)

        images = np.array(images).reshape(self.num_classes, self.num_samples_per_class, self.dim_input)
        image_labels = np.repeat(labels[:, np.newaxis, :], self.num_samples_per_class, axis=1)

        # Transpose to get the required shape [K+1, N, 784] for images and [K+1, N, N] for labels
        images = images.transpose(1, 0, 2)
        image_labels = image_labels.transpose(1, 0, 2)

        # Shuffle the order of examples from the query set
        permutation = np.random.permutation(self.num_classes)

        support_images, query_images = images[:-1], images[-1]
        support_labels, query_labels = image_labels[:-1], image_labels[-1]

        query_images = query_images[permutation]
        query_labels = query_labels[permutation]

        images = np.vstack([support_images, query_images[np.newaxis, :, :]])
        image_labels = np.vstack([support_labels, query_labels[np.newaxis, :, :]])

        # return tuple of image batch with shape [K+1, N, 784] and
        #         label batch with shape [K+1, N, N]
        return (images.astype(np.float32), image_labels.astype(np.float32))


    def __iter__(self):
        while True:
            yield self._sample()
