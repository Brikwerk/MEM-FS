import torch
import numpy as np


class FewShotSampler():
    def __init__(self, dataset, ways, shots, query):
        self.dataset = dataset
        self.ways = ways
        self.shots = shots
        self.query = query

        self.idxs_for_class = {}
        # Get all the idxs for each class in train_dataset
        for i in range(len(self.dataset)):
            path, label = self.dataset.samples[i]
            if label not in self.idxs_for_class:
                self.idxs_for_class[label] = []
            self.idxs_for_class[label].append(i)

        self.class_keys = list(self.idxs_for_class.keys())

    def get_batch(self):
        # Get a random classes and sample idxs for each class
        classes = np.random.choice(self.class_keys, self.ways, replace=False)
        idxs = []
        for c in classes:
            idxs.extend(np.random.choice(self.idxs_for_class[c], self.shots + self.query, replace=False))
        
        # Retrieve selected data from dataset
        x = None
        for i in idxs:
            data, label = self.dataset[i]
            if x is None:
                x = data.unsqueeze(0)
            else:
                x = torch.cat((x, data.unsqueeze(0)), 0)

        
        # Create a list of remapped targets that range from 0 - (ways - 1).
        # Each class contains shots + query examples.
        targets = []
        for i in range(self.ways):
            targets.extend([i] * (self.shots + self.query))
        targets = torch.tensor(targets)

        return x, targets