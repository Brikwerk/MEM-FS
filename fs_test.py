import os
import argparse
from datetime import datetime

import torch
import torchvision
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from torch.nn import functional as F

import timm
import numpy as np
import pandas as pd
from tqdm import tqdm
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
import learn2learn as l2l
from learn2learn.data.transforms import NWays, KShots, LoadData, RemapLabels, ConsecutiveLabels
from scipy import stats

from src.model.vit import MAE_FS
from src.data.fs_sampler import FewShotSampler
from src.data.chestx import ChestX
from src.data.isic import ISICDataset
from src.model.conv4 import Conv4, Conv4_Base
from src.model.resnet12 import Resnet12
from src.model.wrn import WideResNet
from src.utils import NullLayer, pairwise_distances_logits, accuracy


parser = argparse.ArgumentParser()
# Config
parser.add_argument('--root_data_path', required=True, type=str,
                    help="""Path to the root data folder. Must be the
                            parent folder containing dataset folders.""")
parser.add_argument('--datasets', required=False, type=str, nargs='+',
                    choices=["min", "bccd", "hep",
                             "chestx", "isic", "eurosat", "plant", "ikea"],
                    help="Choose a subset of datasets to test")
parser.add_argument('--test_iters', required=True, type=int,
                    help="""Number of testing iterations per dataset.""")
parser.add_argument('--ft_iters', default=800, type=int,
                    help="""Number iterations to finetune for each for each
                    finetuning epoch.""")
# Few-shot params
parser.add_argument('--mae_shots', required=False, default=5, type=int,
                    help="""Number of labelled examples the MAE was trained on""")
parser.add_argument('--shots', required=True, type=int,
                    help="""Number of labelled examples in an episode""")
parser.add_argument('--mae_ways', required=False, default=5, type=int,
                    help="""Number of classes the MAE was trained on.""")
parser.add_argument('--ways', required=False, default=5, type=int,
                    help="""Number of classes used in an episode.""")
parser.add_argument('--mae_query_size', required=False, default=15, type=int,
                    help="""Number of unlabelled examples the MAE was trained on.""")
parser.add_argument('--query_size', required=False, default=15, type=int,
                    help="""Number of unlabelled examples in an episode.""")
# Hyperparams
parser.add_argument('--img_size', required=True, type=int,
                    help="""Image size used.""")
parser.add_argument('--ft_epochs', required=False, type=int, nargs='+',
                    help="Choose a subset of finetuning epochs to test.")
# Model
parser.add_argument('--model_type', default=False, required=True, choices=[
                        'CONV4', "WRN", "RESNET12", "RESNET18",
                        "RESNET50", "DINO_SMALL", "CONV4_BASE"
                    ], help="""Model type. Either CONV4, 
                    RESNET18, or DINO_SMALL""")
parser.add_argument('--model_path', default=False, required=True, type=str,
                    help="Path to model weights")
parser.add_argument('--num_enc_dec', default=1, required=False, type=int,
                    help="Number of encoder/decoder layers in the MAE")
parser.add_argument('--conv4_prot_size', default=512, type=int,
                    help="Size of the CONV4 prototype")
parser.add_argument('--device', default='cuda', type=str,
                    help="Device to use for testing.")
parser.add_argument('--num_classes_init', default=64, type=int,
                    help="""Number of classes to initialize the embedding
                    network with, if required.""")
args = parser.parse_args()


class FewShotLoader():
    
    def __init__(self, dataset, loader_type, shots, ways, query) -> None:
        self.dataset = dataset
        self.loader_type = loader_type
        self.shots = shots
        self.ways = ways
        self.query = query
        self.loader = self.construct_loader(
            dataset, loader_type, shots, ways, query)

    def construct_loader(self, dataset, loader_type, shots, ways, query):
        if self.loader_type == "L2L":
            l2l_dataset = l2l.data.MetaDataset(dataset)
            test_transforms = [
                NWays(l2l_dataset, ways),
                KShots(l2l_dataset, shots + query),
                LoadData(l2l_dataset),
                RemapLabels(l2l_dataset),
            ]
            test_tasks = l2l.data.TaskDataset(l2l_dataset,
                                                task_transforms=test_transforms,
                                                num_tasks=2000)
            return DataLoader(test_tasks, pin_memory=True, shuffle=True)
        else:
            return FewShotSampler(dataset, ways, shots, query)
    
    def get_episode(self):
        if self.loader_type == "L2L":
            return next(iter(self.loader))
        else:
            return self.loader.get_batch()


def load_datasets(root_path: str, img_size: int, shots: int, ways: int,
                  query: int, dataset_subset: list = None) -> list:
    datasets = []

    # ImageNet-1K Mean/Std. for general reuse in datasets
    mean = torch.tensor([0.4707, 0.4495, 0.4026])
    std = torch.tensor([0.2843, 0.2752, 0.2903])

    if "min" in dataset_subset or dataset_subset is None:
        min_transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=mean, std=std), # ImageNet Dataset and other natural image datasets
            T.Resize(size=(img_size, img_size)),
        ])
        min_dataset = torchvision.datasets.ImageFolder(
            os.path.join(root_path, 'mini-imagenet', 'test'), transform=min_transform)
        min_loader = FewShotLoader(min_dataset, "L2L", shots, ways, query)
        datasets.append((min_loader, "MIN"))
        print("MIN Loaded")

    # HEp Dataset
    if "hep" in dataset_subset or dataset_subset is None:
        hep_transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.7940, 0.7940, 0.7940], std=[0.1920, 0.1920, 0.1920]), # HEp-2 Dataset
            T.Resize(size=(IMG_SIZE, IMG_SIZE)),
        ])
        hep_dataset = torchvision.datasets.ImageFolder(
            root=os.path.join(root_path, 'HEp-Dataset'), transform=hep_transform)
        hep_loader = FewShotLoader(hep_dataset, "L2L", shots, ways, query)
        datasets.append((hep_loader, "HEp-2"))
        print("HEp-2 Loaded")

    # BCCD WBC Dataset
    if "bccd" in dataset_subset or dataset_subset is None:
        wbc_transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.6659, 0.6028, 0.7932], std=[0.1221, 0.1698, 0.0543]), # BCCD Dataset
            T.Resize(size=(IMG_SIZE, IMG_SIZE)),
        ])
        wbc_dataset = torchvision.datasets.ImageFolder(
            root=os.path.join(root_path, 'wbc-aug'), transform=wbc_transform)
        wbc_loader = FewShotLoader(wbc_dataset, "L2L", shots, ways, query)
        datasets.append((wbc_loader, "BCCD"))
        print("BCCD Loaded")

    # NHS Chest X-Ray Dataset
    if "chestx" in dataset_subset or dataset_subset is None:
        chestx_transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.4920, 0.4920, 0.4920], std=[0.2288, 0.2288, 0.2288]), # ChestX
            T.Resize(size=(IMG_SIZE, IMG_SIZE)),
        ])
        chestx_dataset = ChestX(os.path.join(
            root_path, "chestx"), transform=chestx_transform)
        chestx_loader = FewShotLoader(chestx_dataset, "FSS", shots, ways, query)
        datasets.append((chestx_loader, "ChestX"))
        print("ChestX Loaded")

    # Skin Lesion Dataset
    if "isic" in dataset_subset or dataset_subset is None:
        isic_transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.7635, 0.5461, 0.5705], std=[0.0891, 0.1179, 0.1325]), # ISIC
            T.Resize(size=(IMG_SIZE, IMG_SIZE)),
        ])
        isic_dataset = ISICDataset(
            os.path.join(root_path, "isic2018"), transform=isic_transform)
        isic_loader = FewShotLoader(isic_dataset, "FSS", shots, ways, query)
        datasets.append((isic_loader, "ISIC"))
        print("ISIC Loaded")

    # Eurosat Satellite Image Dataset
    if "eurosat" in dataset_subset or dataset_subset is None:
        eurosat_transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.3444, 0.3803, 0.4078], std=[0.0884, 0.0621, 0.0521]), # EuroSat Dataset
            T.Resize(size=(IMG_SIZE, IMG_SIZE)),
        ])
        eurosat_dataset = torchvision.datasets.ImageFolder(
            root=os.path.join(root_path, 'eurosat'), transform=eurosat_transform)
        eurosat_loader = FewShotLoader(eurosat_dataset, "L2L", shots, ways, query)
        datasets.append((eurosat_loader, "EuroSat"))
        print("EuroSat Loaded")

    # Plant Disease Dataset
    if "plant" in dataset_subset or dataset_subset is None:
        plant_transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.4662, 0.4888, 0.4101], std=[0.1707, 0.1438, 0.1875]), # Plant Disease Dataset
            T.Resize(size=(IMG_SIZE, IMG_SIZE)),
        ])
        plant_dataset = torchvision.datasets.ImageFolder(
            root=os.path.join(root_path, 'plant_disease', 'train'), transform=plant_transform)
        plant_loader = FewShotLoader(plant_dataset, "L2L", shots, ways, query)
        datasets.append((plant_loader, "Plant Disease"))
        print("Plant Disease Loaded")

    # IKEA Few-Shot Dataset
    if "ikea" in dataset_subset or dataset_subset is None:
        ikea_transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.7073, 0.6915, 0.6744], std=[0.2182, 0.2230, 0.2312]), # Plant Disease Dataset
            T.Resize(size=(IMG_SIZE, IMG_SIZE)),
        ])
        ikea_dataset = torchvision.datasets.ImageFolder(
            root=os.path.join(root_path, 'ikea'), transform=ikea_transform)
        ikea_loader = FewShotLoader(ikea_dataset, "L2L", shots, ways, query)
        datasets.append((ikea_loader, "IKEA-FS"))
        print("IKEA Loaded")

    return datasets


def get_model(model_type, num_classes, ways, shots, query, img_size, 
              num_encoder_decoder, model_path, device, conv4_prot_size=512):
    conv_embed = None
    prototype_size = None
    if model_type == "CONV4":
        conv_embed = Conv4(num_classes, use_fc=False, prototype_size=conv4_prot_size)
        prototype_size = conv4_prot_size
    elif model_type == "CONV4_BASE":
        conv_embed = Conv4_Base(avgpool=True)
        conv_embed.add_classifier(64)
        prototype_size = 64
    elif MODEL_TYPE == "RESNET12":
        conv_embed = Resnet12(1, 0.1, num_classes=num_classes, use_fc=False)
        prototype_size = 512
    elif MODEL_TYPE == "RESNET18":
        conv_embed = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
        conv_embed.fc = NullLayer()
        prototype_size = 512
    elif MODEL_TYPE == "RESNET50":
        conv_embed = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
        conv_embed.fc = NullLayer()
        prototype_size = 2048
    elif MODEL_TYPE == "WRN":
        conv_embed = WideResNet(28, 10, use_fc=False, num_classes=num_classes)
        conv_embed.fc = NullLayer()
        prototype_size = 640
    elif MODEL_TYPE == "DINO_SMALL":
        conv_embed = timm.create_model('vit_small_patch8_224_dino', pretrained=True)
        prototype_size = 384

    print("Using model", model_type)

    model = MAE_FS(
        ways=ways,
        shots=shots,
        query_size=query,
        image_size=img_size,
        patch_size=14,
        num_layers=num_encoder_decoder,
        num_heads=16,
        hidden_dim=60,
        mlp_dim=3060,
        decoder_dim=512,
        num_classes=5,
        num_decoder_layers=num_encoder_decoder,
        conv_embed=conv_embed,
        prototype_dim=prototype_size
    )
    model.load_state_dict(torch.load(model_path))

    # Freeze conv_embed layer
    for param in model.conv_embed.parameters():
        param.requires_grad = False

    model.to(device)

    return model


if __name__ == "__main__":
    ROOT_DATA_PATH = args.root_data_path
    DATASET_SUBSET = args.datasets
    TEST_ITERS = args.test_iters
    FT_ITERS = args.ft_iters
    MAE_SHOTS = args.mae_shots
    SHOTS = args.shots
    MAE_WAYS = args.mae_ways
    WAYS = args.ways
    MAE_QUERY = args.mae_query_size
    QUERY = args.query_size
    IMG_SIZE = args.img_size
    MODEL_TYPE = args.model_type
    MODEL_PATH = args.model_path
    NUM_ENCODER_DECODER = args.num_enc_dec
    CONV4_PROT_SIZE = args.conv4_prot_size
    DEVICE = args.device
    NUM_CLASSES = args.num_classes_init

    datasets = load_datasets(ROOT_DATA_PATH, IMG_SIZE, SHOTS, WAYS, QUERY,
                             dataset_subset=DATASET_SUBSET)

    # Finetuning and Testing
    if args.ft_epochs is None:
        FT_EPOCHS = [0,1,2,5,10,20,50,100]
    else:
        FT_EPOCHS = args.ft_epochs
    WARMUP_EPOCHS = 5
    LR = 1.5e-5

    # Support/query indices for extraction when testing
    support_indices = np.zeros(WAYS * (SHOTS + QUERY), dtype=bool)
    selection = np.arange(WAYS) * (SHOTS + QUERY)
    for offset in range(SHOTS):
        support_indices[selection + offset] = True
    query_indices = torch.from_numpy(~support_indices)
    support_indices = torch.from_numpy(support_indices)

    results = [["Dataset", "Epoch", "MAE Acc.", "Backbone Acc.", "Uncertainty"]]
    for loader, dataset_name in datasets:
        print("--- Testing", dataset_name, "---")

        dataset_results = []

        # Extract a single batch to finetune on
        ft_data, ft_labels = loader.get_episode()
        ft_data, ft_labels = ft_data.to(DEVICE).squeeze(0), ft_labels.to(DEVICE).squeeze(0)

        for EPOCHS in FT_EPOCHS:
            print(f"--- Finetuning {MODEL_TYPE} for {EPOCHS} on {dataset_name}")
            print("Loading model...")
            model = get_model(MODEL_TYPE, NUM_CLASSES, MAE_WAYS, MAE_SHOTS, MAE_QUERY,
                              IMG_SIZE, NUM_ENCODER_DECODER, MODEL_PATH,
                              DEVICE, conv4_prot_size=CONV4_PROT_SIZE)
            
            optimizer = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.95), weight_decay=0.05)
            scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=WARMUP_EPOCHS, max_epochs=EPOCHS)
            scaler = torch.cuda.amp.GradScaler()
            
            if EPOCHS > 0:
                print("Finetuning...")
                data, labels = ft_data, ft_labels
                sort = torch.sort(labels)
                data = data.squeeze(0)[sort.indices].squeeze(0)
                labels = labels.squeeze(0)[sort.indices].squeeze(0)
                
                for epoch in range(EPOCHS):
                    losses = []
                    accuracies = []
                    for i in tqdm(range(FT_ITERS)):

                        # Regularize data through shuffling class sections
                        d, c, h, w = data.shape
                        shuffled_data = data.reshape(WAYS, (SHOTS + QUERY), c, h, w)
                        shuffled_prots = torch.randperm(SHOTS + QUERY)
                        shuffled_data = shuffled_data[:, shuffled_prots]
                        shuffled_ways = torch.randperm(WAYS)
                        shuffled_data = shuffled_data[shuffled_ways].reshape(d,c,h,w)

                        with torch.cuda.amp.autocast():
                            output, prototypes, _, _ = model(
                                shuffled_data, SHOTS, QUERY, WAYS, fs_mode=True, augment=False)

                            loss = (output.squeeze(0) - prototypes.squeeze(0)) ** 2
                            loss = loss.mean(dim=-1).sum()
                            losses.append(loss.item())

                            scaler.scale(loss).backward()

                            support = output.reshape(output.size(0), WAYS, (SHOTS + QUERY), -1)
                            support = support.mean(dim=2).squeeze(0)
                            query = prototypes.squeeze(0)
                            query = query[query_indices]

                            logits = pairwise_distances_logits(query, support)
                            query_labels = labels[query_indices].long()
                            acc = accuracy(logits.squeeze(0), query_labels.squeeze(0))
                            accuracies.append(acc.item())

                            scaler.step(optimizer)
                            scaler.update()
                            optimizer.zero_grad()

                    print(f'Epoch {epoch}: Loss {np.mean(losses)} Accuracy {np.mean(accuracies)}')
                    scheduler.step()
            
            print("Testing...")
            accuracies = []
            prot_accuracies = []
            losses = []
            for i in tqdm(range(TEST_ITERS)):
                data, labels = loader.get_episode()
                data, labels = data.to(DEVICE).squeeze(0), labels.to(DEVICE).squeeze(0)

                sort = torch.sort(labels)
                data = data.squeeze(0)[sort.indices].squeeze(0)
                labels = labels.squeeze(0)[sort.indices].squeeze(0)

                support_labels = labels[support_indices].long()
                labels = labels[query_indices].long()

                with torch.cuda.amp.autocast():
                    with torch.no_grad():
                        output, prototypes, _, _ = model(
                            data, SHOTS, QUERY, WAYS, fs_mode=True, add_noise=False)

                # Use average of reconstructed examples in the output
                support = output.reshape(output.size(0), WAYS, (SHOTS + QUERY), -1)
                # Replace prototypes in output with originals
                support[:, :, :SHOTS] = prototypes.reshape(output.size(0), WAYS, (SHOTS + QUERY), -1)[:, :, :SHOTS]
                support = support.mean(dim=2).squeeze(0)

                query = prototypes.squeeze(0)
                query = query[query_indices]

                # Calculate accuracy
                logits = pairwise_distances_logits(query, support)
                acc = accuracy(logits.squeeze(0), labels.squeeze(0))
                accuracies.append(acc.item())

                # Use average support from prototypes
                support = prototypes.squeeze(0)[support_indices]
                support = support.reshape(WAYS, SHOTS, -1)
                support = support.mean(dim=1).squeeze(0)

                query = prototypes.squeeze(0)
                query = query[query_indices]

                # Calculate accuracy
                logits = pairwise_distances_logits(query, support)
                acc = accuracy(logits.squeeze(0), labels.squeeze(0))
                prot_accuracies.append(acc.item())

            print(f'Accuracy {np.mean(accuracies)*100}% Prot Accuracy {np.mean(prot_accuracies)*100}%')

            confidence_interval = stats.norm.interval(0.95, loc=np.mean(accuracies), scale=stats.sem(accuracies))
            confidence_interval = ((confidence_interval[1] - confidence_interval[0])/2)*100
            print(f'95% confidence interval {confidence_interval}%')

            results.append([dataset_name, EPOCHS, round(np.mean(accuracies)*100, 2), 
                            round(np.mean(prot_accuracies)*100, 2), round(confidence_interval, 2)])
    
    results = pd.DataFrame(results)
    csv_filename = f"MAE_{MODEL_TYPE}_Results_{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}.csv"
    results.to_csv(csv_filename, header=False)
