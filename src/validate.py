import multiprocessing
from argparse import ArgumentParser
from operator import itemgetter
from pathlib import Path
from statistics import mean
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_utils import FashionIQDataset, CIRRDataset
from utils import collate_fn, device


def compute_fiq_val_metrics(classic_val_dataset: FashionIQDataset, relative_val_dataset: CIRRDataset, combiner: callable) -> Tuple[float, float]:
    """
    Compute validation metrics on FashionIQ dataset
    :param relative_val_dataset: FashionIQ validation dataset in relative mode
    :param combiner: function which takes as input (image_features, text_features) and outputs the combined
                            features
    :return: the computed validation metrics
    """

    # Generate predictions
    index_features, predicted_features, index_names, target_names = generate_fiq_val_predictions(classic_val_dataset, relative_val_dataset, combiner)

    print(f"Compute FashionIQ {relative_val_dataset.dress_types} validation metrics")

    # Normalize the index features
    index_features = F.normalize(index_features, dim=-1).float()

    # Compute the distances and sort the results
    distances = 1 - predicted_features @ index_features.T
    sorted_indices = torch.argsort(distances, dim=-1).cpu()
    sorted_index_names = np.array(index_names)[sorted_indices]

    # Compute the ground-truth labels wrt the predictions
    labels = torch.tensor(
        sorted_index_names == np.repeat(np.array(target_names), len(index_names)).reshape(len(target_names), -1))
    assert torch.equal(torch.sum(labels, dim=-1).int(), torch.ones(len(target_names)).int())

    # Compute the metrics
    recall_at10 = (torch.sum(labels[:, :10]) / len(labels)).item() * 100
    recall_at50 = (torch.sum(labels[:, :50]) / len(labels)).item() * 100

    return recall_at10, recall_at50


def generate_fiq_val_predictions(classic_val_dataset: FashionIQDataset, relative_val_dataset: CIRRDataset, combiner: callable) -> \
        Tuple[torch.tensor, List[str]]:
    """
    Compute FashionIQ predictions on the validation set
    :param relative_val_dataset: FashionIQ validation dataset in relative mode
    :param combiner: function which takes as input (image_features, text_features) and outputs the combined
                            features
    :return: predicted features and target names
    """
    classic_val_loader = DataLoader(dataset=classic_val_dataset, batch_size=32,
            num_workers=8, pin_memory=True, collate_fn=collate_fn)
    index_features = torch.empty((0, combiner.clip_feature_dim)).to(device, non_blocking=True)
    target_features = torch.empty((0, combiner.clip_feature_dim)).to(device, non_blocking=True)
    token_index_features = torch.empty((0, combiner.img_tokens, combiner.clip_img_feature_dim)).to(device, non_blocking=True)
    index_names = []
    for names, images in tqdm(classic_val_loader):
        images = images.to(device)
        with torch.no_grad():
            batch_outputs = combiner.encode_image(images)
            batch_features = batch_outputs.image_embeds
            batch_token_features = batch_outputs.last_hidden_state
            batch_target_features = combiner.encode_features(batch_token_features, None, None)
            batch_target_features = combiner.combine_features(batch_features, None,
                                                       batch_target_features[:, :combiner.img_tokens], None)
            target_features = torch.vstack((target_features, batch_target_features))
            index_features = torch.vstack((index_features, batch_features))
            token_index_features = torch.vstack((token_index_features, batch_token_features))
            index_names.extend(names)

    print(f"Compute FashionIQ {relative_val_dataset.dress_types} validation predictions")

    relative_val_loader = DataLoader(dataset=relative_val_dataset, batch_size=32,
                                     num_workers=multiprocessing.cpu_count(), pin_memory=True, collate_fn=collate_fn,
                                     shuffle=False)

    # Get a mapping from index names to index features
    name_to_feat = dict(zip(index_names, index_features))
    name_to_feat_token = dict(zip(index_names, token_index_features))

    # Initialize predicted features and target names
    predicted_features = torch.empty((0, combiner.clip_feature_dim)).to(device, non_blocking=True)
    target_names = []

    for reference_names, batch_target_names, captions in tqdm(relative_val_loader):  # Load data
        # Randomize the training caption in four way: (a) cap1 and cap2 (b) cap2 and cap1 (c) cap1 (d) cap2
        flattened_captions: list = np.array(captions).T.flatten().tolist()
        input_captions = [
            f"{flattened_captions[i].strip('.?, ').capitalize()} and {flattened_captions[i + 1].strip('.?, ')}" for
            i in range(0, len(flattened_captions), 2)]
        text_inputs = combiner.tokenizer(input_captions, padding=True, return_tensors='pt').to(device)
        # Compute the predicted features
        with torch.no_grad():
            text_features = combiner.encode_text(text_inputs)
            if text_features.text_embeds.shape[0] == 1:
                reference_image_features = itemgetter(*reference_names)(name_to_feat).unsqueeze(0)
                reference_image_token_features = itemgetter(*reference_names)(
                    name_to_feat_token).unsqueeze(0)
            else:
                reference_image_features = torch.stack(itemgetter(*reference_names)(name_to_feat))  # To avoid unnecessary computation retrieve the reference image features directly from the index features
                reference_image_token_features = torch.stack(itemgetter(*reference_names)(name_to_feat_token))  # To avoid unnecessary computation retrieve the reference image features directly from the index features

            _, batch_fused_features, _ = combiner.encode_features(reference_image_token_features, text_features,text_inputs['attention_mask'])
            batch_predicted_features = combiner.combine_features(reference_image_features,
                                                                     text_features.text_embeds,
                                                                     batch_fused_features[:, :combiner.img_tokens],
                                                                     batch_fused_features[:, combiner.img_tokens:],
                                                                     text_mask=text_inputs['attention_mask'])

        predicted_features = torch.vstack((predicted_features, F.normalize(batch_predicted_features, dim=-1)))
        target_names.extend(batch_target_names)

    return target_features, predicted_features, index_names, target_names


def compute_cirr_val_metrics(classic_val_dataset: CIRRDataset, relative_val_dataset: CIRRDataset, combiner: callable) ->  Tuple[
     float, float, float, float, float, float, float]:
    """
    Compute validation metrics on CIRR dataset
    :param relative_val_dataset: CIRR validation dataset in relative mode
    :param combiner: function which takes as input (image_features, text_features) and outputs the combined
                            features
    :return: the computed validation metrics
    """
    # Generate predictions
    index_features, predicted_features, index_names, reference_names, target_names, group_members = \
        generate_cirr_val_predictions(classic_val_dataset, relative_val_dataset, combiner)

    print("Compute CIRR validation metrics")

    # Normalize the index features
    index_features = F.normalize(index_features, dim=-1).float()

    # Compute the distances and sort the results
    distances = 1 - predicted_features @ index_features.T
    sorted_indices = torch.argsort(distances, dim=-1).cpu()
    sorted_index_names = np.array(index_names)[sorted_indices]

    # Delete the reference image from the results
    reference_mask = torch.tensor(
        sorted_index_names != np.repeat(np.array(reference_names), len(index_names)).reshape(len(target_names), -1))
    sorted_index_names = sorted_index_names[reference_mask].reshape(sorted_index_names.shape[0],
                                                                    sorted_index_names.shape[1] - 1)
    # Compute the ground-truth labels wrt the predictions
    labels = torch.tensor(
        sorted_index_names == np.repeat(np.array(target_names), len(index_names) - 1).reshape(len(target_names), -1))

    # Compute the subset predictions and ground-truth labels
    group_members = np.array(group_members)
    group_mask = (sorted_index_names[..., None] == group_members[:, None, :]).sum(-1).astype(bool)
    group_labels = labels[group_mask].reshape(labels.shape[0], -1)

    assert torch.equal(torch.sum(labels, dim=-1).int(), torch.ones(len(target_names)).int())
    assert torch.equal(torch.sum(group_labels, dim=-1).int(), torch.ones(len(target_names)).int())

    # Compute the metrics
    recall_at1 = (torch.sum(labels[:, :1]) / len(labels)).item() * 100
    recall_at5 = (torch.sum(labels[:, :5]) / len(labels)).item() * 100
    recall_at10 = (torch.sum(labels[:, :10]) / len(labels)).item() * 100
    recall_at50 = (torch.sum(labels[:, :50]) / len(labels)).item() * 100
    group_recall_at1 = (torch.sum(group_labels[:, :1]) / len(group_labels)).item() * 100
    group_recall_at2 = (torch.sum(group_labels[:, :2]) / len(group_labels)).item() * 100
    group_recall_at3 = (torch.sum(group_labels[:, :3]) / len(group_labels)).item() * 100

    return group_recall_at1, group_recall_at2, group_recall_at3, recall_at1, recall_at5, recall_at10, recall_at50


def generate_cirr_val_predictions(classic_val_dataset: CIRRDataset, relative_val_dataset: CIRRDataset, combiner: callable) -> \
                                Tuple[torch.tensor, List[str], List[str], List[List[str]]]:
    """
    Compute CIRR predictions on the validation set
    :param relative_val_dataset: CIRR validation dataset in relative mode
    :param combiner: function which takes as input (image_features, text_features) and outputs the combined
                            features
    :return: predicted features, reference names, target names and group members
    """
    classic_val_loader = DataLoader(dataset=classic_val_dataset, batch_size=32,
            num_workers=8, pin_memory=True, collate_fn=collate_fn)
    index_features = torch.empty((0, combiner.clip_feature_dim)).to(device, non_blocking=True)
    target_features = torch.empty((0, combiner.clip_feature_dim)).to(device, non_blocking=True)
    token_index_features = torch.empty((0, combiner.img_tokens, combiner.clip_img_feature_dim)).to(device, non_blocking=True)
    index_names = []
    for names, images in tqdm(classic_val_loader):
        images = images.to(device)
        with torch.no_grad():
            batch_outputs = combiner.encode_image(images)
            batch_features = batch_outputs.image_embeds
            batch_token_features = batch_outputs.last_hidden_state
            batch_target_features = combiner.encode_features(batch_token_features, None, None)
            batch_target_features = combiner.combine_features(batch_features, None,
                                                       batch_target_features[:, :combiner.img_tokens], None)
            target_features = torch.vstack((target_features, batch_target_features))
            index_features = torch.vstack((index_features, batch_features))
            token_index_features = torch.vstack((token_index_features, batch_token_features))
            index_names.extend(names)

    print("Compute CIRR validation predictions")
    relative_val_loader = DataLoader(dataset=relative_val_dataset, batch_size=32,
            num_workers=multiprocessing.cpu_count(), pin_memory=True, collate_fn=collate_fn)

    # Get a mapping from index names to index features
    name_to_feat = dict(zip(index_names, index_features))
    name_to_feat_token = dict(zip(index_names, token_index_features))

    # Initialize predicted features, target_names, group_members and reference_names
    predicted_features = torch.empty((0, combiner.clip_feature_dim)).to(device, non_blocking=True)
    target_names = []
    group_members = []
    reference_names = []

    for batch_reference_names, batch_target_names, texts, batch_group_members in tqdm(
            relative_val_loader):  # Load data
        batch_group_members = np.array(batch_group_members).T.tolist()
        text_inputs = combiner.tokenizer(texts, padding=True, return_tensors='pt').to(device)

        # Compute the predicted features
        with torch.no_grad():
            text_features = combiner.encode_text(text_inputs)
            reference_image_features = torch.stack(itemgetter(*batch_reference_names)(name_to_feat))  # To avoid unnecessary computation retrieve the reference image features directly from the index features
            reference_image_token_features = torch.stack(itemgetter(*batch_reference_names)(name_to_feat_token))  # To avoid unnecessary computation retrieve the reference image features directly from the index features

            _, batch_fused_features, _ = combiner.encode_features(reference_image_token_features, text_features, text_inputs['attention_mask'])
            batch_predicted_features = combiner.combine_features(reference_image_features,
                                                                     text_features.text_embeds,
                                                                     batch_fused_features[:, :combiner.img_tokens],
                                                                     batch_fused_features[:, combiner.img_tokens:],
                                                                     text_mask=text_inputs['attention_mask'])

        predicted_features = torch.vstack((predicted_features, F.normalize(batch_predicted_features, dim=-1)))
        target_names.extend(batch_target_names)
        group_members.extend(batch_group_members)
        reference_names.extend(batch_reference_names)

    return target_features, predicted_features, index_names, reference_names, target_names, group_members

