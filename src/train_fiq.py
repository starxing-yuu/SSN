from comet_ml import Experiment
import json
import multiprocessing
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
from statistics import mean, harmonic_mean, geometric_mean
import numpy as np

import torch
from torch import optim, nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import models

from data_utils import base_path, FashionIQDataset
from utils import collate_fn, update_train_running_results, generate_randomized_fiq_caption, set_train_bar_description, save_model, device
from validate import compute_fiq_val_metrics


def combiner_training_fiq(args):
    """
    Train the Combiner on Fashion-IQ dataset
    """
    # Make training folder
    training_start = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    training_path: Path = Path(
        base_path / 'outputs' / f"{args.dataset}" / f"{args.model}_{training_start}")
    training_path.mkdir(exist_ok=False, parents=True)

    model = models.setup(args).to(device, non_blocking=True)

    def preprocess(im):
        return model.processor(images=im, return_tensors='pt')

    # Define the validation datasets and extract the validation index features
    val_dress_types = ['dress', 'toptee', 'shirt']
    train_dress_types = ['dress', 'toptee', 'shirt']
    idx_to_dress_mapping = {}
    relative_val_datasets = []
    classic_val_datasets = []
    for idx, dress_type in enumerate(val_dress_types):
        idx_to_dress_mapping[idx] = dress_type
        relative_val_dataset = FashionIQDataset('val', [dress_type], 'relative', preprocess, )
        relative_val_datasets.append(relative_val_dataset)
        classic_val_dataset = FashionIQDataset('val', [dress_type], 'classic', preprocess, )
        classic_val_datasets.append(classic_val_dataset)
    relative_train_dataset = FashionIQDataset('train', train_dress_types, 'relative', preprocess)

    # Define the model and the train dataset
    relative_train_loader = DataLoader(dataset=relative_train_dataset, batch_size=args.batch_size,
                                       num_workers=4,
                                       pin_memory=True, collate_fn=collate_fn, drop_last=True, shuffle=True)

    # Define the optimizer, the loss and the grad scaler
    clip_params, proj_params, modality_params, fusion_params, head_params =[], [], [], [], []
    for name, para in model.named_parameters():
        if 'clip' in name:
            clip_params += [para]
        elif 'token_proj' in name:
            proj_params += [para]
        elif 'token_type' in name:
            modality_params += [para]
        elif 'fusion' in name or 'token_selection' in name:
            fusion_params += [para]
        else:
            head_params += [para]

    optimizer = optim.AdamW([{'params': head_params},
                            {'params': fusion_params, 'lr': args.lr * args.lr_ratio * 4},
                            {'params': modality_params, 'lr': args.lr * args.lr_ratio * 4},
                            {'params': proj_params, 'lr': args.lr * args.lr_ratio * 2},
                            {'params': filter(lambda p: p.requires_grad, clip_params), 'lr': args.lr * args.lr_ratio, 'betas': (0.9, 0.999), 'eps': 1e-4}
                            ], lr=args.lr)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
    scaler = torch.cuda.amp.GradScaler()

    # When save_best == True initialize the best results to zero
    if args.save_best:
        best_avg_recall = 0

    # Start with the training loop
    print('Training loop started')
    for epoch in range(args.num_epochs):
        with experiment.train():
            if epoch == 1:
                print('Only the CLIP text encoder will be fine-tuned')
                for param in model.clip_vision_model.parameters():
                    param.requires_grad = False
            train_running_results = {'images_in_epoch': 0, 'accumulated_train_loss': 0}
            model.train()
            train_bar = tqdm(relative_train_loader, ncols=150)
            for idx, (reference_images, target_images, texts) in enumerate(train_bar):  # Load a batch of triplets
                step = len(train_bar) * epoch + idx
                num_samples = len(reference_images)

                optimizer.zero_grad()

                reference_images = reference_images.to(device)
                target_images = target_images.to(device)
                # Randomize the training caption in four way: (a) cap1 and cap2 (b) cap2 and cap1 (c) cap1 (d) cap2
                flattened_captions: list = np.array(texts).T.flatten().tolist()
                input_captions = generate_randomized_fiq_caption(flattened_captions)
                text_inputs = model.tokenizer(input_captions, padding=True, return_tensors='pt').to(device)

                # Compute the logits and loss
                with torch.cuda.amp.autocast():
                    ground_truth = torch.arange(num_samples, dtype=torch.long, device=device)
                    loss = model(reference_images, text_inputs, target_images, ground_truth)

                # Backpropagate and update the weights
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                experiment.log_metric('step_loss', loss.detach().cpu().item(), step=step)
                update_train_running_results(train_running_results, loss, num_samples)
                set_train_bar_description(train_bar, epoch, args.num_epochs, train_running_results)

            train_epoch_loss = float(
                train_running_results['accumulated_train_loss'] / train_running_results['images_in_epoch'])
            experiment.log_metric('epoch_loss', train_epoch_loss, epoch=epoch)

            scheduler.step()
            experiment.log_metric('learning_rate', scheduler.get_last_lr()[0], epoch=epoch)

        if epoch % args.validation_frequency == 0:
            with experiment.validate():
                model.eval()
                recalls_at10 = []
                recalls_at50 = []
                # Compute and log validation metrics
                for relative_val_dataset, classic_val_dataset, idx in zip(relative_val_datasets, classic_val_datasets,
                                                                          idx_to_dress_mapping):
                    recall_at10, recall_at50 = compute_fiq_val_metrics(classic_val_dataset, relative_val_dataset, model)
                    recalls_at10.append(recall_at10)
                    recalls_at50.append(recall_at50)

                results_dict = {}
                for i in range(len(recalls_at10)):
                    results_dict[f'{idx_to_dress_mapping[i]}_recall_at10'] = recalls_at10[i]
                    results_dict[f'{idx_to_dress_mapping[i]}_recall_at50'] = recalls_at50[i]
                results_dict.update({
                    f'average_recall_at10': mean(recalls_at10),
                    f'average_recall_at50': mean(recalls_at50),
                    f'average_recall': (mean(recalls_at50) + mean(recalls_at10)) / 2
                })

                print(json.dumps(results_dict, indent=4))
                experiment.log_metrics(
                    results_dict,
                    epoch=epoch
                )

                # Save model
                if args.save_training:
                    if args.save_best and results_dict['average_recall'] > best_avg_recall:
                        best_avg_recall = results_dict['average_recall']
                        save_model('combiner', epoch, model, training_path)
                    elif not args.save_best:
                        save_model(f'combiner_{epoch}', epoch, model, training_path)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="should be either 'CIRR' or 'fashionIQ'")
    parser.add_argument("--model", type=str, required=True, help="could be 'FusionModel'")
    parser.add_argument("--clip_model_name", default="openai/clip-vit-base-patch32", type=str, help="CLIP model to use")

    parser.add_argument("--projection_dim", default=1024, type=int, help='Combiner projection dim')
    parser.add_argument("--hidden_dim", default=2048, type=int, help="Combiner hidden dim")

    parser.add_argument("--lr", default=4e-5, type=float, help="Init learning rate")
    parser.add_argument("--lr_step_size", default=5, type=int)
    parser.add_argument("--lr_gamma", default=0.1, type=float)
    parser.add_argument("--lr_ratio", default=0.2, type=float)

    parser.add_argument("--num_epochs", default=100, type=int, help="number training epochs")
    parser.add_argument("--batch_size", default=384, type=int, help="Batch size of the Combiner training")
    parser.add_argument("--validation_frequency", default=1, type=int, help="Validation frequency expressed in epochs")

    parser.add_argument("--save_training", action='store_true', help="Whether save the training model")
    parser.add_argument("--save_best", action='store_true', help="Save only the best model during training")

    # comet.ml
    parser.add_argument("--api_key", type=str, default='xxx', help="api for Comet logging")
    parser.add_argument("--workspace", type=str, help="workspace of Comet logging")
    parser.add_argument("--project_name", default="xxx", type=str, help="name of the project on Comet")

    args = parser.parse_args()

    if args.api_key and args.workspace:
        print("Comet logging ENABLED")
        experiment = Experiment(
            api_key=args.api_key,
            project_name=args.project_name,
            workspace=args.workspace,
            disabled=False
        )
    else:
        print("Comet loging DISABLED, to enable it provide an api key and a workspace")
        experiment = Experiment(
            api_key="",
            project_name="",
            workspace="",
            disabled=True
        )

    experiment.log_code(folder=str(base_path / 'code'))
    experiment.log_parameters(args)

    combiner_training_fiq(args)

