import torch
from torch import distributed
from torch import backends
from torch import cuda
from torch import utils
from torch import optim
from torch import nn
from torchvision import models
from torchvision import transforms
from models import SelfAttentionResNet
from datasets import ImageDataset
from tensorboardX import SummaryWriter
from distributed import *
from utils import *
import argparse
import json
import os


def main(args):

    # Multi-process single-GPU distributed training
    # See https://pytorch.org/docs/1.1.0/distributed.html
    # and https://pytorch.org/docs/stable/nn.html#torch.nn.parallel.DistributedDataParallel

    # On PyTorch, we should specify `MASTER_ADDR` and `MASTER_PORT` by environment variable.
    init_process_group(backend='nccl')  # For PyTorch
    # On Parrots, we don't have to specify them.
    # distributed.init_process_group(backend='nccl') # For Parrots

    with open(args.config) as file:
        config = json.load(file)
        config.update(vars(args))
        config = apply_dict(Dict, config)

    backends.cudnn.benchmark = True
    backends.cudnn.fastest = True

    # Force each process to run on a single device.
    cuda.set_device(distributed.get_rank() % cuda.device_count())

    train_dataset = ImageDataset(
        root=config.train_root,
        meta=config.train_meta,
        transform=transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)
            )
        ])
    )
    val_dataset = ImageDataset(
        root=config.val_root,
        meta=config.val_meta,
        transform=transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)
            )
        ])
    )

    # Sampler for distributed training.
    # This guarantees that each process loads a different batch in each training step.
    train_sampler = utils.data.distributed.DistributedSampler(train_dataset)
    val_sampler = utils.data.distributed.DistributedSampler(val_dataset)

    train_data_loader = utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=config.local_batch_size,
        sampler=train_sampler,
        num_workers=config.num_workers,
        pin_memory=True
    )
    val_data_loader = utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=config.local_batch_size,
        sampler=val_sampler,
        num_workers=config.num_workers,
        pin_memory=True
    )

    model = SelfAttentionResNet(
        conv_param=Dict(in_channels=3, out_channels=64, kernel_size=7, padding=3, mixtures=4),
        pool_param=Dict(kernel_size=7, padding=3, stride=4),
        residual_params=[
            Dict(in_channels=64, out_channels=64, kernel_size=7, padding=3, stride=1, groups=8, expansion=4, blocks=3),
            Dict(in_channels=256, out_channels=128, kernel_size=7, padding=3, stride=2, groups=8, expansion=4, blocks=4),
            Dict(in_channels=512, out_channels=256, kernel_size=7, padding=3, stride=2, groups=8, expansion=4, blocks=6),
            Dict(in_channels=1024, out_channels=512, kernel_size=7, padding=3, stride=2, groups=8, expansion=4, blocks=3),
        ],
        num_classes=1000
    )

    model.cuda()

    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[distributed.get_rank() % cuda.device_count()])

    # Scale learning rate following the `global` batch size (`local batch size` * `world size`)
    config.global_batch_size = config.local_batch_size * distributed.get_world_size()
    config.optimizer.lr *= config.global_batch_size / config.global_batch_denom
    optimizer = optim.SGD(model.parameters(), **config.optimizer)

    last_epoch = -1
    global_step = 0
    if config.checkpoint:
        checkpoint = Dict(torch.load(config.checkpoint))
        model.load_state_dict(checkpoint.model_state_dict)
        optimizer.load_state_dict(checkpoint.optimizer_state_dict)
        last_epoch = checkpoint.last_epoch
        global_step = checkpoint.global_step

    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, **config.lr_scheduler, last_epoch=last_epoch)

    summary_writer = SummaryWriter('logs')

    def train():
        nonlocal global_step
        model.train()
        for images, labels in train_data_loader:
            images = images.cuda()
            labels = labels.cuda()
            optimizer.zero_grad()
            logits = model(images).squeeze()
            loss = nn.functional.cross_entropy(logits, labels)
            loss.backward(retain_graph=False)
            if not isinstance(model, nn.parallel.DistributedDataParallel):
                average_gradients(model.parameters())
            optimizer.step()
            predictions = logits.topk(k=1, dim=1)[1].squeeze()
            accuracy = torch.mean((predictions == labels).float())
            average_tensors([loss, accuracy])
            global_step += 1
            if not distributed.get_rank():
                summary_writer.add_scalars(
                    main_tag='loss',
                    tag_scalar_dict=dict(train=loss),
                    global_step=global_step
                )
                summary_writer.add_scalars(
                    main_tag='accuracy',
                    tag_scalar_dict=dict(train=accuracy),
                    global_step=global_step
                )
                print(f'[training] epoch: {epoch} global_step: {global_step} '
                      f'loss: {loss:.4f} accuracy: {accuracy:.4f}')

    @torch.no_grad()
    def validate():
        model.eval()
        losses = []
        accuracies = []
        for images, labels in val_data_loader:
            images = images.cuda()
            labels = labels.cuda()
            logits = model(images).squeeze()
            loss = nn.functional.cross_entropy(logits, labels)
            predictions = logits.topk(k=1, dim=1)[1].squeeze()
            accuracy = torch.mean((predictions == labels).float())
            average_tensors([loss, accuracy])
            losses.append(loss)
            accuracies.append(accuracy)
        loss = torch.mean(torch.stack(losses)).item()
        accuracy = torch.mean(torch.stack(accuracies)).item()
        if not distributed.get_rank():
            summary_writer.add_scalars(
                main_tag='loss',
                tag_scalar_dict=dict(val=loss),
                global_step=global_step
            )
            summary_writer.add_scalars(
                main_tag='accuracy',
                tag_scalar_dict=dict(val=accuracy),
                global_step=global_step
            )
            print(f'[validation] epoch: {epoch} global_step: {global_step} '
                  f'loss: {loss:.4f} accuracy: {accuracy:.4f}')

    def save():
        if not distributed.get_rank():
            os.makedirs('checkpoints', exist_ok=True)
            torch.save(dict(
                model_state_dict=model.state_dict(),
                optimizer_state_dict=optimizer.state_dict(),
                last_epoch=epoch,
                global_step=global_step
            ), os.path.join('checkpoints', f'epoch_{epoch}'))

    if config.training:
        broadcast_tensors(model.state_dict().values())
        for epoch in range(last_epoch + 1, config.num_epochs):
            train_sampler.set_epoch(epoch)
            lr_scheduler.step(epoch)
            train()
            validate()
            save()

    if config.validation:
        broadcast_tensors(model.state_dict().values())
        validate()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Self-Attention ResNet')
    parser.add_argument('--config', type=str, default='config.json')
    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument('--training', action='store_true')
    parser.add_argument('--validation', action='store_true')
    args = parser.parse_args()

    main(args)
