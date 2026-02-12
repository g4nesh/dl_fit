# BONe_DLFit/core/fitting_loop.py
import copy
import gc
import json
import multiprocessing
import numpy as np
import os
import random
import threading
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
import time
from pathlib import Path
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from fvcore.nn import FlopCountAnalysis, parameter_count_table
from BONe_utils.utils import (
    set_global_seed, 
    log_to_console,
    get_console_queue, 
    get_progress_queue, 
    load_config,
    CONFIG_FILE,
    CONFIG_DIR,
    remove_module_prefix,
    GPUMonitor,
    _get_manager,
)
from BONe_utils.model_factory import model_factory
from BONe_utils.FocalTverskyLoss import FocalTverskyLoss

# ---------------------------------------------------------------------------------
# Interprocess communication to enable the Stop button
# ---------------------------------------------------------------------------------
ctx = multiprocessing.get_context('spawn')
stop_event = ctx.Event()

# ---------------------------------------------------------------------------------
# Ensures each worker gets a unique deterministic seed (Linux)
# ---------------------------------------------------------------------------------
def worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# ---------------------------------------------------------------------------------
# DDP Setup and Clean-up Functions
# ---------------------------------------------------------------------------------
def setup_ddp(rank, world_size):
    # ---- Initialize the distributed environment ----
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355' # verified that neither 12355 nor 29500 are unused
    
    # ---- Initialize the process group ----
    dist.init_process_group('nccl', rank=rank, world_size=world_size)
    
    # ---- Set the device for this process ----
    torch.cuda.set_device(rank)

def cleanup_ddp():
    # ---- Clean up the distributed environment ----
    if dist.is_initialized():
        dist.destroy_process_group()

# ---------------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------------
def training_pipeline(
    rank, 
    world_size, 
    config: dict, 
    stop_event=None, 
    console_queue=None, 
    progress_queue=None,
    gpu_shared_dict=None,
):
    """
    Args:
        rank: Process rank (GPU ID)
        world_size: Total number of processes (GPUs)
        config: Configuration dictionary
        stop_event: Event to signal early stopping
        console_queue: Allows console updates
        progress_queue: Allows progress bar updates
        gpu_shared_dict: Stores gpu monitor stats
    """
    # ---- Setup DDP if using multiple GPUs ----
    is_distributed = world_size > 1
    if is_distributed:
        setup_ddp(rank, world_size)
    
    # ---- Determine if this is the main process (for logging and saving) ----
    is_main_process = rank == 0
    
    # ---- DDP-compatible local GUI logging helper (only rank 0 sends logs to GUI) ----
    def ddp_log(message, log_file_path=None):
        if is_main_process and console_queue is not None:
            try:
                console_queue.put(message, timeout=0.1)
            except Exception:
                pass

        if log_file_path is not None:
            try:
                with open(log_file_path, 'a', encoding='utf-8') as f:
                    f.write(message + '\n')
            except Exception:
                pass

    # ---- DDP-compatible progress bar helper (only rank 0 sends updates to GUI) ----
    def ddp_progress(value):
        if is_main_process and progress_queue is not None:
            try:
                progress_queue.put(value, timeout=0.1)
            except Exception:
                pass

    # ---- DDP-compatible stop button helper (ensures that all ranks see early-exit command) ----
    def ddp_stop():
        local_stop = stop_event is not None and stop_event.is_set()

        if not is_distributed:
            return local_stop

        # ---- Broadcast stop signal from rank 0 to all ranks ----
        stop_tensor = torch.tensor(
            1 if (is_main_process and local_stop) else 0,
            device=device if torch.cuda.is_available() else 'cpu'
        )

        dist.broadcast(stop_tensor, src=0)
        return stop_tensor.item() == 1
    
    # ---- Unpack config ---- 
    train_pairs = config['train_pairs']
    val_pairs = config['val_pairs']
    gpu_mode = config['gpu_mode']
    num_classes = config['num_classes']
    channels = config['in_channels']
    patch_size = config['patch_size']
    patch_depth = config['patch_depth']
    tile_depth = config['tile_depth']
    max_overlap = config['max_overlap']
    num_patches_per_tile = config['num_patches_per_tile']
    batch_size = config['batch_size']
    augmentation = config['augmentation']
    norm_mode = config['norm_mode']
    fitting_mode = config['fitting_mode']
    is_3d = config['is_3d']
    arch = config['model_arch']
    menu_index = config['menu_index']
    backbone_name = config['backbone']
    stride_str = config['stride']
    num_epochs = config['num_epochs']
    criterion_mode = config['loss_fxn']
    criterion_alpha = config['criterion_alpha']
    criterion_beta = config['criterion_beta']
    criterion_gamma = config['criterion_gamma']
    criterion_foc_wt = config['criterion_foc_wt']
    criterion_tver_wt = config['criterion_tver_wt']
    optim_name = config['optimizer']
    optim_weight_decay = config['optim_weight_decay']
    optim_momentum = config['optim_momentum']
    learning_rate = config['learn_rate']
    peak_ram_gb = config['peak_ram']
    pt_workers = config['pt_workers']
    prefetch_factor = config['prefetch_factor']
    save_path = Path(config['save_model_path'])
    save_modelname = config['model_filename']
    log_path = Path(config['log_path'])
    weights = config['load_weights']
    weights_path = Path(config['load_weights_path']) if config['load_weights_path'] is not None else None 
    seed = config['seed']

    # ---- Return PyTorch Generator to use with Data ----
    generator = set_global_seed(seed) if seed is not None else None

    # ---- Dataset constructor ----
    if fitting_mode in ('2D','2.5D'):
        import segmentation_models_pytorch as smp
        from .data_preprocess import ImageTilesDataset
        train_dataset = ImageTilesDataset(
            input_mask_pairs=train_pairs,
            transform=augmentation,
            patch_size=(patch_size, patch_size),
            num_patches_per_tile=num_patches_per_tile,
            max_overlap=max_overlap,
            in_channels=channels,
            stride=channels,
            norm=norm_mode,
        )

        val_dataset = ImageTilesDataset(
            input_mask_pairs=val_pairs,
            transform=False,
            patch_size=(patch_size, patch_size),
            num_patches_per_tile=num_patches_per_tile,
            max_overlap=max_overlap,
            in_channels=channels,
            stride=channels,
            norm=norm_mode,
        )

    elif fitting_mode == '3D':
        import segmentation_models_pytorch_3d as smp
        from .data_preprocess import Image3dTilesDataset
        train_dataset = Image3dTilesDataset(
            input_mask_pairs=train_pairs,
            transform=augmentation,
            tile_depth=tile_depth,
            patch_size=(patch_size, patch_size, patch_depth),
            num_patches_per_tile=num_patches_per_tile,
            max_overlap=max_overlap,
            norm=norm_mode,
        )

        val_dataset = Image3dTilesDataset(
            input_mask_pairs=val_pairs,
            transform=False,
            tile_depth=tile_depth,
            patch_size=(patch_size, patch_size, patch_depth),
            num_patches_per_tile=num_patches_per_tile,
            max_overlap=max_overlap,
            norm=norm_mode,
        )

    # ---- Convert stride string to tuple if 3D U-Net with ResNet ----
    stride_tuple = None
    if is_3d and arch == 'U-Net' and backbone_name in [
        'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'
    ]:
        try:
            # ---- Convert string to tuple ----
            stride_tuple = eval(stride_str)
        except Exception as e:
            print(f"[Warning] Could not parse stride '{stride_str}': {e}")
            stride_tuple = None

    # ---- Model constructor ----
    model_kwargs = {
        'is_3d': is_3d,
        'model_arch': arch,
        'backbone_name': backbone_name,
        'channels': channels,
        'num_classes': num_classes,
        'weights': weights,
    }
    
    if stride_tuple is not None:
        model_kwargs['strides'] = stride_tuple
        
    model = model_factory(**model_kwargs)    

    # ---- Manually load custom pre-trained weights ----
    if weights_path is not None:
        checkpoint = torch.load(weights_path, map_location='cpu')

        if 'model_state' not in checkpoint:
            raise ValueError('Custom model is not compatible with BONe DLFit.')

        if fitting_mode == '3D':
            # ---- Check if this is a 2D state_dict by inspecting shape of any conv weight ----
            sample_key = next(
                (k for k in checkpoint['model_state'] if checkpoint['model_state'][k].ndim == 4),
                None
            )

            # ---- Convert 2D weights to 3D before applying to 3D model ----
            if sample_key is not None:
                from .convert_weights_mod import convert_2d_weights_to_3d
                if is_main_process:
                    ddp_log(
                        'Detected 2D weights. Converting to 3D weights, but results may vary.', 
                        log_path
                    )
                checkpoint['model_state'] = convert_2d_weights_to_3d(
                    checkpoint['model_state'], verbose=False
                )

        model.load_state_dict(checkpoint['model_state'])

    # ---- Setup device and wrap model with DDP ----
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{rank}')
        model = model.to(device)
        
        if is_distributed:
            # ---- Wrap model with DDP ----
            model = DDP(model, device_ids=[rank], output_device=rank)
            if is_main_process:
                ddp_log(f'Using {world_size} GPUs with DistributedDataParallel.', log_path)
        else:
            if is_main_process:
                ddp_log('Using single GPU without DDP.', log_path)
        
        torch.cuda.empty_cache()
    else:
        device = torch.device('cpu')
        if is_main_process:
            ddp_log('CUDA not available. Using CPU.', log_path)
        model = model.to(device)

    # ---- Print model architecture and backbone to log ----
    if is_main_process:
        ddp_log(f'Loaded {arch} model with {backbone_name} backbone.', log_path)

    # ---- Initialize loss function ----
    """
    tversky_alpha: increase value (e.g., 1.25) to penalize FPs more
    tversky_beta:  increase value (e.g., 1.25) to penalize FNs more
    gamma:         increase value (e.g., 2.0) to emphasize challenging features more
    focal_weight, tversky_weight: (0.5, 0.5) default; (0.7, 0.3) imbalanced classes
                                  (0.4, 0.6) small objects; (0.6, 0.4) boundaries
    """
    loss_registry = {
        'Jaccard': lambda: smp.losses.JaccardLoss(mode='multiclass'),
        'Dice': lambda: smp.losses.DiceLoss(mode='multiclass'),
        'Lovasz-Softmax': lambda: smp.losses.LovaszLoss(mode='multiclass'),
        'Focal': lambda: smp.losses.FocalLoss(mode='multiclass', gamma=criterion_gamma),
        'Tversky': lambda: smp.losses.TverskyLoss(
            mode='multiclass', alpha=criterion_alpha, beta=criterion_beta
        ),
        'Focal Tversky': lambda: FocalTverskyLoss(
            mode='multiclass',
            focal_weight=criterion_foc_wt,
            tversky_weight=criterion_tver_wt,
            gamma=criterion_gamma,
            tversky_alpha=criterion_alpha,
            tversky_beta=criterion_beta
        )
    }
    criterion = loss_registry[criterion_mode]()
    
    # ---- Initialize Jaccard Index metric ----
    from torchmetrics.classification import MulticlassJaccardIndex
    train_metric = MulticlassJaccardIndex(
        num_classes=num_classes, average=None, sync_on_compute=True
    ).to(device)
    val_metric = MulticlassJaccardIndex(
        num_classes=num_classes, average=None, sync_on_compute=True
    ).to(device)

    # ---- Print sample size to log ----
    if is_main_process:
        ddp_log(
            f'Size of training set: {len(train_dataset)} patches from '
            f'{int(len(train_dataset) / num_patches_per_tile)} '
            f'image-mask pairs', log_path
        )
        ddp_log(
            f'Size of validation set: {len(val_dataset)} patches from '
            f'{int(len(val_dataset) / num_patches_per_tile)} '
            f'image-mask pairs', log_path
        )

    # ---- Initialize optimizer ----
    optim_registry = {
        'Adam': lambda: torch.optim.Adam(model.parameters(), lr=learning_rate),
        'AdamW': lambda: torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=optim_weight_decay
        ),
        'SGD': lambda: torch.optim.SGD(
            model.parameters(), lr=learning_rate, momentum=optim_momentum, weight_decay=optim_weight_decay
        )
    }
    optimizer = optim_registry[optim_name]()

    # ---- Initialize the CosineAnnealingLR scheduler ----
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

    # ---- Create a GradScaler to scale gradients during mixed precision training ----
    scaler = GradScaler(device.type)

    # ---- Gradient accummulation (currently deactivated using steps=1) ----
    gradient_accumulation_steps = 1

    # ---- Defaults (tested on 2D, 2.5D, and 3D) ----
    # prefetch_factor = 2
    persistent_workers = True
    pin_memory = True

    # ---- Create DistributedSampler for DDP ----
    train_sampler = DistributedSampler(
        train_dataset, 
        num_replicas=world_size, 
        rank=rank,
        shuffle=True,
        seed=seed if seed is not None else 0
    ) if is_distributed else None
    
    val_sampler = DistributedSampler(
        val_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False
    ) if is_distributed else None

    # ---- Initialize data loaders ----
    loader_kwargs = {
        'batch_size': batch_size,
        'shuffle': (train_sampler is None),  # Shuffle if not using sampler
        'pin_memory': pin_memory,
        'num_workers': 0 if os.name == 'nt' else pt_workers,
        'sampler': None,  # Set per loader
    }

    # ---- Add Linux-only multiprocessing features to data loader ----
    if loader_kwargs['num_workers'] > 0:
        loader_kwargs.update({
            'worker_init_fn': worker_init_fn,
            'generator': generator,
            'prefetch_factor': prefetch_factor,
            'persistent_workers': persistent_workers,
            'multiprocessing_context': 'fork',
        })

    # ---- Assemble training and validation loaders ----
    train_loader = DataLoader(
        train_dataset, 
        **{**loader_kwargs, 'sampler': train_sampler, 'shuffle': False if train_sampler else True}
    )
    val_loader = DataLoader(
        val_dataset, 
        **{**loader_kwargs, 'sampler': val_sampler, 'shuffle': False}
    )

    # ---- Add variables for tracking the best score and total fitting time ----
    best_val_score = 0.0
    best_model_state = None
    best_epoch = -1
    total_fitting_time = 0.0
    temp_save_path = config['save_model_path'] + '_temp.pth' if is_main_process else None

    # ---- Create metadata dictionary for saved model ----
    metadata = {
        # ---- Experiment setup ----
        'train_dataset': [str(data[4]) for data in train_pairs],
        'val_dataset': [str(data[4]) for data in val_pairs],
        'num_classes': num_classes,
        'num_patches': len(train_dataset) + len(val_dataset),
        'patch_size': patch_size,
        'patch_depth': patch_depth,
        'max_overlap': max_overlap,
        'batch_size': batch_size,
        'augmentations': 'flips_rot90_bright_contrast' if augmentation else None,
        'normalization': norm_mode,
        'pt_workers': pt_workers,
        'prefetch_factor': prefetch_factor,
        'random_seed': seed,
        'version': None,

        # ---- Model architecture ----
        'is_3d': is_3d,
        'model_arch': arch,
        'backbone': backbone_name,
        'stride': stride_str if stride_tuple else None,
        'in_channels': channels,
        'mode': fitting_mode,

        # ---- Fitting configuration ----
        'optimizer': optim_name,
        'optim_momentum': optim_momentum,
        'optim_weight_decay': optim_weight_decay,
        'learning_rate': learning_rate,
        'criterion': criterion_mode,
        'criterion_alpha': criterion_alpha,
        'criterion_beta': criterion_beta,
        'criterion_gamma': criterion_gamma,
        'criterion_foc_wt': criterion_foc_wt,
        'criterion_tver_wt': criterion_tver_wt,
        'metric': 'JaccardIndex',

        # ---- Performance benchmarking ----
        'flops_G': None,
        'param_count': None,
        'peak_ram_GB': peak_ram_gb,
        'peak_vram_GB': None,
        'avg_gpu_util_percent': None,
        'wall_time_sec': None,
        'fitting_time_sec': None,
        'best_val_score': None,
        'best_epoch': None,
    }

    # ---------------------------------------------------------------------------------
    # Profile FLOPs and parameters before training starts
    # ---------------------------------------------------------------------------------
    if is_main_process:
        # ---- Unwrap DDP if needed for profiling ----
        model_for_flops = model.module if isinstance(model, DDP) else model
        model_for_flops = copy.deepcopy(model).cpu().eval()

        # ---- Get a single patch from the train_loader (batch size 1) ----
        sample_batch = next(iter(train_loader))
        sample_data = sample_batch[0][0:1].view(-1, *sample_batch[0][0:1].shape[2:]).cpu()

        with torch.no_grad():
            flops = FlopCountAnalysis(model_for_flops, sample_data)
            total_flops = flops.total()
            metadata['flops_G'] = round(total_flops / 1e9, 2)
            metadata['param_count'] = sum(p.numel() for p in model.parameters())

        # ---- Clean-up ----
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        del model_for_flops, sample_batch, sample_data
    # ---------------------------------------------------------------------------------
    # ---- Synchronize all processes before training starts ----
    if is_distributed:
        dist.barrier()

    # ---- Start GPU monitoring thread (only on main process) ----
    gpu_shared_dict = None
    gpu_monitor = None
    
    if is_main_process and torch.cuda.is_available():
        try:
            # ---- Create shared dictionary for GPU monitoring ----
            manager = _get_manager()
            gpu_shared_dict = manager.dict()
            gpu_shared_dict['util_samples'] = {}
            gpu_shared_dict['vram_samples'] = {}
            
            # ---- Initialize GPU monitor with shared dict ----
            gpu_ids = list(range(world_size)) if world_size > 1 else [0]
            gpu_monitor = GPUMonitor(
                interval=0.5,
                gpu_ids=gpu_ids, 
                shared_dict=gpu_shared_dict,
                rank=rank,
                world_size=world_size
            )

            if gpu_monitor:
                gpu_monitor.start()
                ddp_log('[INFO] GPU monitoring started')
        except Exception as e:
            ddp_log(f'[WARNING] Failed to start GPU monitoring: {e}')
            import traceback
            ddp_log(traceback.format_exc(), log_path)
            gpu_monitor = None

    # ---- Start fitting loop ----
    if is_main_process:
        ddp_log(f'Fitting {fitting_mode} model...')
        
    for epoch in range(num_epochs):
        # ---- Set epoch for DistributedSampler to ensure proper shuffling ----
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
            
        epoch_start_time = time.time()

        # ---- Training phase ----
        model.train()
        total_train_loss = 0.0
        optimizer.zero_grad(set_to_none=True)

        for step, (data, target) in enumerate(train_loader):
            if ddp_stop():
                if is_main_process:
                    ddp_log(f'[STOPPED] Exiting during training at epoch {epoch+1}', log_path)
                
                # ---- Clean-up before early exit ----
                del train_loader, val_loader
                gc.collect()
                if gpu_monitor:
                    gpu_monitor.stop()
                if is_distributed:
                    cleanup_ddp()
                return None, None, None, None  # Early exit

            data = data.view(-1, *data.shape[2:]).to(device, non_blocking=True)
            target = target.view(-1, *target.shape[2:]).to(device, non_blocking=True)

            is_accum_step = (
                (step + 1) % gradient_accumulation_steps == 0 or (step + 1) == len(train_loader)
            )
            
            with autocast(device.type):
                output = model(data)
                loss = criterion(output, target)
                loss = loss / gradient_accumulation_steps

            if is_distributed and not is_accum_step:
                with model.no_sync():
                    scaler.scale(loss).backward()
            else:
                scaler.scale(loss).backward()

            if is_accum_step:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            output = F.softmax(output, dim=1)
            preds = torch.argmax(output, dim=1)

            total_train_loss += loss.item() * gradient_accumulation_steps
            train_metric.update(preds, target)

        train_iou = train_metric.compute()

        # ---- Validation phase ----
        model.eval()
        total_val_loss = 0.0

        with torch.no_grad():
            for data, target in val_loader:
                if ddp_stop():
                    if is_main_process:
                        ddp_log(f'[STOPPED] Exiting validation at epoch {epoch+1}', log_path)
                    
                    # ---- Clean-up before early exit ----
                    del train_loader, val_loader
                    gc.collect()
                    if gpu_monitor:
                        gpu_monitor.stop()
                    if is_distributed:
                        cleanup_ddp()
                    return None, None, None, None  # Early exit

                data = data.view(-1, *data.shape[2:]).to(device, non_blocking=True)
                target = target.view(-1, *target.shape[2:]).to(device, non_blocking=True)

                output = model(data)
                loss = criterion(output, target)
                total_val_loss += loss.item()

                output = F.softmax(output, dim=1)

                preds = torch.argmax(output, dim=1)
                val_metric.update(preds, target)

        val_iou = val_metric.compute()
        
        # ---- Synchronize metrics across all processes if using DDP ----
        if is_distributed:
            # ---- Average losses across all processes ----
            total_train_loss_tensor = torch.tensor(total_train_loss).to(device)
            total_val_loss_tensor = torch.tensor(total_val_loss).to(device)
            dist.all_reduce(total_train_loss_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_val_loss_tensor, op=dist.ReduceOp.SUM)
            total_train_loss = (total_train_loss_tensor / world_size).item()
            total_val_loss = (total_val_loss_tensor / world_size).item()
        
        epoch_duration = time.time() - epoch_start_time
        total_fitting_time += epoch_duration

        # ---- Print out the epoch info (only on main process) ----
        if is_main_process:
            ddp_log(
                f'\n'
                f'Epoch [{epoch + 1}/{num_epochs}]\n'
                f'  {len(train_loader)} iterations - epoch duration: {epoch_duration:.2f} seconds',
                log_path
            )

            # ---- Display training loss ----
            ddp_log(f'  Training Loss: {total_train_loss / len(train_loader):.4f}', log_path)

            # ---- Display training metric ----
            train_score = train_iou.mean().item()
            ddp_log(f'  Training mean IoU: {train_score:.4f}', log_path)

            # ---- Display per-class training IoU ----
            for class_idx in range(config['num_classes']):
                ddp_log(
                    f'  Training Class {class_idx} IoU: {train_metric.compute()[class_idx].item():.4f}', 
                    log_path
                )

            # ---- Display validation loss ----
            ddp_log(f'  Validation Loss: {total_val_loss / len(val_loader):.4f}', log_path)

            # ---- Display validation metric ----
            val_score = val_iou.mean().item()
            ddp_log(f'  Validation mean IoU: {val_score:.4f}', log_path)

            # ---- Display per-class validation IoU ----
            for class_idx in range(num_classes):
                ddp_log(
                    f'  Validation Class {class_idx} IoU: {val_metric.compute()[class_idx].item():.4f}', 
                    log_path
                )

        # ---- Compute validation score for model saving ----
        val_score = val_iou.mean().item()
        
        # ---- Save model weights if validation metric improved (only on main process) ----
        if is_main_process and val_score > best_val_score:
            best_val_score = val_score
            # ---- Unwrap DDP model to get the actual model state dict ----
            model_to_save = model.module if isinstance(model, DDP) else model
            best_model_state_dict = model.state_dict()  # Save the model weights
            best_epoch = epoch + 1
            metadata['best_epoch'] = best_epoch
            metadata['best_val_score'] = round(best_val_score, 4)

            ddp_log(
                f'A better model was found at epoch {epoch+1} with Jaccard score {best_val_score:.4f}', 
                log_path
            )

            # ---- Save weights to a temp file ---- 
            torch.save({'model_state': best_model_state_dict,'metadata': metadata}, temp_save_path)

        # ---- Reset metric for the next epoch ----
        train_metric.reset()
        val_metric.reset()

        # ---- Update the learning rate at the end of the epoch ----
        scheduler.step()

        # ---- Clear cache at end of epoch ----
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # ---- Update progress bar (only on main process) ----
        ddp_progress(epoch + 1)

    # ---- Record fitting time in metadata ----
    metadata['fitting_time_sec'] = round(total_fitting_time)

    # ---- Stop GPU monitoring and collect results (only on main process) ----
    if is_main_process:
        if gpu_monitor:
            try:
                ddp_log('[INFO] Stopping GPU monitor...')
                gpu_monitor.stop()
                gpu_stats = gpu_monitor.get_results()

                # ---- Update metadata with performance stats ----
                if gpu_stats:
                    metadata['peak_vram_GB'] = gpu_stats.get('peak_vram_GB', {})
                    metadata['avg_gpu_util_percent'] = gpu_stats.get('avg_util_percent', {})
        
                    # ---- Log GPU Monitor errors ----
                    if gpu_stats.get('errors'):
                        ddp_log(
                            f"[GPUMonitor] {len(gpu_stats['errors'])} monitoring errors occurred.",
                            log_path
                        )
                        for gpu_id, msg, ts in gpu_stats['errors']:
                            ddp_log(
                                f"[GPUMonitor] GPU {gpu_id} error at {time.ctime(ts)}: {msg}",
                                log_path
                            )
                else:
                    ddp_log('[WARNING] No GPU stats collected')
            except Exception as e:
                ddp_log(f'[WARNING] Error collecting GPU stats: {e}')
                import traceback
                ddp_log(traceback.format_exc(), log_path)

        if best_model_state_dict is not None:
            torch.save(
                {'model_state': best_model_state_dict, 'metadata': metadata}, 
                temp_save_path
            )

    # ---- Synchronize all processes before cleanup ----
    if is_distributed:
        dist.barrier()

    # ---- Remove 'module.' prefix from state_dict if exists ----
    if is_main_process:
        best_model_state_dict = remove_module_prefix(best_model_state_dict)
    
    # ---- Final dataset and memmap clean-up ----
    del train_loader, val_loader, train_dataset, val_dataset, model
    gc.collect()

    # ---- Cleanup DDP ----
    if is_distributed:
        cleanup_ddp()
    
    # ---- Return values to main script (only main process has valid data) ----
    if is_main_process:
        return best_model_state_dict, metadata, best_epoch, temp_save_path
    else:
        return None, None, None, None

# ---------------------------------------------------------------------------------
# Wrapper to enable DDP multiprocessing
# ---------------------------------------------------------------------------------
def run_training_pipeline():
    config = load_config()
    temp_save_path = config['save_model_path'] + '_temp.pth'
    log_path = Path(config.get('log_path'))
    console_queue = get_console_queue()
    progress_queue = get_progress_queue()

    # ---- Create manager at parent process level ----
    manager = _get_manager()
    gpu_shared_dict = manager.dict()
    gpu_shared_dict['util_samples'] = {}
    gpu_shared_dict['vram_samples'] = {}

    try:
        # ---- Determine world size based on gpu_mode and available GPUs ----
        gpu_mode = config['gpu_mode']
        
        if torch.cuda.is_available():
            if gpu_mode == 0:
                # ---- Single GPU mode ----
                world_size = 1
                result = training_pipeline(
                    rank=0,
                    world_size=world_size,
                    config=config,
                    stop_event=stop_event,
                    console_queue=console_queue,
                    progress_queue=progress_queue,
                    gpu_shared_dict=gpu_shared_dict,
                )
                if result is not None and len(result) == 4:
                    best_model_state_dict, metadata, best_epoch, temp_save_path = result
                else:
                    log_to_console('[ERROR] Training pipeline returned invalid result', log_path)
                    return None, None, None, None
                    
            elif gpu_mode == 1:
                # ---- Multi-GPU mode with DDP ----
                world_size = torch.cuda.device_count()
                
                if world_size > 1:
                    # ---- Use multiprocessing to spawn processes for each GPU ----
                    mp.spawn(
                        training_pipeline,
                        args=(
                            world_size, 
                            config, 
                            stop_event, 
                            console_queue, 
                            progress_queue, 
                            gpu_shared_dict,
                        ),
                        nprocs=world_size,
                        join=True
                    )
                    
                    # ---- Load results from temp file (saved by rank 0) ----
                    temp_save_path = config['save_model_path'] + '_temp.pth'
                    if os.path.exists(temp_save_path):
                        try:
                            checkpoint = torch.load(temp_save_path, map_location='cpu')
                            best_model_state_dict = checkpoint['model_state']
                            metadata = checkpoint['metadata']
                            best_epoch = metadata.get('best_epoch') if metadata else None
                            
                            if best_model_state_dict is None or metadata is None:
                                log_to_console('[ERROR] Incomplete checkpoint in temp file', log_path)
                                return None, None, None, None
                        except Exception as e:
                            log_to_console(f'[ERROR] Failed to load temp checkpoint: {e}', log_path)
                            return None, None, None, None
                    else:
                        log_to_console('[ERROR] Temp checkpoint file not found', log_path)
                        return None, None, None, None
                else:
                    # ---- Only 1 GPU available, fall back to single GPU mode ----
                    world_size = 1
                    result = training_pipeline(
                        rank=0,
                        world_size=world_size,
                        config=config,
                        stop_event=stop_event,
                        console_queue=console_queue,
                        progress_queue=progress_queue,
                    )
                    
                    if result is not None and len(result) == 4:
                        best_model_state_dict, metadata, best_epoch, temp_save_path = result
                    else:
                        log_to_console('[ERROR] Training pipeline returned invalid result', log_path)
                        return None, None, None, None
            else:
                raise ValueError(f'Invalid gpu_mode: {gpu_mode}')
        else:
            # ---- CPU mode ----
            world_size = 1
            result = training_pipeline(
                rank=0,
                world_size=world_size,
                config=config,
                stop_event=stop_event,
                console_queue=console_queue,
                progress_queue=progress_queue,
            )
            
            if result is not None and len(result) == 4:
                best_model_state_dict, metadata, best_epoch, temp_save_path = result
            else:
                log_to_console('[ERROR] Training pipeline returned invalid result', log_path)
                return None, None, None, None

        return best_model_state_dict, metadata, best_epoch, temp_save_path

    except Exception as e:
        log_to_console(f'[ERROR] Training failed: {e}')
        import traceback
        log_to_console(traceback.format_exc(), log_path)
        return None, None, None, None

    finally:
        # ---- CUDA clean-up ----
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # ---- Python GC ----
        gc.collect()
        
        # ---- Delete temp weights file and clean up temp npz files ---- 
        if temp_save_path and os.path.exists(temp_save_path):
            try:
                os.remove(temp_save_path)
            except Exception as e:
                log_to_console(f'Failed to delete temp weights file: {e}')
        
        for npy_file in CONFIG_DIR.glob('*.npy'):
            try:
                npy_file.unlink()
            except Exception as e:
                log_to_console(f'Could not delete {npy_file.name}: {e}')

if __name__ == '__main__':
    run_training_pipeline()
