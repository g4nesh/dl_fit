# BONe_DLFit/core/trainer.py
import ctypes
import gc
import json
import numpy as np
import os
import random
import threading
import time
import tkinter as tk
import torch
import BONe_DLFit.app.widgets.model_widgets as model_widgets
from BONe_DLFit.app.widgets.base import update_fit_button_state
from pathlib import Path
from shutil import copyfile
from tkinter import filedialog
from BONe_utils.utils import (
    log_to_console, 
    has_multiple_gpus, 
    set_global_seed, 
    reset_global_seed,
    load_config,
    set_config_file,
    save_config,
    CONFIG_FILE,
    CONFIG_DIR,
    initialize_queues,
    get_console_queue,
    get_progress_queue,
)
from .volume_loader import load_volumes_and_stats_streaming
from .fitting_loop import run_training_pipeline
from .fitting_loop import stop_event

# ---------------------------------------------------------------------------------
# Navigate to path of data folders
# ---------------------------------------------------------------------------------
DATA_FOLDERS = Path(__file__).resolve().parent.parent.parent

class Trainer:
    __version__ = '2.0.0.standalone'

    def __init__(self, app):
        self.app = app
        self.thread = None
        self.app.root.after(0, lambda: self.app.fit_button.config(state='disabled'))
        self.app.root.after(0, lambda: self.app.stop_button.config(state='disabled'))

    # ---- Ensure dynamic input-mask widgets are created ---- 
    def _load_dynamic_pairs(self, input_dirs, mask_dirs):
        if hasattr(self.app, 'additional_data_frames') and self.app.additional_data_frames:
            for i, (input_path, mask_path) in enumerate(zip(input_dirs, mask_dirs), start=1):
                input_entry = getattr(self.app, f'input_entry_{i}', None)
                mask_entry = getattr(self.app, f'mask_entry_{i}', None)
                if input_entry and mask_entry:
                    input_entry.insert(0, input_path)
                    mask_entry.insert(0, mask_path)

    def _load_custom_weights_from_config(self, config):
        self.app.weights_var.set(2)
        model_widgets.load_custom_weights_section(self.app)

        if hasattr(self.app, 'cust_weights_entry'):
            self.app.cust_weights_path.set(config['load_weights_path'])

    def load_config_from_file(self):
        path = filedialog.askopenfilename(
            title='Load custom config',
            filetypes=[('JSON files', '*.json')],
        )
        if not path:
            return

        path = Path(path)

        # ---- Point CONFIG_FILE to the default working location ----
        from BONe_utils.utils import CONFIG_FILE

        CONFIG_FILE.parent.mkdir(exist_ok=True)
        copyfile(path, CONFIG_FILE)

        # ---- Reload config and update UI ----
        config = load_config()
        self.config = config
        self.load_previous_config(config)
        log_to_console(f'Loaded config from {path.name}')

    def collect_config_from_ui(self) -> dict:
        config = {}

        # ---- Input/mask directories ----
        input_dirs = [self.app.input_entry.get()]
        mask_dirs = [self.app.mask_entry.get()]
        if hasattr(self.app, 'num_pairs_var') and hasattr(self.app, 'additional_data_frames'):
            for i in range(1, self.app.num_pairs_var.get() + 1):
                input_entry = getattr(self.app, f'input_entry_{i}', None)
                mask_entry = getattr(self.app, f'mask_entry_{i}', None)
                if input_entry and mask_entry:
                    input_dirs.append(input_entry.get())
                    mask_dirs.append(mask_entry.get())

        config['input_dirs'] = input_dirs
        config['mask_dirs'] = mask_dirs
        config['num_input_dirs'] = (
            self.app.num_pairs_var.get() + 1
            if hasattr(self.app, 'num_pairs_var')
            else 1
        )

        # ---- Basic settings ----
        fit_mode = self.app.fitting_mode_var.get()
        arch = self.app.model_arch_var.get()
        is_resnet = self.app.backbone_var.get() in [
                'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'
        ]
        config.update({
            'train_val_split': self.app.train_val_split_var.get(),
            'repro_mode': bool(self.app.repro_mode_var.get()),
            'gpu_mode': self.app.gpu_mode_var.get(),
            'patch_size': self.app.patch_var.get(),
            'patch_depth': None if fit_mode in (0, 1) else self.app.patch_depth_var.get(),
            'tile_depth': None if fit_mode in (0, 1) else self.app.tile_depth_var.get(),
            'num_patches_per_tile': self.app.patch_per_tile_var.get(),
            'max_overlap': self.app.patch_overlap_var.get(),
            'batch_size': self.app.batch_size_var.get(),
            'augmentation': bool(self.app.augment_var.get()),
            'norm_mode': 'ZScore' if self.app.norm_mode_var.get() == 0 else 'minMax',
            'fitting_mode': '2D' if fit_mode == 0 else '2.5D' if fit_mode == 1 else '3D',
            'in_channels': self.app.tiles2channels_var.get() if fit_mode == 1 else 1,
            'is_3d': fit_mode == 2,
            'model_arch': arch,
            'menu_index': (
                self.app.model_arch_vals_2d.index(arch)
                if fit_mode in (0,1)
                else self.app.model_arch_vals_3d.index(arch)
            ),
            'backbone': self.app.backbone_var.get(),
            'stride': (
                self.app.stride_var.get()
                if fit_mode == 2 and arch == 'U-Net' and is_resnet
                else None
            ),
            'num_epochs': self.app.num_epochs_var.get(),
            'model_filename': str(self.app.model_name_var.get()),
            'save_model_path': str(
                Path(self.app.save_entry.get()) / self.app.model_name_var.get()
            ),
        })

        # ---- Loss function settings ----
        loss_name = self.app.loss_var.get() or 'Jaccard'
        config['loss_fxn'] = loss_name
        config['criterion_alpha'] = (
            float(self.app.tver_alpha_var.get())
            if loss_name in ('Tversky', 'Focal Tversky')
            else None
        )
        config['criterion_beta'] = (
            float(self.app.tver_beta_var.get())
            if loss_name in ('Tversky', 'Focal Tversky')
            else None
        )
        config['criterion_gamma'] = (
            float(self.app.foc_gamma_var.get())
            if loss_name in ('Focal', 'Focal Tversky')
            else None
        )
        config['criterion_foc_wt'] = (
            float(self.app.foc_tver_var.get())
            if loss_name == 'Focal Tversky'
            else None
        )
        config['criterion_tver_wt'] = (
            1 - float(self.app.foc_tver_var.get())
            if loss_name == 'Focal Tversky'
            else None
        )

        # ---- Optimizer settings ----
        optimizer = self.app.optim_var.get()
        config['optimizer'] = optimizer
        config['optim_weight_decay'] = (
            float(self.app.weight_decay_var.get())
            if optimizer in ('AdamW', 'SGD') and self.app.weight_decay_var.get() != ''
            else None
        )
        config['optim_momentum'] = (
            float(self.app.momentum_var.get())
            if optimizer == 'SGD' and self.app.momentum_var.get() != ''
            else None
        )
        config['learn_rate'] = self.app.learning_rate_var.get()

        # ---- Workers, prefetch, and weights ----
        config['pt_workers'] = self.app.pt_workers_var.get()
        config['prefetch_factor'] = self.app.prefetch_factor_var.get()
        weights_var = self.app.weights_var.get()
        config['load_weights'] = 'imagenet' if weights_var == 1 else None
        config['load_weights_path'] = (
            str(self.app.cust_weights_entry.get()) if weights_var == 2 else None
        )

        # ---- Seed ----
        config['seed'] = (
            int(self.app.random_seed_var.get())
            if self.app.repro_mode_var.get() == 1
            else None
        )

        return config
        
    def save_config_to_file(self):
        config = self.collect_config_from_ui()
        self.config = config

        # ---- Prompt user to save archival copy of config ----
        path = filedialog.asksaveasfilename(
            title='Save custom config as',
            defaultextension='.json',
            filetypes=[('JSON files', '*.json')],
        )
        if not path:
            return

        Path(path).write_text(json.dumps(self.config, indent=4))
        save_config(config)

        log_to_console(f'Configuration saved to {Path(path).name}')

    def load_previous_config(self, config):
        self.config = config

        # ---- Dictionary mappings for simple previous settings ---- 
        simple_vars = {
            'patch_size': self.app.patch_var,
            'num_patches_per_tile': self.app.patch_per_tile_var,
            'max_overlap': self.app.patch_overlap_var,
            'train_val_split': self.app.train_val_split_var,
            'model_arch': self.app.model_arch_var,
            'backbone': self.app.backbone_var,
            'stride': self.app.stride_var,
            'num_epochs': self.app.num_epochs_var,
            'loss_fxn': self.app.loss_var,
            'optimizer': self.app.optim_var,
            'learn_rate': self.app.learning_rate_var,
            'batch_size': self.app.batch_size_var,
            'prefetch_factor': self.app.prefetch_factor_var,
            'model_filename': self.app.model_name_var,
        }

        for key, var in simple_vars.items():
            value = config.get(key)
            if value is not None:
                var.set(value)

        # ---- Previous input1 and mask1 ---- 
        input_dirs = config.get('input_dirs', [])
        mask_dirs = config.get('mask_dirs', [])

        if input_dirs and mask_dirs:
            self.app.input_entry.delete(0, 'end')
            self.app.mask_entry.delete(0, 'end')
            self.app.input_entry.insert(0, input_dirs[0])
            self.app.mask_entry.insert(0, mask_dirs[0])

        num_additional = len(input_dirs) - 1
        if num_additional > 0:
            self.app.num_pairs_var.set(num_additional)

            # Triggers widget builder for additional input-mask pairs
            self.app.update_add_pairs()
            self.app.root.after(50, self._load_dynamic_pairs, input_dirs[1:], mask_dirs[1:])

        # ---- Previous reproducibility mode ---- 
        repro_mode = config.get('repro_mode')
        if repro_mode is not None:
            if repro_mode == 'false':
                self.app.repro_mode_var.set(0)
            else:
                self.app.repro_mode_var.set(1)
                self.app.frame_dynamic_seed.grid()
                self.app.random_seed_var.set(config.get('seed', 42))

        # ---- Previous gpu mode ----
        gpu_mode = config.get('gpu_mode', 0)
        if (gpu_mode > 0) and (has_multiple_gpus()):
            self.app.gpu_mode_var.set(1)
        else:
            self.app.gpu_mode_var.set(0)

        # ---- Previous fitting mode ---- 
        fitting_mode_map = {'3D': 2, '2.5D': 1, '2D': 0}
        self.app.fitting_mode_var.set(fitting_mode_map.get(config.get('fitting_mode', '2D'), 0))
        self.app.toggle_fitting_mode_fields()
        mode = config.get('fitting_mode', '2D')

        if mode == '2.5D':
            self.app.frame_dynamic_tile2channels.grid()
            self.app.tiles2channels_var.set(config.get('in_channels'))

        if mode == '3D':
            self.app.frame_dynamic_tile_depth.grid()
            self.app.tile_depth_var.set(config.get('tile_depth'))
            self.app.frame_dynamic_patch_depth.grid()
            self.app.patch_depth_var.set(config.get('patch_depth'))
            
        # ---- Previous stride ----
        stride_val = config.get('stride')
        if stride_val is not None:
            is_3d = self.app.fitting_mode_var.get() == 2
            is_unet = self.app.model_arch_var.get() == 'U-Net'
            is_resnet = self.app.backbone_var.get() in [
                'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'
            ]

            if is_3d and is_unet and is_resnet:
                self.app.stride_var.set(stride_val)
                self.app.frame_stride.grid()
            else:
                self.app.frame_stride.grid_remove()

        # ---- Previous augmentation mode ---- 
        augmentation = config.get('augmentation', True)
        self.app.augment_var.set(1 if augmentation else 0)

        # ---- Previous normalization mode ---- 
        self.app.norm_mode_var.set(1 if config.get('norm_mode', '').lower() == 'minmax' else 0)

        # ---- Previous weights (random init., imagenet, or custom weights) ---- 
        weights = config.get('load_weights')
        if weights == 'imagenet':
            self.app.weights_var.set(1)
        elif config.get('load_weights_path'):
            self.app.weights_var.set(2)
            self.app.root.after(50, lambda: self._load_custom_weights_from_config(config))
        else:
            self.app.weights_var.set(0)

        # ---- Previous optimizer settings ----
        optimizer = config.get('optimizer') or 'Adam'
        self.app.optim_var.set(optimizer)
        
        if optimizer in ('AdamW', 'SGD'):
            if config.get('optim_weight_decay') is not None:
                self.app.weight_decay_var.set(str(config['optim_weight_decay']))
            if optimizer == 'SGD' and config.get('optim_momentum') is not None:    
                self.app.momentum_var.set(str(config['optim_momentum']))

        # ---- Sync UI state ----
        self.app.update_optimizer_settings()

        # ---- Previous loss function settings ----
        loss_fxn_name = config.get('loss_fxn') or 'Jaccard'
        self.app.loss_var.set(loss_fxn_name)
        
        loss_config_to_var = {
            'criterion_gamma': self.app.foc_gamma_var,
            'criterion_alpha': self.app.tver_alpha_var,
            'criterion_beta': self.app.tver_beta_var,
            'criterion_foc_wt': self.app.foc_tver_var,
        }
            
        for cfg_key, var in loss_config_to_var.items():
            val = config.get(cfg_key)
            if val is not None:
                var.set(str(val))

        # ---- Sync UI state ----
        self.app.update_loss_settings()

        # ---- Previous DataLoader workers ----
        pt_workers = config.get('pt_workers', 4)
        if (pt_workers > 0) and (os.name == 'nt'):
            self.app.pt_workers_var.set(0)
        else:
            self.app.pt_workers_var.set(pt_workers)

        # ---- Previous Prefetch factor ----
        pf_factor = config.get('prefetch_factor', 2)
        if (pf_factor > 0) and (os.name == 'nt'):
            self.app.prefetch_factor_var.set(0)
        else:
            self.app.prefetch_factor_var.set(pf_factor)

        # ---- Previous save folder ---- 
        save_path = config.get('save_model_path')
        if save_path:
            self.app.save_entry.insert(0, os.path.dirname(save_path))

        # ---- Check if all required entry fields are valid and activate Fit button ----
        self.app.root.after(100, lambda: update_fit_button_state(self.app))

    def browse_input(self, on_update=None):
        path = filedialog.askdirectory(
            initialdir=str(DATA_FOLDERS),
            title=f'Select Input Folder'
        )
        if path:
            self.app.input_entry.delete(0, 'end')
            self.app.input_entry.insert(0, path)
            self.config['input_dirs'] = [path]
            save_config(self.config)

            if on_update:
                on_update(self)

    def browse_mask(self, on_update=None):
        path = filedialog.askdirectory(
            initialdir=str(DATA_FOLDERS),
            title=f'Select Mask Folder'
        )
        if path:
            self.app.mask_entry.delete(0, 'end')
            self.app.mask_entry.insert(0, path)
            self.config['mask_dirs'] = [path]
            save_config(self.config)

            if on_update:
                on_update(self)

    def browse_save_folder(self, on_update=None):
        path = filedialog.askdirectory(
            initialdir=str(DATA_FOLDERS), 
            title=f'Select Save Folder'
        )
        if path:
            self.app.save_entry.delete(0, 'end')
            self.app.save_entry.insert(0, path)

            if on_update:
                on_update(self)

    def browse_input_dynamic(self, entry, on_update=None):
        path = filedialog.askdirectory(
            initialdir=str(DATA_FOLDERS),
            title=f'Select Input Folder'
        )
        if path:
            entry.delete(0, tk.END)
            entry.insert(0, path)

            if on_update:
                on_update(self)

    def browse_mask_dynamic(self, entry, on_update=None):
        path = filedialog.askdirectory(
            initialdir=str(DATA_FOLDERS),
            title=f'Select Mask Folder'
        )
        if path:
            entry.delete(0, tk.END)
            entry.insert(0, path)

            if on_update:
                on_update(self)

    def browse_pth_dynamic(self, entry):
        path = filedialog.askopenfilename(
            initialdir=str(DATA_FOLDERS),
            title='Select .pth File',
            filetypes=[('PyTorch Model Files', '*.pth'), ('All Files', '*.*')]
        )
        if path:
            entry.delete(0, tk.END)
            entry.insert(0, path)

    # ---- Aggressive clean-up ----
    def aggressive_memory_cleanup(self):
        log_to_console('[INFO] Starting aggressive memory cleanup...')
        
        # Clear large data from config
        if hasattr(self, 'config'):
            if 'train_pairs' in self.config:
                self.config['train_pairs'] = []
            if 'val_pairs' in self.config:
                self.config['val_pairs'] = []
        
        # Force garbage collection three times
        for _ in range(3):
            gc.collect()
            
        # Clear PyTorch cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            for obj in gc.get_objects():
                try:
                    if torch.is_tensor(obj):
                        del obj
                except:
                    pass
            gc.collect()
            torch.cuda.empty_cache()
        
        # Force glibc to release memory back to the OS (Linux-specific)
        try:
            libc = ctypes.CDLL('libc.so.6')
            libc.malloc_trim(0)
            log_to_console('[INFO] Called malloc_trim to release memory to OS')
        except Exception as e:
            log_to_console(f'[WARN] Could not call malloc_trim: {e}')
        log_to_console(f'[INFO] Cleanup complete\n') 

    def start(self, on_complete=None):
        self.on_complete = on_complete
        stop_event.clear()

        # ---------------------------------------------------------------------------------
        # Initialize multiprocessing queues BEFORE any logging
        # ---------------------------------------------------------------------------------
        try:
            initialize_queues()
            console_queue = get_console_queue()
            progress_queue = get_progress_queue()
        except Exception as e:
            print(f'[ERROR] Failed to initialize queues: {e}')
            return

        # ---------------------------------------------------------------------------------
        # Define log file path based on model path
        # ---------------------------------------------------------------------------------
        self.log_file_txt = (
            str(Path(self.app.save_entry.get()) / f'{self.app.model_name_var.get()}_log.txt')
        )
        os.makedirs(os.path.dirname(self.log_file_txt), exist_ok=True)

        # ---------------------------------------------------------------------------------
        # Gather static input/mask entries
        # ---------------------------------------------------------------------------------
        if not self.app.input_entry.get() or not self.app.mask_entry.get():
            log_to_console('[ERROR] Missing Input1 or Mask1.', self.log_file_txt)
            return
        else:
            input_dirs = [self.app.input_entry.get()]
            mask_dirs = [self.app.mask_entry.get()]

        # Gather dynamically added input/mask entries
        if hasattr(self.app, 'num_pairs_var') and hasattr(self.app, 'additional_data_frames'):
            for i in range(1, self.app.num_pairs_var.get() + 1):
                input_entry = getattr(self.app, f'input_entry_{i}', None)
                mask_entry = getattr(self.app, f'mask_entry_{i}', None)
                if input_entry and mask_entry:
                    input_dirs.append(input_entry.get())
                    mask_dirs.append(mask_entry.get())

        else:
            log_to_console(
                '[ERROR] Attach at least 1 additional input and mask pair '
                'for scan-level train/val split.', self.log_file_txt
            )
            return

        # ---------------------------------------------------------------------------------
        # Check for Reproducibility Mode
        # ---------------------------------------------------------------------------------
        if self.app.repro_mode_var.get() == 0:
            seed_val = None
            log_to_console('Reproducibility mode is OFF.', self.log_file_txt)
            reset_global_seed()
        elif self.app.repro_mode_var.get() == 1:
            try:
                seed_val = int(self.app.random_seed_var.get())
                log_to_console(f'Reproducibility mode is ON (seed = {seed_val}).', self.log_file_txt)
                _ = set_global_seed(seed_val)
            except ValueError:
                log_to_console('[Error] Random seed must be an integer.', self.log_file_txt)
                return
        
        # ---------------------------------------------------------------------------------
        # Save to config
        # ---------------------------------------------------------------------------------
        self.config = self.collect_config_from_ui()
        save_config(self.config)

        # ---------------------------------------------------------------------------------
        # Start training thread
        # ---------------------------------------------------------------------------------
        self.thread = threading.Thread(target=self._run)
        self.thread.start()

    def _run(self):
        # ---------------------------------------------------------------------------------
        # Start wall time counter and initialize progressbar
        # ---------------------------------------------------------------------------------
        try:
            start_wall = time.perf_counter()
            self.app.progress['maximum'] = self.config['num_epochs']
            self.app.progress['value'] = 0

            # -----------------------------------------------------------------------------
            # Load volumes and calculate stats (parallel process 2 scans at a time)
            # -----------------------------------------------------------------------------
            gc.collect()

            input_mask_pairs, peak_ram, num_classes = load_volumes_and_stats_streaming(
                input_dirs=self.config['input_dirs'],
                mask_dirs=self.config['mask_dirs'],
                norm_mode=self.config['norm_mode'],
                patch_size=self.config['patch_size'],
                patch_depth=self.config.get('patch_depth'),
                tile_depth=self.config.get('tile_depth'),
                logger=log_to_console,
                npy_dir=CONFIG_DIR,
                max_parallel=2
            )

            if num_classes is not None:
                self.config['num_classes'] = num_classes
                save_config(self.config)

            # -----------------------------------------------------------------------------
            # Shuffle and split dataset at scan-level and save config as json
            # -----------------------------------------------------------------------------
            random.shuffle(input_mask_pairs)
            val_ratio = 1 - self.config['train_val_split']
            num_val = max(1, int(len(input_mask_pairs) * val_ratio))

            val_pairs = input_mask_pairs[:num_val]
            train_pairs = input_mask_pairs[num_val:]

            # -----------------------------------------------------------------------------
            # Save config as json
            # -----------------------------------------------------------------------------
            self.config['train_pairs'] = train_pairs
            self.config['val_pairs'] = val_pairs
            self.config['peak_ram'] = peak_ram
            self.config['log_path'] = str(self.log_file_txt)
            save_config(self.config)
            gc.collect()

            # -----------------------------------------------------------------------------
            # Enable Stop button
            # -----------------------------------------------------------------------------
            self.app.root.after(0, lambda: self.app.stop_button.config(state='normal'))

            # ------------------------
            # Begin model fitting loop
            # ------------------------
            # ---- Fitting loop ---- 
            (
                best_model_state_dict, 
                metadata, 
                best_epoch, 
                temp_save_path
            ) = run_training_pipeline()
        
            # ---- Check if stopped or incomplete ----
            if stop_event.is_set() or metadata is None:
                log_to_console('[INFO] Training stopped or incomplete', self.log_file_txt)
                return

            # -------------------------------------------------------
            # Save best model only if training completed successfully
            # -------------------------------------------------------
            if best_model_state_dict is not None and metadata is not None:
                wall_time = time.perf_counter() - start_wall
                metadata['wall_time_sec'] = round(wall_time)
                metadata['version'] = self.__version__

                fitting_time = metadata.get('fitting_time_sec', 0)
                log_to_console(
                    f'Model fitting completed after {round(fitting_time)} seconds.', 
                    self.log_file_txt
                )

                # Save the best model
                best_val_score = metadata.get('best_val_score')
                if best_val_score is not None and best_epoch is not None:
                    suffix = f'_best_epoch_{best_epoch}_val_jaccard_{best_val_score:.4f}.pth'
                    save_path = Path(self.config['save_model_path'] + suffix)

                    torch.save({'model_state': best_model_state_dict,'metadata': metadata}, save_path)
                    log_to_console(
                        f'Best model found at epoch {best_epoch} with validation Jaccard score '
                        f'{best_val_score:.4f}', self.log_file_txt
                    )
                    log_to_console(json.dumps(metadata, indent=4), self.log_file_txt)
                    log_to_console(f'Saved to {save_path}', self.log_file_txt)
                else:
                    log_to_console(
                        '[WARNING] Training completed but but best_val_score or best_epoch is None',
                        self.log_file_txt
                    )
            else:
                log_to_console(
                    '[ERROR] Training failed: model_state or metadata is None',
                    self.log_file_txt
                )
        except Exception as e:
            log_to_console(f'[ERROR] Training failed: {e}', self.log_file_txt)

        finally:
            self._cleanup_after_stop()

    # ---------------------------------------------------------------------------------
    # Cleanup
    # ---------------------------------------------------------------------------------
    def _cleanup_after_stop(self):
        log_to_console('[INFO] Starting cleanup...')
        # Reset progress bar
        self.app.progress['value'] = 0
        
        # Disable Stop button
        self.app.root.after(0, lambda: self.app.stop_button.config(state='disabled'))
        
        # Clear CUDA cache
        torch.cuda.empty_cache()
        
        # Aggressive memory cleanup
        self.aggressive_memory_cleanup()
        gc.collect()

        # Re-enable Fit Model button
        if self.on_complete:
            self.app.root.after(0, self.on_complete)

    # ---------------------------------------------------------------------------------
    # Stop button functionality
    # ---------------------------------------------------------------------------------
    def stop(self):
        stop_event.set()
        log_to_console('[INFO] Stop requested...')
