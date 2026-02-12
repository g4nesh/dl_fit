# BONe_DLFit/app/widgets/button_widgets.py
import os
import tkinter as tk
from tkinter import ttk
from BONe_utils.utils import clear_console
from .base import (
    make_styled_entry,
    make_slider_with_entry,
    build_file_input_row
)

# ------------------------
# Helper functions
# ------------------------
# ---- Reset console ----
def reset_to_defaults(app):
    # ---- Clear original input and mask entries ---- 
    if hasattr(app, 'input_entry'):
        app.input_entry.delete(0, 'end')
    if hasattr(app, 'mask_entry'):
        app.mask_entry.delete(0, 'end')
    if hasattr(app, 'save_entry'):
        app.save_entry.delete(0, 'end')

    # ---- Reset scalar variables to defaults ---- 
    app.gpu_mode_var.set(0)
    app.fitting_mode_var.set(0)
    app.tiles2channels_var.set(3)
    app.tile_depth_var.set(32)
    app.patch_depth_var.set(32)
    app.patch_var.set(256)
    app.patch_per_tile_var.set(4)
    app.patch_overlap_var.set(0.3)
    app.augment_var.set(1)
    app.norm_mode_var.set(0)
    app.train_val_split_var.set(0.8)
    app.model_arch_var.set('U-Net')
    app.backbone_var.set('resnet18')
    app.num_epochs_var.set(25)
    app.learning_rate_var.set(0.001)
    app.batch_size_var.set(32)
    app.loss_var.set(0)
    app.pt_workers_var.set(0 if os.name =='nt' else 2)
    app.model_name_var.set('')
    
    # ---- Reset Reproducibility Mode to OFF ---- 
    app.repro_mode_var.set(0)
    app.frame_dynamic_seed.grid_remove()

    # ---- Reset patch size entry display ---- 
    app.patch_var.set(256)
    app.patch_size_entry.delete(0, 'end')
    app.patch_size_entry.insert(0, '256')

    # ---- Reset 2.5D and 3D widgets ---- 
    app.frame_dynamic_tile2channels.grid_remove()
    app.frame_dynamic_tile_depth.grid_remove()
    app.frame_dynamic_patch_depth.grid_remove()

    # ---- Reset Weights ---- 
    app.weights_var.set(0)
    app.frame_dynamic_cust_weights.grid_remove()

    # ---- Reset additional input-mask pairs ---- 
    app.num_pairs_var.set(0)
    for frame in getattr(app, 'additional_data_frames', []):
        frame.grid_remove()
    app.additional_data_frames.clear()
    app.frame_dynamic_inputs.grid_remove()

    # ---- Clear any stored dynamic entry attributes ---- 
    for attr in dir(app):
        if attr.startswith('input_entry_') or attr.startswith('mask_entry_'):
            setattr(app, attr, None)

    # ---- Reset optimizer to Adam ---- 
    app.optim_var.set('Adam')
    app.frame_dynamic_optim_settings.grid_remove()

    # ---- Reset loss function to Jaccard ---- 
    app.loss_var.set('Jaccard')
    app.frame_dynamic_loss_settings.grid_remove()

    # ---- Resize the window to adjust for removed dynamic content ---- 
    app.root.update_idletasks()
    app.root.geometry('')

# ------------------------
# UI Sections
# ------------------------
# ---- Training buttons ----
def build_training_buttons(parent, trainer, app):
    clear_button = ttk.Button(
        parent,
        text='Clear Console',
        command=lambda: clear_console(app)
    )
    clear_button.grid(row=0, column=0, padx=(0,5))

    load_cfg_button = ttk.Button(
        parent,
        text='Load config',
        command=lambda: trainer.load_config_from_file()
    )
    load_cfg_button.grid(row=0, column=1, padx=5)

    save_cfg_button = ttk.Button(
        parent,
        text='Save config',
        command=lambda: trainer.save_config_to_file()
    )
    save_cfg_button.grid(row=0, column=2, padx=5)

    reset_button = ttk.Button(
        parent,
        text='Reset to defaults',
        command=lambda: reset_to_defaults(app)
    )
    reset_button.grid(row=0, column=3, padx=5)

    fit_button = ttk.Button(parent, text='Fit Model', width=9)
    fit_button.grid(row=0, column=4, padx=5)
    app.fit_button = fit_button

    stop_button = ttk.Button(
        parent, 
        text='Stop', 
        width=7,
        command=lambda: trainer.stop()
    )
    stop_button.grid(row=0, column=5, padx=5)
    app.stop_button = stop_button

    # ---- Define control functions ----
    def disable_console_buttons():
        fit_button.state(['disabled'])
        clear_button.state(['disabled'])
        reset_button.state(['disabled'])
        load_cfg_button.state(['disabled'])
        save_cfg_button.state(['disabled'])

    def enable_console_buttons():
        fit_button.state(['!disabled'])
        clear_button.state(['!disabled'])
        reset_button.state(['!disabled'])
        load_cfg_button.state(['!disabled'])
        save_cfg_button.state(['!disabled'])
        stop_button.state(['disabled'])

    def on_fit():
        disable_console_buttons()
        trainer.start(on_complete=enable_console_buttons)

    fit_button.config(command=on_fit)
