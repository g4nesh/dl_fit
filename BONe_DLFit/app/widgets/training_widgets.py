# BONe_DLFit/app/widgets/training_widgets.py
import os
import tkinter as tk
from tkinter import ttk
import psutil
from BONe_utils.utils import has_multiple_gpus
from .base import (
    make_styled_entry,
    make_slider_with_entry,
    build_file_input_row,
    validate_int_partial,
    validate_sci_float_partial,
    clamp_or_default,
    bind_float_range
)

# ------------------------
# UI Sections
# ------------------------
# ---- Reproducibility Mode toggle ----
def build_repro_mode_section(parent, app):
    def toggle_seed_field():
        if app.repro_mode_var.get() == 1:
            app.frame_dynamic_seed.grid()
        else:
            app.frame_dynamic_seed.grid_remove()

    parent.grid_columnconfigure(0, minsize=175)
    ttk.Label(parent, text='Reproducibility Mode:').grid(row=0, column=0, sticky='w')
    app.repro_mode_var = tk.IntVar(value=0)
    ttk.Radiobutton(
        parent,
        text='Off',
        variable=app.repro_mode_var,
        value=0,
        command=toggle_seed_field
    ).grid(row=0, column=1, sticky='w')
    ttk.Radiobutton(
        parent, 
        text='On', 
        variable=app.repro_mode_var, 
        value=1,
        command=toggle_seed_field
    ).grid(row=0, column=2, sticky='w')

    seed_frame = app.frame_dynamic_seed
    seed_frame.grid_columnconfigure(0, minsize=175)
    ttk.Label(seed_frame, text='Random Seed:').grid(row=0, column=0, sticky='w')
    app.random_seed_var = tk.IntVar(value=42)
    app.random_seed_entry = make_styled_entry(seed_frame, app.random_seed_var, validate='int')
    app.random_seed_entry.grid(row=0, column=1, sticky='w')
    seed_frame.grid_remove()

# ---- GPU Mode toggle ----
def build_gpu_mode_section(parent, app):
    parent.grid_columnconfigure(0, minsize=175)
    ttk.Label(parent, text='GPU Mode:').grid(row=0, column=0, sticky='w')
    app.gpu_mode_var = tk.IntVar(value=0)
    ttk.Radiobutton(
        parent,
        text='Single GPU',
        variable=app.gpu_mode_var,
        value=0
    ).grid(row=0, column=1, sticky='w')
    app.multi_gpu_button = ttk.Radiobutton(
        parent, 
        text='Multi GPU', 
        variable=app.gpu_mode_var, 
        value=1
    )
    app.multi_gpu_button.grid(row=0, column=2, sticky='w')
    if not has_multiple_gpus():
        app.multi_gpu_button.state(['disabled'])

# ---- Fitting Mode toggle ----
def build_fitting_mode_section(parent, app):
    # ---- Define lists of model architectures ----
    app.model_arch_vals_2d = [
        'U-Net',
        'UNet++',
        'FPN',
        'PSPNet',
        'DeepLabV3+',
        'Linknet',
        'MAnet',
        'UPerNet',
        'Segformer'
    ]

    app.model_arch_vals_3d = [
        'U-Net',
        'UNet++',
        'FPN',
        'PSPNet',
        'DeepLabV3+',
        'Linknet',
        'MAnet',
        'PAN',
    ]

    # ---- Manual entry snapping logic ----
    def snap_channels(val):
        val = int(round(val))
        if val % 2 == 0:
            val += 1 if val < 9 else -1
        return max(3, min(val, 9))

    # ---- Toggle logic ---- 
    def toggle_fitting_mode_fields():
        mode = app.fitting_mode_var.get()
        if mode == 1:  # 2.5D
            app.frame_dynamic_tile2channels.grid()
            app.frame_dynamic_tile_depth.grid_remove()
            app.frame_dynamic_patch_depth.grid_remove()
        elif mode == 2:  # 3D
            app.frame_dynamic_tile_depth.grid()
            app.frame_dynamic_patch_depth.grid()
            app.frame_dynamic_tile2channels.grid_remove()
        else:  # 2D
            app.frame_dynamic_patch_depth.grid_remove()
            app.frame_dynamic_tile2channels.grid_remove()
            app.frame_dynamic_tile_depth.grid_remove()

        if mode == 2:
            new_vals = app.model_arch_vals_3d
        else:
            new_vals = app.model_arch_vals_2d
        app.model_arch_menu['values'] = new_vals

        if app.model_arch_var.get() not in new_vals:
            app.model_arch_var.set(new_vals[0])

        app.frame_model_arch_menu.grid()
        app.toggle_stride_widget()
    
    # ---- Make accessible for trainer.py ----
    app.toggle_fitting_mode_fields = toggle_fitting_mode_fields

    # ---- Toggle logic for stride ----
    def toggle_stride_widget():
        is_3d = app.fitting_mode_var.get() == 2
        is_unet = app.model_arch_var.get() == 'U-Net'
        is_resnet = app.backbone_var.get() in [
            'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'
        ]
        if is_3d and is_unet and is_resnet:
            app.frame_stride.grid()
        else:
            app.frame_stride.grid_remove()
    
    # ---- Make accessible by the backbone widget ----
    app.toggle_stride_widget = toggle_stride_widget

    # ---- Layout for radio buttons ---- 
    parent.grid_columnconfigure(0, minsize=175)
    ttk.Label(parent, text='Fitting Mode:').grid(row=0, column=0, sticky='w')
    app.fitting_mode_var = tk.IntVar(value=0)
    ttk.Radiobutton(
        parent,
        text='2D',
        variable=app.fitting_mode_var,
        value=0,
        command=toggle_fitting_mode_fields
    ).grid(row=0, column=1, sticky='w')
    ttk.Radiobutton(
        parent, 
        text='2.5D', 
        variable=app.fitting_mode_var, 
        value=1,
        command=toggle_fitting_mode_fields
    ).grid(row=0, column=2, sticky='w')
    ttk.Radiobutton(
        parent, 
        text='3D', 
        variable=app.fitting_mode_var, 
        value=2,
        command=toggle_fitting_mode_fields
    ).grid(row=0, column=3, sticky='w')

    # ---- 2.5 frame setup ---- 
    frame_25d = ttk.Frame(app.frame_dynamic_tile2channels)
    frame_25d.grid(row=0, column=0, sticky='w')
    frame_25d.grid_columnconfigure(0, minsize=175)

    app.tiles2channels_var = tk.IntVar(value=3)
    app.tiles2channels_entry = make_slider_with_entry(
        frame_25d, 'Num. tiles for 2.5D mode:', app.tiles2channels_var, row=0, from_=3, to_=9,
        format_func=str,
        parse_func=int,
        snap_func=snap_channels,
    )

    # ---- 3D tile depth frame setup ---- 
    frame_tile_depth = ttk.Frame(app.frame_dynamic_tile_depth)
    frame_tile_depth.grid(row=0, column=0, sticky='w')
    frame_tile_depth.grid_columnconfigure(0, minsize=175)

    app.tile_depth_var = tk.IntVar(value=32)
    app.tile_depth_entry = make_slider_with_entry(
        frame_tile_depth, 'Tile Depth [px]:', app.tile_depth_var, row=0, from_=32, to_=256,
        format_func=str,
        parse_func=int,
        snap_func=lambda v: round(v / 32) * 32,
    )

    # ---- 3D patch depth frame setup ---- 
    frame_patch_depth = ttk.Frame(app.frame_dynamic_patch_depth)
    frame_patch_depth.grid(row=0, column=0, sticky='w')
    frame_patch_depth.grid_columnconfigure(0, minsize=175)

    app.patch_depth_var = tk.IntVar(value=32)
    app.patch_depth_entry = make_slider_with_entry(
        frame_patch_depth, 'Patch Depth [px]:', app.patch_depth_var, row=0, from_=32, to_=256,
        format_func=str,
        parse_func=int,
        snap_func=lambda v: round(v / 32) * 32,
    )

    # ---- Model architecture menu ----
    frame_arch_menu = ttk.Frame(app.frame_model_arch_menu)
    frame_arch_menu.grid(row=0, column=0, sticky='w')
    frame_arch_menu.grid_columnconfigure(0, minsize=175)

    ttk.Label(frame_arch_menu, text='Model architecture:').grid(row=0, column=0, sticky='w')

    app.model_arch_var = tk.StringVar(value='U-Net')
    app.model_arch_menu = ttk.Combobox(
        frame_arch_menu,
        values=(
            app.model_arch_vals_2d
            if app.fitting_mode_var.get() != 2
            else app.model_arch_vals_3d
        ),
        state='readonly',
        textvariable=app.model_arch_var,
        width=15
    )
    app.model_arch_var.trace_add('write', lambda *args: app.toggle_stride_widget())
    app.model_arch_menu.grid(row=0, column=1)
    
    # ---- 3D U-Net stride entry ----
    frame_stride = ttk.Frame(app.frame_stride)
    frame_stride.grid(row=0, column=0, sticky='w')
    frame_stride.grid_columnconfigure(0, minsize=175)
    
    ttk.Label(frame_stride, text='Custom stride:').grid(row=0, column=0, sticky='w')
    
    app.stride_var = tk.StringVar(value='((2,2,2),(2,2,2),(2,2,2),(2,2,2),(2,2,2))')
    app.stride_entry = make_styled_entry(frame_stride, app.stride_var, width=45)
    app.stride_entry.grid(row=0, column=1)
    
    # ---- Initial state ---- 
    app.frame_dynamic_tile2channels.grid_remove()
    app.frame_dynamic_tile_depth.grid_remove()
    app.frame_dynamic_patch_depth.grid_remove()
    app.frame_stride.grid_remove()

# ---- Patch size ----
def build_patch_size_section(parent, app):
    parent.grid_columnconfigure(0, minsize=175)
    app.patch_var = tk.IntVar(value=256)
    app.patch_size_entry = make_slider_with_entry(
        parent, 'Patch Size [px]:', app.patch_var, row=0, from_=32, to_=2048,
        format_func=str,
        parse_func=int,
        snap_func=lambda v: round(v / 32) * 32,
    )

# ---- Patches per tile ----
def build_patch_per_tile_section(parent, app):
    parent.grid_columnconfigure(0, minsize=175)
    ttk.Label(parent, text='Patches per tile:').grid(row=0, column=0, sticky='w')
    app.patch_per_tile_var = tk.IntVar(value=4)
    app.patch_per_tile_entry = make_styled_entry(
        parent, app.patch_per_tile_var, width=5, validate='int'
    )
    app.patch_per_tile_entry.grid(row=0, column=1)

# ---- Maximum patch overlap ----
def build_patch_overlap_section(parent, app):
    parent.grid_columnconfigure(0, minsize=175)
    app.patch_overlap_var = tk.DoubleVar(value=0.3)
    app.patch_overlap_entry = make_slider_with_entry(
        parent,
        'Max. patch overlap:',
        app.patch_overlap_var,
        row=0, from_=0, to_=1,
        format_func=lambda v: f'{v:.2e}' if abs(v) < 0.001 or abs(v) > 100 else f'{v:.4f}',
        parse_func=float,
    )

# ---- Data augmentation toggle ----
def build_augmentations_section(parent, app):
    parent.grid_columnconfigure(0, minsize=175)
    ttk.Label(parent, text='Augmentations:').grid(row=0, column=0, sticky='w')
    app.augment_var = tk.IntVar(value=1)
    ttk.Radiobutton(
        parent,
        text='Off',
        variable=app.augment_var,
        value=0
    ).grid(row=0, column=1, sticky='w')
    ttk.Radiobutton(
        parent, 
        text='On', 
        variable=app.augment_var, 
        value=1
    ).grid(row=0, column=2, sticky='w')

# ---- Normalization toggle ----
def build_norm_mode_section(parent, app):
    parent.grid_columnconfigure(0, minsize=175)
    ttk.Label(parent, text='Normalization:').grid(row=0, column=0, sticky='w')
    app.norm_mode_var = tk.IntVar(value=0)
    ttk.Radiobutton(
        parent,
        text='ZScore',
        variable=app.norm_mode_var,
        value=0
    ).grid(row=0, column=1, sticky='w')
    ttk.Radiobutton(
        parent, 
        text='minMax', 
        variable=app.norm_mode_var, 
        value=1
    ).grid(row=0, column=2, sticky='w')

# ---- Training / Validation split ----
def build_train_val_split_section(parent, app):
    parent.grid_columnconfigure(0, minsize=175)
    app.train_val_split_var = tk.DoubleVar(value=0.8)
    app.train_val_split_entry = make_slider_with_entry(
        parent,
        'Train/Val split:',
        app.train_val_split_var,
        row=0, from_=0, to_=1,
        format_func=lambda v: f'{v:.2e}' if abs(v) < 0.001 or abs(v) > 100 else f'{v:.4f}',
        parse_func=float,
    )

# ---- Number of epochs ----
def build_epochs_section(parent, app):
    parent.grid_columnconfigure(0, minsize=175)
    ttk.Label(parent, text='Num. of epochs:').grid(row=0, column=0, sticky='w')
    app.num_epochs_var = tk.IntVar(value=25)
    app.num_epochs_entry = make_styled_entry(parent, app.num_epochs_var, width=7, validate='int')
    app.num_epochs_entry.grid(row=0, column=1)

# ---- Optimizer ----
def build_optimizer_section(parent, app):
    parent.grid_columnconfigure(0, minsize=175)
    ttk.Label(parent, text='Optimizer:').grid(row=0, column=0, sticky='w')
    app.optim_var = tk.StringVar(value='Adam')
    
    optim_names = ['Adam','AdamW', 'SGD']
    
    optim_combo = ttk.Combobox(
        parent,
        values=optim_names,
        state='readonly',
        textvariable=app.optim_var,
        width=15
    )
    optim_combo.grid(row=0, column=1, sticky='w')
    
    # ---- Dynamic optimizer settings frame to right of menu ----
    app.frame_dynamic_optim_settings = ttk.Frame(parent)
    app.frame_dynamic_optim_settings.grid(
        row=0, column=2, columnspan=4, sticky='w'
    )
    build_dynamic_optimizer_settings(app.frame_dynamic_optim_settings, app)
    app.frame_dynamic_optim_settings.grid_remove()
    
    # ---- Bind the optimizer changes ----
    app.optim_var.trace_add('write', lambda *args: update_optimizer_settings(app))
    
    # ---- Exposes update to trainer.py ----
    app.update_optimizer_settings = lambda: update_optimizer_settings(app)

# ---- Dynamic Optimizer Settings ----
def build_dynamic_optimizer_settings(parent, app):
    # ---- Shared key validator for all float entries ----
    vcmd_key = (parent.register(validate_sci_float_partial), '%P')
    
    # ---- Weight decay ----
    ttk.Label(parent, text='Weight decay:').grid(row=0, column=0, sticky='w', padx=(10, 0))
    app.weight_decay_var = tk.StringVar(value='0.01')
    app.weight_decay_entry = make_styled_entry(parent, app.weight_decay_var, width=7)
    app.weight_decay_entry.grid(row=0, column=1, sticky='w', padx=(2, 5))
    
    # ---- Check for valid weight decay values (0.0 to 0.1) ----
    app.weight_decay_entry.config(validate='key', validatecommand=vcmd_key)
    bind_float_range(
        app.weight_decay_entry,
        app.weight_decay_var,
        min_val=0.0,
        max_val=0.1,
        default='0.01'
    )
    
    # ---- Momentum ----
    ttk.Label(parent, text='Momentum:').grid(row=0, column=2, sticky='w', padx=(5, 0))
    app.momentum_var = tk.StringVar(value='0.9')
    app.momentum_entry = make_styled_entry(parent, app.momentum_var, width=7)
    app.momentum_entry.grid(row=0, column=3, sticky='w', padx=(2, 5))
    
    # ---- Check for valid momentum values (0.0 to 0.99) ----
    app.momentum_entry.config(validate='key', validatecommand=vcmd_key)
    bind_float_range(
        app.momentum_entry,
        app.momentum_var,
        min_val=0.0,
        max_val=0.99,
        default='0.9'
    )

def update_optimizer_settings(app):
    # ---- Show/hide and update optimizer settings based on selected optimizer ----
    optimizer = app.optim_var.get()
    
    if optimizer == 'Adam':
        # ---- Hide the dynamic settings frame ----
        app.frame_dynamic_optim_settings.grid_remove()
        return
    
    app.frame_dynamic_optim_settings.grid()
        
    if optimizer == 'AdamW':
        # ---- Enable AdamW settings ----
        app.weight_decay_entry.config(state='normal')
        if not app.weight_decay_var.get():
            app.weight_decay_var.set('0.01')
        app.momentum_entry.config(state='disabled')
        
    elif optimizer == 'SGD':
        # ---- Enable SGD settings ----
        app.weight_decay_entry.config(state='normal')
        app.momentum_entry.config(state='normal')
        if not app.weight_decay_var.get():
            app.weight_decay_var.set('0.0')
        if not app.momentum_var.get():
            app.momentum_var.set('0.9')

# ---- Learning rate ----
def build_learning_rate_section(parent, app):
    parent.grid_columnconfigure(0, minsize=175)
    app.learning_rate_var = tk.DoubleVar(value=0.001)
    app.learning_rate_entry = make_slider_with_entry(
        parent,
        'Learning rate:',
        app.learning_rate_var,
        row=0, from_=1e-15, to_=100,
        format_func=lambda v: f'{v:.2e}' if abs(v) < 0.001 or abs(v) > 100 else f'{v:.4f}',
        parse_func=float,
    )

# ---- Batch size ----
def build_batch_size_section(parent, app):
    parent.grid_columnconfigure(0, minsize=175)
    ttk.Label(parent, text='Batch size:').grid(row=0, column=0, sticky='w')
    app.batch_size_var = tk.IntVar(value=32)
    app.batch_size_entry = make_styled_entry(parent, app.batch_size_var, width=7, validate='int')
    app.batch_size_entry.grid(row=0, column=1)

# ---- Loss function ----
def build_loss_fxn_section(parent, app):
    parent.grid_columnconfigure(0, minsize=175)
    ttk.Label(parent, text='Loss function:').grid(row=0, column=0, sticky='w')
    app.loss_var = tk.StringVar(value='Jaccard')
    
    loss_names = [
        'Jaccard','Dice', 'Lovasz-Softmax', 'Focal','Tversky', 'Focal Tversky'
    ]
    
    ttk.Combobox(
        parent,
        values=loss_names,
        state='readonly',
        textvariable=app.loss_var,
        width=15
    ).grid(row=0, column=1, sticky='w')

    # ---- Bind the loss function selection to update dynamic settings ----
    app.loss_var.trace_add('write', lambda *args: update_loss_settings(app))

    # ---- Exposes update to trainer.py ----
    app.update_loss_settings = lambda: update_loss_settings(app)

# ---- Dynamic Loss function settings ----
def build_dynamic_loss_settings(parent, app):
    # ---- Shared key validator for all float entries ----
    vcmd_key = (parent.register(validate_sci_float_partial), '%P')
    
    # ---- Build frame for alpha, beta, gamma, and Focal:Tversky Wt entries ----
    parent.grid_columnconfigure(0, minsize=175)
    ttk.Label(parent, text='Loss settings:').grid(row=0, column=0, sticky='w')
    
    # ---- Tversky alpha ----
    ttk.Label(parent, text='α:').grid(row=0, column=1, sticky='w')
    app.tver_alpha_var = tk.StringVar(value='1.25')
    app.tver_alpha_entry = make_styled_entry(parent, app.tver_alpha_var, width=5)
    app.tver_alpha_entry.grid(row=0, column=2, sticky='w', padx=(2, 5))
    
    # ---- Check for valid Tversky alpha values (0.0 to 5.0) ----
    app.tver_alpha_entry.config(validate='key', validatecommand=vcmd_key)
    bind_float_range(
        app.tver_alpha_entry,
        app.tver_alpha_var,
        min_val=0.0,
        max_val=5.0,
        default='1.25'
    )
    
    # ---- Tversky beta ----
    ttk.Label(parent, text='β:').grid(row=0, column=3, sticky='w', padx=(5, 0))
    app.tver_beta_var = tk.StringVar(value='1.25')
    app.tver_beta_entry = make_styled_entry(parent, app.tver_beta_var, width=5)
    app.tver_beta_entry.grid(row=0, column=4, sticky='w', padx=(2, 5))

    # ---- Check for valid Tversky beta values (0.0 to 5.0) ----
    app.tver_beta_entry.config(validate='key', validatecommand=vcmd_key)
    bind_float_range(
        app.tver_beta_entry,
        app.tver_beta_var,
        min_val=0.0,
        max_val=5.0,
        default='1.25'
    )
    
    # ---- Focal loss gamma ----
    ttk.Label(parent, text='γ:').grid(row=0, column=5, sticky='w', padx=(5, 0))
    app.foc_gamma_var = tk.StringVar(value='2.0')
    app.foc_gamma_entry = make_styled_entry(parent, app.foc_gamma_var, width=5)
    app.foc_gamma_entry.grid(row=0, column=6, sticky='w', padx=(2, 5))

    # ---- Check for valid Focal loss gamma values (0.0 to 5.0) ----
    app.foc_gamma_entry.config(validate='key', validatecommand=vcmd_key)
    bind_float_range(
        app.foc_gamma_entry,
        app.foc_gamma_var,
        min_val=0.0,
        max_val=5.0,
        default='2.0'
    )
    
    # ---- Focal:Tversky weight ----
    ttk.Label(parent, text='Focal:Tversky Wt:').grid(row=0, column=7, sticky='w', padx=(5, 0))
    app.foc_tver_var = tk.StringVar(value='0.5')
    app.foc_tver_entry = make_styled_entry(parent, app.foc_tver_var, width=5)
    app.foc_tver_entry.grid(row=0, column=8, sticky='w', padx=(2, 5))
    
    # ---- Check for valid Focal:Tversky weight (0.1 to 0.9) ----
    app.foc_tver_entry.config(validate='key', validatecommand=vcmd_key)
    bind_float_range(
        app.foc_tver_entry,
        app.foc_tver_var,
        min_val=0.1,
        max_val=0.9,
        default='0.5'
    )
    
    # Initially hide the frame
    parent.grid_remove()

def update_loss_settings(app):
    # ---- Show/hide and update optimizer settings based on selected optimizer ----
    loss_name = app.loss_var.get()
    
    if loss_name in ('Jaccard', 'Dice', 'Lovasz-Softmax'):
        # ---- Hide the dynamic settings frame ----
        app.frame_dynamic_loss_settings.grid_remove()
        return
    
    app.frame_dynamic_loss_settings.grid()
    
    settings_map ={
        'Focal': {
            'enabled': {'foc_gamma_entry': '2.0'},
            'disabled': ['tver_alpha_entry', 'tver_beta_entry', 'foc_tver_entry']
        },
        'Tversky': {
            'enabled': {'tver_alpha_entry': '1.25', 'tver_beta_entry': '1.25'},
            'disabled': ['foc_gamma_entry', 'foc_tver_entry']
        },
        'Focal Tversky': {
            'enabled': {
                'tver_alpha_entry': '1.25',
                'tver_beta_entry': '1.25',
                'foc_gamma_entry': '2.0',
                'foc_tver_entry': '0.5',
            },
        }
    }
    settings = settings_map.get(loss_name, {})
    
    # ---- Enable fields and set defaults if empty ----
    for field_name, default_val in settings.get('enabled', {}).items():
        entry = getattr(app, field_name)
        entry.config(state='normal')
        var_name = field_name.replace('_entry', '_var')
        var = getattr(app, var_name)
        if not var.get():
            var.set(default_val)

    # ---- Disable fields ----
    for field_name in settings.get('disabled', []):
        getattr(app, field_name).config(state='disabled')
    
# ---- Number of PyTorch DataLoader workers to assign ----
def build_pt_settings_section(parent, app):
    parent.grid_columnconfigure(0, minsize=175)
    ttk.Label(parent, text='PyTorch settings:').grid(row=0, column=0, sticky='w')
    ttk.Label(parent, text='DataLoader workers:').grid(row=0, column=1, sticky='w')
    is_windows = (os.name == 'nt')
    max_ptworkers = psutil.cpu_count(logical=False)
    app.pt_workers_var = tk.IntVar(value=0 if is_windows else 4)
    worker_selector = ttk.Combobox(
        parent,
        values=list(range(0, max_ptworkers + 1)),
        state='readonly',
        textvariable=app.pt_workers_var,
        width=5
    )
    worker_selector.grid(row=0, column=2, sticky='w', padx=(2, 5))
    if is_windows:
        app.pt_workers_var.set(0)
        worker_selector.state(['disabled'])
    
    ttk.Label(parent, text='Prefetch factor:').grid(row=0, column=3, sticky='w', padx=(5, 0))
    max_prefetch = 16
    is_windows = (os.name == 'nt')
    app.prefetch_factor_var = tk.IntVar(value=0 if is_windows else 2)
    prefetch_selector = ttk.Combobox(
        parent,
        values=list(range(0, max_prefetch + 1, 2)),
        state='readonly',
        textvariable=app.prefetch_factor_var,
        width=5
    )
    prefetch_selector.grid(row=0, column=4, sticky='w', padx=(2, 5))
    if is_windows:
        app.prefetch_factor_var.set(0)
        prefetch_selector.state(['disabled'])
        
# ---- Save folder ----
def build_save_folder_section(parent, app):
    parent.grid_columnconfigure(0, minsize=175)
    build_file_input_row(
        parent,
        'Save folder:',
        0,
        'save_entry',
        app.trainer.browse_save_folder,
        app
    )

# ---- Model filename ----
def build_model_name_section(parent, app):
    parent.grid_columnconfigure(0, minsize=175)
    ttk.Label(parent, text='Model filename:').grid(row=0, column=0, sticky='w')
    app.model_name_var = tk.StringVar(value='')
    app.model_name_entry = make_styled_entry(parent, app.model_name_var, width=45)
    app.model_name_entry.grid(row=0, column=1)
