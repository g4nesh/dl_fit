# BONe_DLFit/app/gui.py
import tkinter as tk
from tkinter import ttk
from BONe_DLFit.core.trainer import Trainer
from BONe_utils.utils import log_to_console, load_config
from .widgets.base import update_fit_button_state
from .widgets.theme import build_theme
from .widgets.input_widgets import (
    build_input_section,
    build_mask_section,
    build_add_pair_selector_section
)
from .widgets.model_widgets import (
    build_backbone_menu_section,
    build_weights_section
)
from .widgets.training_widgets import (
    build_repro_mode_section,
    build_gpu_mode_section,
    build_fitting_mode_section,
    build_patch_size_section,
    build_patch_per_tile_section,
    build_patch_overlap_section,
    build_augmentations_section,
    build_norm_mode_section,
    build_train_val_split_section,
    build_epochs_section,
    build_optimizer_section,
    build_dynamic_optimizer_settings,
    build_learning_rate_section,
    build_batch_size_section,
    build_loss_fxn_section,
    build_dynamic_loss_settings,
    build_pt_settings_section,
    build_save_folder_section,
    build_model_name_section
)
from .widgets.button_widgets import (
    build_training_buttons
)

from .widgets.console_widgets import (
    build_console_section,
    poll_console,
    poll_progress,
    build_progressbar_section
)

class BONe_DLFit:
    def __init__(self, root):
        self.root = root

        # ---- Apply and store the theme ---- 
        self.theme = build_theme(self.root)

        # ---- Load configuration and initialize trainer ---- 
        self.config = load_config()
        self.trainer = Trainer(self)

        # ---- Make logger aware of the GUI ---- 
        log_to_console.app = self

        # ---- Build UI and load config ---- 
        self._build_ui()
        self.trainer.load_previous_config(self.config)

        # ---- Check Fit button state ----
        self.root.after(200, lambda: update_fit_button_state(self))

        # ---- Start polling to console after UI is built ----
        poll_console(self)
        poll_progress(self)

    def _build_ui(self):
        # ---- Create Scrollable Canvas Layout ----
        container = ttk.Frame(self.root)
        container.grid(row=0, column=0, sticky='nsew')

        # ---- Configure weight for responsiveness ----
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)

        # ---- Scrollbar ----
        self.scrollbar = ttk.Scrollbar(container, orient='vertical')
        self.scrollbar.grid(row=0, column=1, sticky='ns')

        # ---- Canvas ----
        self.canvas = tk.Canvas(
            container,
            highlightthickness=0,
            yscrollcommand=self.scrollbar.set,
            bg=self.theme['bg']
        )
        self.canvas.grid(row=0, column=0, sticky='nsew')

        # ---- Link scrollbar command to Canvas ----
        self.scrollbar.config(command=self.canvas.yview)

        # ---- Scrollable Frame ----
        self.scrollable_frame = ttk.Frame(self.canvas)
        self.scroll_window = self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor='nw')

        # ---- Allow resizing content ----
        self.scrollable_frame.grid_columnconfigure(0, weight=1)

        # ---- Binds for Canvas behavior ----
        def _on_mousewheel(event):
            self.canvas.yview_scroll(int(-1 * (event.delta / 120)), 'units')

        def on_canvas_configure(event):
            self.canvas.itemconfig(self.scroll_window, width=event.width)

        def on_frame_configure(event):
            self.canvas.configure(scrollregion=self.canvas.bbox('all'))

            # ---- Dynamically show/hide scrollbar ---- 
            bbox = self.canvas.bbox('all')
            if bbox:
                content_height = bbox[3]
                if content_height > 792:
                    if not self.scrollbar.winfo_ismapped():
                        self.scrollbar.grid()
                else:
                    if self.scrollbar.winfo_ismapped():
                        self.scrollbar.grid_remove()

        # ---- Event bindings ----
        self.canvas.bind('<Configure>', on_canvas_configure)
        self.scrollable_frame.bind('<Configure>', on_frame_configure)
        self.scrollable_frame.bind('<Enter>', lambda e: self.canvas.bind_all('<MouseWheel>', _on_mousewheel))
        self.scrollable_frame.bind('<Leave>', lambda e: self.canvas.unbind_all('<MouseWheel>'))

        # ---- All UI elements in scrollable frame ----
        # list of 'frame name', row number, and optional grid_columnconfigure control
        frame_defs = [
            ('frame_inputs', 0),
            ('frame_masks', 1),
            ('frame_dynamic_inputs', 2, {'grid_columnconfigure': (0, 1)}),
            ('frame_add_pair_selector', 3),
            ('frame_repro_mode', 4),
            ('frame_dynamic_seed', 5, {'grid_columnconfigure': (1, 0)}),
            ('frame_gpu_mode', 6),
            ('frame_fitting_mode', 7),
            ('frame_dynamic_tile2channels', 8, {'grid_columnconfigure': (1, 0)}),
            ('frame_dynamic_tile_depth', 9, {'grid_columnconfigure': (0, 1)}),
            ('frame_patch_size', 10),
            ('frame_dynamic_patch_depth', 11, {'grid_columnconfigure': (1, 0)}),
            ('frame_patch_per_tile', 12),
            ('frame_patch_overlap', 13),
            ('frame_aug', 14),
            ('frame_norm_mode', 15),
            ('frame_train_val_split', 16),
            ('frame_model_arch_menu', 17, {'grid_columnconfigure': (0, 1)}),
            ('frame_backbone_menu', 18),
            ('frame_stride', 19),
            ('frame_weights', 20),
            ('frame_dynamic_cust_weights', 21, {'grid_columnconfigure': (0, 1)}),
            ('frame_epochs', 22),
            ('frame_optimizer', 23),
            ('frame_learning_rate', 24),
            ('frame_batch_size', 25),
            ('frame_loss_fxn', 26),
            ('frame_dynamic_loss_settings', 27, {'grid_columnconfigure': (1, 0)}),
            ('frame_pt_settings', 28),
            ('frame_save_folder', 29),
            ('frame_model_name', 30),
            ('frame_buttons', 31),
            ('frame_console', 32, {'sticky': 'nsew'}),
            ('frame_progressbar', 33, {'sticky': 'nsew'}),
        ]

        for name, row, *opts in frame_defs:
            # Constructs frames from list
            frame = ttk.Frame(self.scrollable_frame)
            kwargs = {'sticky': 'ew', 'padx': 5, 'pady': 5}
            if opts:
                extra = opts[0]
                if 'sticky' in extra:
                    kwargs['sticky'] = extra['sticky']
            frame.grid(row=row, column=0, **kwargs)

            # Adds grid column configuration if specified
            if opts and 'grid_columnconfigure' in opts[0]:
                col, weight = opts[0]['grid_columnconfigure']
                frame.grid_columnconfigure(col, weight=weight)

            setattr(self, name, frame)

        # ---- Build sections ----
        build_input_section(self.frame_inputs, self)
        build_mask_section(self.frame_masks, self)
        build_add_pair_selector_section(self.frame_add_pair_selector, self)
        build_repro_mode_section(self.frame_repro_mode, self)
        build_gpu_mode_section(self.frame_gpu_mode, self)
        build_fitting_mode_section(self.frame_fitting_mode, self)
        build_patch_size_section(self.frame_patch_size, self)
        build_patch_per_tile_section(self.frame_patch_per_tile, self)
        build_patch_overlap_section(self.frame_patch_overlap, self)
        build_augmentations_section(self.frame_aug, self)
        build_norm_mode_section(self.frame_norm_mode, self)
        build_train_val_split_section(self.frame_train_val_split, self)
        build_backbone_menu_section(self.frame_backbone_menu, self)
        build_weights_section(self.frame_weights, self)
        build_epochs_section(self.frame_epochs, self)
        build_optimizer_section(self.frame_optimizer, self)
        build_dynamic_optimizer_settings(self.frame_dynamic_optim_settings, self)
        build_learning_rate_section(self.frame_learning_rate, self)
        build_batch_size_section(self.frame_batch_size, self)
        build_loss_fxn_section(self.frame_loss_fxn, self)
        build_dynamic_loss_settings(self.frame_dynamic_loss_settings, self)
        build_pt_settings_section(self.frame_pt_settings, self)
        build_save_folder_section(self.frame_save_folder, self)
        build_model_name_section(self.frame_model_name, self)
        build_training_buttons(self.frame_buttons, self.trainer, self)
        build_console_section(self.frame_console, self)
        build_progressbar_section(self.frame_progressbar, self)

        # ---- Bind manual typing updates to Fit button state ----
        for attr in ['input_entry', 'mask_entry', 'save_entry']:
            entry = getattr(self, attr)
            entry.bind('<KeyRelease>', lambda e: update_fit_button_state(self))
