# BONe_DLFit/app/widgets/__init__.py
from .theme import build_theme
from .base import (
    make_styled_entry,
    make_slider_with_entry,
    build_file_input_row,
    validate_int_partial,
    validate_sci_float_partial,
    clamp_or_default,
    bind_float_range,
    update_fit_button_state
)
from .input_widgets import (
    build_input_section,
    build_mask_section,
    build_add_pair_selector_section
)
from .model_widgets import (
    build_backbone_menu_section,
    build_weights_section
)
from .training_widgets import (
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
    update_optimizer_settings,
    build_learning_rate_section,
    build_batch_size_section,
    build_loss_fxn_section,
    build_dynamic_loss_settings,
    update_loss_settings,
    build_pt_settings_section,
    build_save_folder_section,
    build_model_name_section
)
from .button_widgets import (
    reset_to_defaults,
    build_training_buttons
)
from .console_widgets import (
    build_console_section,
    poll_console,
    poll_progress,
    build_progressbar_section,
)
