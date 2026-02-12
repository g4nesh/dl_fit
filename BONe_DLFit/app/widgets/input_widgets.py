# BONe_DLFit/app/widgets/input_widgets.py
import tkinter as tk
from tkinter import ttk
from .base import (
    make_styled_entry,
    build_file_input_row,
    update_fit_button_state
)

# ------------------------
# UI Sections
# ------------------------
# ---- Input 1 ----
def build_input_section(parent, app):
    parent.grid_columnconfigure(0, minsize=175)
    build_file_input_row(
        parent,
        'Input1 folder',
        0,
        'input_entry',
        app.trainer.browse_input,
        app
    )

# ---- Mask 1 ----
def build_mask_section(parent, app):
    parent.grid_columnconfigure(0, minsize=175)
    build_file_input_row(
        parent,
        'Mask1 folder',
        0,
        'mask_entry',
        app.trainer.browse_mask,
        app
    )

# ---- Additional input-mask pair selector ----
def build_add_pair_selector_section(parent, app):
    parent.grid_columnconfigure(0, minsize=175)
    ttk.Label(parent, text='Num. of add. input-mask pairs:').grid(row=0, column=0, sticky='w')
    app.num_pairs_var = tk.IntVar(value=0)
    pair_selector = ttk.Combobox(
        parent,
        values=list(range(0, 20)),
        state='readonly',
        textvariable=app.num_pairs_var,
        width=5
    )
    pair_selector.grid(row=0, column=1, sticky='w')

    app.additional_data_frames = []

    def _update_add_pairs(event=None):
        # ---- Clear existing additional frames ---- 
        for f in app.additional_data_frames:
            f.destroy()
        app.additional_data_frames.clear()

        # ---- Also forget frame_dynamic_inputs' grid if empty ---- 
        if app.frame_dynamic_inputs.winfo_children():
            for child in app.frame_dynamic_inputs.winfo_children():
                child.destroy()

        count = app.num_pairs_var.get()

        if count == 0:
            app.frame_dynamic_inputs.grid_remove()
        else:
            app.frame_dynamic_inputs.grid()
        
        for i in range(1, count + 1):
            frame = ttk.Frame(app.frame_dynamic_inputs)
            frame.grid(row=i-1, column=0, sticky='w')
            frame.grid_columnconfigure(0, minsize=175)

            input_label = f'Input{i+1} folder'
            mask_label = f'Mask{i+1} folder'

            input_entry = make_styled_entry(frame, var=None, width=45)
            mask_entry = make_styled_entry(frame, var=None, width=45)

            # ---- Store reference on app if needed ---- 
            setattr(app, f'input_entry_{i}', input_entry)
            setattr(app, f'mask_entry_{i}', mask_entry)

            # ---- Input row ---- 
            ttk.Label(frame, text=input_label).grid(row=0, column=0, sticky='w')
            input_entry.grid(row=0, column=1)
            ttk.Button(
                frame,
                text='Browse',
                command=lambda e=input_entry: app.trainer.browse_input_dynamic(
                    e,
                    on_update=lambda _: update_fit_button_state(app)
                )
            ).grid(row=0, column=2, pady=5)
            input_entry.bind('<KeyRelease>', lambda e: update_fit_button_state(app))

            # ---- Mask row ---- 
            ttk.Label(frame, text=mask_label).grid(row=1, column=0, sticky='w')
            mask_entry.grid(row=1, column=1)
            ttk.Button(
                frame,
                text='Browse',
                command=lambda e=mask_entry: app.trainer.browse_mask_dynamic(
                    e,
                    on_update=lambda _: update_fit_button_state(app)
                )
            ).grid(row=1, column=2, pady=5)
            mask_entry.bind('<KeyRelease>', lambda e: update_fit_button_state(app))

            app.additional_data_frames.append(frame)

        # ---- Force geometry update so window resizes ---- 
        parent.update_idletasks()
        app.canvas.configure(scrollregion=app.canvas.bbox('all'))
        
        # ---- Explicitly reset window size if 0 dynamic fields ---- 
        MAX_HEIGHT = 820
        WIDTH = 600

        content_bbox = app.canvas.bbox('all')
        if content_bbox:
            content_height = content_bbox[3]
            window_height = min(content_height, MAX_HEIGHT)
            app.root.geometry(f'{WIDTH}x{window_height}')

            app.canvas.configure(scrollregion=content_bbox)

    # ---- Bind update function ---- 
    app.update_add_pairs = _update_add_pairs
    pair_selector.bind('<<ComboboxSelected>>', _update_add_pairs)
