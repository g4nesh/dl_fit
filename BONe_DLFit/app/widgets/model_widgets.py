# BONe_DLFit/app/widgets/model_widgets.py
import tkinter as tk
from tkinter import ttk
from .base import make_styled_entry

# ------------------------
# UI Sections
# ------------------------
# ---- Backbone menu ----
def build_backbone_menu_section(parent, app):
    parent.grid_columnconfigure(0, minsize=175)
    ttk.Label(parent, text='Backbone:').grid(row=0, column=0, sticky='w')
    backbone_vals = [
        'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
        'resnext50_32x4d', 'resnext101_32x4d', 'resnext101_32x8d', 
        'resnext101_32x16d', 'resnext101_32x32d', 'resnext101_32x48d',
        'dpn68', 'dpn68b', 'dpn92', 'dpn98', 'dpn107', 'dpn131',
        'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn',
        'senet154', 'se_resnet50', 'se_resnet101', 'se_resnet152', 'se_resnext50_32x4d',
        'se_resnext101_32x4d',
        'densenet121', 'densenet169', 'densenet201', 'densenet161',
        'inceptionresnetv2', 'inceptionv4',
        'efficientnet-b0', 'efficientnet-b1', 'efficientnet-b2', 'efficientnet-b3', 'efficientnet-b4',
        'efficientnet-b5', 'efficientnet-b6', 'efficientnet-b7',
        'mobilenet_v2',
        'xception',
        'mit_b0', 'mit_b1', 'mit_b2', 'mit_b3', 'mit_b4', 'mit_b5',
        'mobileone_s0', 'mobileone_s1', 'mobileone_s2', 'mobileone_s3', 'mobileone_s4'
    ]

    app.backbone_var = tk.StringVar(value='resnet18') # resnet-18 is default

    backbone_menu = ttk.Combobox(
        parent,
        values=backbone_vals,
        state='readonly',
        textvariable=app.backbone_var,
        width=15
    )
    app.backbone_var.trace_add('write', lambda *args: app.toggle_stride_widget())
    backbone_menu.grid(row=0, column=1)

# ---- Weights ----
def load_custom_weights_section(app):
    # ---- Dynamically create the custom weights entry field and button. ---- 
    if app.frame_dynamic_cust_weights.winfo_children():
        for child in app.frame_dynamic_cust_weights.winfo_children():
            child.destroy()

    frame = ttk.Frame(app.frame_dynamic_cust_weights)
    frame.grid(row=0, column=0, sticky='w')
    frame.grid_columnconfigure(0, minsize=175)

    ttk.Label(frame, text='Load custom weights [.PTH]:').grid(row=0, column=0, sticky='w')

    app.cust_weights_path = tk.StringVar(value='')
    app.cust_weights_entry = make_styled_entry(frame, app.cust_weights_path, width=45)
    app.cust_weights_entry.grid(row=0, column=1)

    ttk.Button(
        frame,
        text='Browse',
        command=lambda e=app.cust_weights_entry: app.trainer.browse_pth_dynamic(e)
    ).grid(row=0, column=2, pady=5)

    app.frame_dynamic_cust_weights.grid()

def build_weights_section(parent, app):
    def _load_cust_weights(event=None):
        selection = app.weights_var.get()

        if selection == 2:
            load_custom_weights_section(app)
        else:
            app.frame_dynamic_cust_weights.grid_remove()

    parent.grid_columnconfigure(0, minsize=175)
    ttk.Label(parent, text='Weights:').grid(row=0, column=0, sticky='w')
    app.weights_var = tk.IntVar(value=0)
    ttk.Radiobutton(
        parent,
        text='Random init.',
        variable=app.weights_var,
        value=0,
        command=_load_cust_weights
    ).grid(row=0, column=1)
    ttk.Radiobutton(
        parent, 
        text='imagenet', 
        variable=app.weights_var, 
        value=1,
        command=_load_cust_weights
    ).grid(row=0, column=2)
    ttk.Radiobutton(
        parent, 
        text='Custom weights', 
        variable=app.weights_var, 
        value=2,
        command=_load_cust_weights
    ).grid(row=0, column=3)

    content_bbox = app.canvas.bbox('all')
    if content_bbox:
        content_height = content_bbox[3]
        new_height = max(content_height, 820)
        app.root.geometry(f'600x{new_height}')
