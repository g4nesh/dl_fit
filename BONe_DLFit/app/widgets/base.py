# BONe_DLFit/app/widgets/base.py
import os
import tkinter as tk
from tkinter import ttk

# ------------------------
# Reusable Widget Helpers
# ------------------------
def validate_int_partial(value):
    if value == '':
        return True
    if value == '-':
        return True
    return value.isdigit() or (value.startswith('-') and value[1:].isdigit())

def validate_sci_float_partial(value):
    if value == '':
        return True
    try:
        float(value)
        return True
    except ValueError:
        # allow partial scientific notation states
        return value.lower() in {
            '-', '.', '-.', 'e', 'e-', '.e', '.e-', '-e', '-e-'
        } or value.lower().endswith(('e', 'e+', 'e-'))

def clamp_or_default(var, min_val, max_val, default):
    try:
        v = float(var.get())
        if min_val <= v <= max_val:
            return
    except ValueError:
        pass

    var.set(default)

def bind_float_range(entry, var, min_val, max_val, default):
    entry.bind(
        '<FocusOut>',
        lambda e: clamp_or_default(var, min_val, max_val, default)
    )

def make_styled_entry(parent, var=None, width=10, validate=None):
    entry = tk.Entry(
        parent,
        width=width,
        textvariable=var,
        bg='#2b2b2b',
        fg='#d4d4d4',
        insertbackground='white',
        relief='flat',
        highlightthickness=2,
        highlightbackground='#d4d4d4',
        highlightcolor='#4a6984'
    )
    
    if validate == 'int':
        vcmd = (parent.register(validate_int_partial), '%P')
        entry.configure(validate='key', validatecommand=vcmd)

    return entry

def make_slider_with_entry(
    parent,
    label,
    var,
    row,
    from_,
    to_,
    format_func=None,
    parse_func=None,
    snap_func=None,
    resolution=None,
):
    ttk.Label(parent, text=label).grid(row=row, column=0, sticky='w')
    entry = make_styled_entry(parent, width=8)
    entry.grid(row=row, column=2, padx=5)
    
    vcmd = (parent.register(validate_sci_float_partial), '%P')
    entry.config(validate='key', validatecommand=vcmd)

    def default_format(val):
        return str(val)

    def default_parse(text):
        return float(text)

    def apply_snap_and_clamp(val):
        if snap_func:
            val = snap_func(val)
        return min(max(val, from_), to_)

    def format_value(*_):
        try:
            val = var.get()
            val = apply_snap_and_clamp(val)
            var.set(val)  # This ensures it's always in range
            formatted = (format_func or default_format)(val)
            if entry.get() != formatted:
                entry.delete(0, tk.END)
                entry.insert(0, formatted)
        except tk.TclError:
            pass  # Variable might be empty/unset temporarily

    var.trace_add('write', format_value)

    def on_entry_change(event):
        try:
            val = (parse_func or default_parse)(entry.get())
            val = apply_snap_and_clamp(val)
            var.set(val)
        except (ValueError, tk.TclError):
            pass

    entry.bind('<FocusOut>', on_entry_change)
    entry.bind('<Return>', on_entry_change)

    scale = ttk.Scale(parent, from_=from_, to=to_, variable=var)
    if resolution:
        scale.config(resolution=resolution)
    scale.grid(row=row, column=1)
    format_value()
    return entry

def build_file_input_row(parent, label, row, var_attr, browse_command, app):
    ttk.Label(parent, text=label).grid(row=row, column=0, sticky='w')
    entry = make_styled_entry(parent, width=45)
    setattr(app, var_attr, entry)
    entry.grid(row=row, column=1)

    def wrapped_browse():
        browse_command(on_update=lambda _: update_fit_button_state(app))

    ttk.Button(parent, text='Browse', command=wrapped_browse).grid(row=row, column=2)

def update_fit_button_state(app):
    # ---- Enable Fit Button only if all required fields have values ----
    paths = [
        app.input_entry.get().strip(),
        app.mask_entry.get().strip(),
        app.save_entry.get().strip()
    ]

    i = 1
    while hasattr(app, f'input_entry_{i}') and hasattr(app, f'mask_entry_{i}'):
        paths.extend([
            getattr(app, f'input_entry_{i}').get().strip(),
            getattr(app, f'mask_entry_{i}').get().strip(),
        ])
        i += 1

    all_ok = all(os.path.isdir(p) for p in paths)
    app.fit_button.config(state='normal' if all_ok else 'disabled')
