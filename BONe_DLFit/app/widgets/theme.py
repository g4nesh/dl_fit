# BONe_DLFit/app/widgets/theme.py
from tkinter import ttk

# ------------------------
# Style + Theme Setup
# ------------------------
def build_theme(root):
    style = ttk.Style(root)
    style.theme_use('clam')

    # ---- Auto-detect screen resolution ---- 
    try:
        dpi = root.winfo_fpixels('1i')
        scale = dpi / 72
        scale = min(max(scale, 0.75), 1.5)
        root.tk.call('tk', 'scaling', scale)
    except Exception:
        root.tk.call('tk', 'scaling', 1.0)  # fallback

    bg = '#1e1e1e'; fg = '#d4d4d4'; fbg = '#2b2b2b'
    style.configure('.', background=bg, foreground=fg, fieldbackground=fbg)
    style.configure('TButton', background='#3c3c3c', foreground='#ffffff')
    style.map('TButton', 
        background=[
            ('active', '#5c5c5c'),
            ('disabled', '#2b2b2b')
        ], 
        foreground=[
            ('active', '#ffffff'),
            ('disabled', '#777777')
        ]
    )
    style.map('TRadiobutton', 
        background=[
            ('active', bg),
            ('disabled', bg)
        ], 
        foreground=[
            ('active', '#ffffff'),
            ('disabled', '#777777')
        ]
    )

    style.configure('Vertical.TScrollbar',
        gripcount=0,
        background='#444444',
        troughcolor='#2b2b2b',
        bordercolor='#2b2b2b',
        arrowcolor='#d4d4d4'
    )

    style.map('Vertical.TScrollbar',
        background=[('active', '#666666')],
        arrowcolor=[('active', '#ffffff')]
    )

    style.configure('TCombobox',
        arrowcolor=fg,
        foreground=fg,
        background=bg
    )
    style.map('TCombobox',
        fieldbackground=[('readonly', fbg)],
        foreground=[('readonly', fg)],
        background=[('readonly', bg)]
    )

    style.configure('TProgressbar', background='green')

    root.configure(bg=bg)
    root.geometry('+0+10') # Start GUI close to the upper-left corner

    # Return the colors for external use
    return {'bg': bg, 'fg': fg, 'field_bg': fbg}
