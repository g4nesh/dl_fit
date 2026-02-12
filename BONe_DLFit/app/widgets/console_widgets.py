# BONe_DLFit/app/widgets/console_widgets.py
import tkinter as tk
from tkinter import ttk

# ------------------------
# UI Sections
# ------------------------
# ---- Console window ----
def build_console_section(parent, app):
    parent.grid_rowconfigure(0, weight=1)
    parent.grid_columnconfigure(0, weight=1)
    app.console_text = tk.Text(parent, height=10, bg='#252526', fg='#d4d4d4', wrap=tk.WORD)
    app.console_text.grid(row=0, column=0, sticky='nsew')

# ---- Setup console queueing to be DDP-safe ----
def poll_console(app, interval=100):
    try:
        from BONe_utils.utils import get_console_queue
        
        # Get the queue (lazy initialization happens here)
        console_queue = get_console_queue()
        
        # Process all available messages
        messages_processed = 0
        max_messages_per_poll = 50  # Prevent UI freezing
        
        while not console_queue.empty() and messages_processed < max_messages_per_poll:
            try:
                message = console_queue.get_nowait()
                if hasattr(app, 'console_text'):
                    app.console_text.insert('end', message + '\n')
                    app.console_text.see('end')
                    app.console_text.update_idletasks()
                messages_processed += 1
            except Exception as e:
                if 'Empty' not in str(e):
                    print(f'[Console Message Error] {e}')
                break # Queue is empty or error occurred
    
    except Exception as e:
        if 'Empty' not in str(e):
            print(f'[Console Polling Error] {e}')

    # Schedule the next poll (GUI-safe)
    if hasattr(app, 'root') and app.root:
        app.root.after(interval, lambda: poll_console(app, interval))

# ---- Setup progressbar queueing to be DDP-safe ----
def poll_progress(app, interval=100):
    try:
        from BONe_utils.utils import get_progress_queue
        
        # Get the queue (lazy initialization happens here)
        progress_queue = get_progress_queue()
        
        # Get the most recent progress value
        latest_progress = None
        while not progress_queue.empty():
            try:
                latest_progress = progress_queue.get_nowait()
            except Exception:
                break  # Queue is empty
        
        # Update progress bar with latest value
        if latest_progress is not None and hasattr(app, 'progress') and app.progress:
            app.progress['value'] = latest_progress
            app.progress.update_idletasks()
    
    except Exception as e:
        if 'Empty' not in str(e):
            print(f'[Progress Polling Error] {e}')

    if hasattr(app, 'root') and app.root:
        app.root.after(interval, lambda: poll_progress(app, interval))

# ---- Progressbar ----
def build_progressbar_section(parent, app):
    parent.grid_rowconfigure(0, weight=1)
    parent.grid_columnconfigure(0, weight=1)
    app.progress = ttk.Progressbar(
        parent,
        style='TProgressbar',
        orient='horizontal',
        mode='determinate'
    )
    app.progress.grid(row=0, column=0, sticky='nsew')
