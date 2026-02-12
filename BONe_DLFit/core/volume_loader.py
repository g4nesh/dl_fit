# BONe_DLFit/core/volume_loader.py
import gc
import os
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from BONe_utils.utils import getarray, monitor_peak_ram
import psutil

# -----------------------------------------------------------------------------
# Process one volume at a time to minimize memory footprint
# -----------------------------------------------------------------------------
def compute_single_volume_stats(vol, mode='ZScore', chunk_depth=32):
    depth = vol.shape[0]
    
    if mode == 'ZScore':
        n_total = 0
        sum_total = 0.0
        sum_sq_total = 0.0
    elif mode == 'minMax':
        mn, mx = None, None
    else:
        raise ValueError(f'Unsupported mode {mode}')
    
    # ---- Process in chunks ----
    for z0 in range(0, depth, chunk_depth):
        z1 = min(z0 + chunk_depth, depth)
        chunk = vol[z0:z1].astype(np.float32, copy=False)
        mask = chunk > 0
        
        if not np.any(mask):
            continue
            
        v = chunk[mask].astype(np.float64)  # Avoid rounding errors

        if mode == 'ZScore':
            n_total += v.size
            sum_total += v.sum()
            sum_sq_total += (v**2).sum()
        elif mode == 'minMax':
            chunk_min, chunk_max = v.min(), v.max()
            mn = chunk_min if mn is None else min(mn, chunk_min)
            mx = chunk_max if mx is None else max(mx, chunk_max)
        
        # ---- Clean up chunk immediately ----
        del chunk, mask, v
   
    if mode == 'ZScore':
        if n_total == 0:
            return 0.0, 1.0
        mean = sum_total / n_total
        std = np.sqrt((sum_sq_total / n_total) - mean**2)
        return float(mean), float(std)
    elif mode == 'minMax':
        return float(mn), float(mx)

# -----------------------------------------------------------------------------
# Load ONE scan pair, compute stats, save to .npy, return metadata
# -----------------------------------------------------------------------------
def process_single_scan_pair(
    input_dir, 
    mask_dir, 
    norm_mode, 
    npy_dir, 
    tile_depth,
    patch_size,
    patch_depth,
    logger
):
    # ---- Extract name for .npy files ---- 
    parts = Path(input_dir).parts
    if parts[-1] == '_Scan' and len(parts) >= 2:
        scan_name = parts[-2]
    else:
        scan_name = Path(input_dir).stem
    
    input_npy_path = Path(npy_dir) / f'{scan_name}_input.npy'
    mask_npy_path = Path(npy_dir) / f'{scan_name}_mask.npy'
    
    try:
        # ---- Load input volume ---- 
        logger(f'[INFO] Loading {scan_name} input...')
        input_vol, _ = getarray(
            input_dir,
            save_npy=True,
            npy_dir=npy_dir,
            save_npy_suffix='_input',
            pad_depth_to=tile_depth,
        )
        
        # ---- Load mask volume ---- 
        logger(f'[INFO] Loading {scan_name} mask...')
        mask_vol, _ = getarray(
            mask_dir,
            save_npy=True,
            npy_dir=npy_dir,
            save_npy_suffix='_mask',
            pad_depth_to=tile_depth,
        )
        
        # ---- Validate shapes ---- 
        depth, H, W = input_vol.shape
        
        if patch_size and (patch_size > H or patch_size > W):
            max_valid_patch_size = (min(H, W) // 32) * 32
            raise ValueError(
                f'[{scan_name}] Volume too small for patch size {patch_size}.\n'
                f'Suggested max patch size: {max_valid_patch_size} px.')
        
        if patch_depth:
            if patch_depth > depth:
                max_valid_patch_depth = (depth // 32) * 32
                raise ValueError(
                    f'[{scan_name}] Patch depth ({patch_depth}) exceeds depth ({depth}).\n'
                    f'Reduce to 32–{max_valid_patch_depth} px.')
            
            if tile_depth and patch_depth > tile_depth:
                raise ValueError(
                    f'[{scan_name}] Patch depth ({patch_depth}) > tile depth ({tile_depth}).')
        
        if input_vol.shape != mask_vol.shape:
            raise ValueError(
                f'[{scan_name}] Shape mismatch:\n'
                f'  Input: {input_vol.shape}\n  Mask: {mask_vol.shape}')
        
        # ---- Get mask max value for num_classes ---- 
        mask_max = int(np.max(mask_vol))
        
        # ---- Compute stats ---- 
        logger(f'[INFO] Computing stats for {scan_name}...')
        stats = compute_single_volume_stats(input_vol, mode=norm_mode, chunk_depth=32)
        
        if norm_mode == 'ZScore':
            param1, param2 = stats[0], stats[1]  # mean, std
        elif norm_mode == 'minMax':
            param1, param2 = stats[0], stats[1]  # min, max
        else:
            raise ValueError(f'Unknown norm_mode: {norm_mode}')
        
        # ---- Create result tuple ---- 
        result = {
            'input_npy_path': str(input_npy_path),
            'mask_npy_path': str(mask_npy_path),
            'param1': float(param1),
            'param2': float(param2),
            'input_dir': str(input_dir),
            'mask_max': mask_max,
            'scan_name': scan_name
        }
        
        # ---- Delete volumes immediately after processing ---- 
        del input_vol, mask_vol
        gc.collect()
        
        logger(f'[INFO] Completed {scan_name}')
        return result
        
    except Exception as e:
        logger(f'[ERROR] Failed to process {scan_name}: {e}')
        raise

# -----------------------------------------------------------------------------
# Parallel streaming several scans at a time
# -----------------------------------------------------------------------------
def load_volumes_and_stats_streaming(
    input_dirs, 
    mask_dirs, 
    norm_mode='ZScore',
    patch_size=None, 
    patch_depth=None, 
    tile_depth=None, 
    logger=print, 
    npy_dir=None,
    max_parallel=2
):
    if not npy_dir:
        raise ValueError('npy_dir must be specified')
    
    npy_dir = Path(npy_dir)
    npy_dir.mkdir(parents=True, exist_ok=True)
    
    # ---- Clean up old .npy files ---- 
    old_npy_files = list(npy_dir.glob('*_input.npy')) + list(npy_dir.glob('*_mask.npy'))
    if old_npy_files:
        logger(f'[INFO] Cleaning up {len(old_npy_files)} old .npy files...')
        for old_file in old_npy_files:
            try:
                old_file.unlink()
            except Exception as e:
                logger(f'[WARN] Could not remove {old_file.name}: {e}')
    
    # ---- Auto-tune max_parallel based on available RAM ----
    available_ram_gb = psutil.virtual_memory().available / 1e9
    estimated_safe_parallel = max(1, int((available_ram_gb - 100) / 25)) # Assume scan ~25 GB
    max_parallel = min(max_parallel, estimated_safe_parallel)
    
    # ---- Start RAM monitoring ---- 
    monitoring, mon_thread, get_peak_ram = monitor_peak_ram()
    
    try:
        results = []
        mask_max_values = []
        
        # ---- Process scans with limited parallelism using ThreadPoolExecutor ----
        with ThreadPoolExecutor(max_workers=max_parallel) as executor:
            # Submit all tasks
            future_to_dir = {}
            for input_dir, mask_dir in zip(input_dirs, mask_dirs):
                future = executor.submit(
                    process_single_scan_pair,
                    input_dir=input_dir,
                    mask_dir=mask_dir,
                    norm_mode=norm_mode,
                    npy_dir=npy_dir,
                    tile_depth=tile_depth,
                    patch_size=patch_size,
                    patch_depth=patch_depth,
                    logger=logger
                )
                future_to_dir[future] = input_dir
            
            # Collect results upon completion
            completed = 0
            for future in as_completed(future_to_dir):
                input_dir = future_to_dir[future]
                try:
                    result = future.result()
                    results.append(result)
                    mask_max_values.append(result['mask_max'])
                    
                    completed += 1
                    logger(f'[INFO] Progress: {completed}/{len(input_dirs)} scans completed')
                    
                    # ---- Force cleanup after each completion ---- 
                    gc.collect()
                    
                except Exception as e:
                    logger(f'[ERROR] Failed to process {input_dir}: {e}')
                    raise

        # ---- Stop RAM monitoring ----
        monitoring[0] = False
        mon_thread.join()
        peak_ram = get_peak_ram()
        
        # ---- Check mask consistency ---- 
        unique_max_vals = set(mask_max_values)
        if len(unique_max_vals) > 1:
            logger('[ERROR] Inconsistent max pixel values across mask volumes:')
            for i, val in enumerate(mask_max_values, start=1):
                logger(f'  mask{i}: max value = {val}')
            raise ValueError('Inconsistent mask class counts.')
        
        num_classes = int(mask_max_values[0]) + 1
        logger(f'[INFO] All masks consistent. num_classes = {num_classes}')
        
        # ---- Sort results to maintain original order ---- 
        dir_to_result = {r['input_dir']: r for r in results}
        sorted_results = []
        for d in input_dirs:
            d_str = str(d)
            if d_str in dir_to_result:
                sorted_results.append(dir_to_result[d_str])
            else:
                raise ValueError(f'Missing result for input_dir: {d_str}')
        
        # ---- Prepare results for trainer.py ---- 
        input_mask_pairs = [
            (
                r['input_npy_path'],
                r['mask_npy_path'],
                r['param1'],
                r['param2'],
                r['input_dir']
            )
            for r in sorted_results
        ]
        
        return input_mask_pairs, peak_ram, num_classes
        
    except Exception as e:
        monitoring[0] = False
        logger(f'[ERROR] Streaming loader failed: {e}')
        raise
