# BONe_DLFit/core/data_preprocess.py
import numpy as np
import os
import random
import tifffile
import torch
from torch.utils.data import Dataset
from BONe_utils.utils import log_to_console, getarray

class ImageTilesDataset(Dataset):
# ---- works for 2D and 2.5D data ---- 
    def __init__(
        self,
        input_mask_pairs,
        transform=False, 
        patch_size=(256, 256), 
        num_patches_per_tile=1, 
        max_overlap=0.0,
        in_channels=1,
        stride=None,
        norm=None,
    ):
        assert in_channels % 2 == 1, 'in_channels must be odd for 2.5D mode' 
        self.input_mask_pairs = input_mask_pairs
        self.transform = transform
        self.patch_size = patch_size
        self.num_patches_per_tile = num_patches_per_tile
        self.max_overlap = max_overlap
        self.in_channels = in_channels
        self.half_span = in_channels // 2  # Number of tiles before and after the center
        self.norm = norm
        if stride is None or stride == in_channels:
            self.stride = 1
        else:
            self.stride = stride

        # ---- Precompute valid center indices for all volumes ---- 
        self.input_vols =[]
        self.mask_vols = []
        self.valid_indices = []
        for vol_idx, (input_path, mask_path, _, _, _) in enumerate(self.input_mask_pairs):
            input_vol, _ = getarray(input_path)
            mask_vol, _ = getarray(mask_path)
            assert input_vol.shape == mask_vol.shape, f'Shape mismatch in volume {vol_idx}'
            self.input_vols.append(input_vol)
            self.mask_vols.append(mask_vol)

            depth = input_vol.shape[0]
            centers = list(range(self.half_span, depth - self.half_span, self.stride))
            for center in centers:
                for patch_idx in range(self.num_patches_per_tile):
                    self.valid_indices.append((vol_idx, center))

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        vol_idx, center_slice_idx = self.valid_indices[idx]
        input_path, mask_path, vol_param1, vol_param2, _ = self.input_mask_pairs[vol_idx]

        # ---- Dynamically fetch the volume by name in the worker ---- 
        input_vol = self.input_vols[vol_idx]
        mask_vol = self.mask_vols[vol_idx]

        # ---- Assemble image tiles (C, H, W) for 2D and 2.5D cases ---- 
        image = input_vol[
            center_slice_idx - self.half_span : center_slice_idx + self.half_span + 1
        ].astype(np.float32)
        mask = mask_vol[center_slice_idx] # Single mask for central slice (H,W)

        # ---- Apply random geometric transformation if transform is True ---- 
        if self.transform:
            image, mask = self.random_augmentation(image, mask)

        # ---- Extract patches ---- 
        patches_image, patches_mask = self.random_crop(
            image,
            mask,
            self.num_patches_per_tile,
            self.max_overlap,
            vol_param1,
            vol_param2,
            max_retries=100,
            norm_mode=self.norm,
        )

        # ---- Apply random intensity augmentation if transform is True ---- 
        if self.transform:
            patches_image = self.intensity_augment(patches_image)

        # ---- Convert patches to tensor - image (B, C, H, W), mask (B, H, W) ---- 
        patches_image = torch.tensor(patches_image.astype(np.float32))
        patches_mask = torch.tensor(patches_mask, dtype=torch.long)

        return patches_image, patches_mask

    def random_augmentation(self, image, mask):
        # ---------------------------------------------------------------------------------
        # (C, H, W) for image, (H, W) for mask
        # Define the augmentation probabilities
        # ---------------------------------------------------------------------------------
        augmentations = [
            ('flip_horiz', 0.166),
            ('flip_vert', 0.166),
            ('rotate', 0.5),
            ('none', 0.168)
        ]

        # ---- Choose an augmentation based on the given probabilities ---- 
        augmentation_type = random.choices(
            [aug[0] for aug in augmentations],
            [aug[1] for aug in augmentations],
            k=1
        )[0]

        # ---- Apply the chosen augmentation ---- 
        if augmentation_type == 'flip_horiz':
            image = np.flip(image, axis=2)  # Flip horizontally (along width)
            mask = np.flip(mask, axis=1)    # Flip horizontally (along width)
        elif augmentation_type == 'flip_vert':
            image = np.flip(image, axis=1)  # Flip vertically (along height)
            mask = np.flip(mask, axis=0)    # Flip vertically (along height)
        elif augmentation_type == 'rotate':
            rotation_times = random.choice([1, 2, 3])  # 90, 180, or 270 degrees
            image = np.rot90(image, k=rotation_times, axes=[1, 2])  # Rotate along height and width
            mask = np.rot90(mask, k=rotation_times, axes=[0, 1])    # Rotate along height and width

        return image, mask

    def random_crop(
        self, 
        image, 
        mask, 
        num_patches, 
        max_overlap=0.3, 
        vol_param1=None, 
        vol_param2=None, 
        max_retries=100,
        norm_mode=None,
    ):
        # ---------------------------------------------------------------------------------
        # Get image and mask dimensions
        # image: (C, H, W), mask: (H, W)
        # ---------------------------------------------------------------------------------
        _, img_height, img_width = image.shape
        patch_width, patch_height = self.patch_size

        if img_width < patch_width or img_height < patch_height:
            raise ValueError(f'Image size {image.shape} is smaller than patch size {self.patch_size}')

        patches_image = []
        patches_mask = []
        top_lefts = []

        for _ in range(num_patches):
            retries = 0
            while retries < max_retries:
                # ---- Randomly select top-left corner coordinates ---- 
                top = random.randint(0, img_height - patch_height)
                left = random.randint(0, img_width - patch_width)

                # ---- Check overlap with previous patches ---- 
                overlap_found = False
                for prev_top, prev_left in top_lefts:
                    # ---- Calculate overlap area ---- 
                    overlap_height = max(0, min(top + patch_height, prev_top + patch_height) - max(top, prev_top))
                    overlap_width = max(0, min(left + patch_width, prev_left + patch_width) - max(left, prev_left))
                    overlap_area = overlap_height * overlap_width
                    patch_area = patch_width * patch_height

                    # ---- If the overlap is more than the allowed threshold, skip this patch ---- 
                    if overlap_area / patch_area > max_overlap:
                        overlap_found = True
                        break

                if not overlap_found:
                    # ---- No overlap issue, accept this patch ---- 
                    top_lefts.append((top, left))
                    break

                retries += 1

            # ---- If we've reached max retries, just accept the patch regardless of overlap ---- 
            if retries == max_retries:
                top_lefts.append((top, left))

            # ---- Slice the patches directly from the tensors using advanced indexing ---- 
            image_patch = image[:, top:top + patch_height, left:left + patch_width]
            mask_patch = mask[top:top + patch_height, left:left + patch_width]

            patches_image.append(image_patch)
            patches_mask.append(mask_patch)

        # ---- Stack all patches into a batch dimension ---- 
        patches_image = np.stack(patches_image)  # (B, C, H, W)
        patches_mask = np.stack(patches_mask)    # (B, H, W)

        # ---- Normalize after stacking (for all patches at once) ---- 
        if norm_mode == 'ZScore':
            patches_image = (patches_image - vol_param1) / (vol_param2 + 1e-8)
        elif norm_mode == 'minMax':
            patches_image = (patches_image - vol_param1) / (vol_param2 - vol_param1 + 1e-8)
        else:
            raise ValueError(f'Unknown normalization method: {norm_mode}')

        del top_lefts, image, mask

        return patches_image, patches_mask

    def intensity_augment(self, patches_image):
        clip_min, clip_max = {
            'ZScore': (-5.0, 5.0),
            'minMax': (0.0, 1.0),
        }.get(self.norm, (-np.inf, np.inf))

        if self.norm not in ('ZScore', 'minMax'):
            log_to_console(f'Unknown normalization mode. No clipping will be applied.', self.log_file_txt)

        # ---- Randomly decide whether to apply shift and/or scale ---- 
        apply_shift = np.random.rand() < 0.4
        apply_scale = np.random.rand() < 0.4

        # ---- Generate one random shift and scale value for the whole batch ---- 
        shift_val = np.random.uniform(-0.15, 0.15)
        scale_val = np.random.uniform(0.75, 1.25)

        if apply_shift:
            patches_image += shift_val

        if apply_scale:
            patches_image *= scale_val

        # ---- Clip the values ---- 
        np.clip(patches_image, clip_min, clip_max, out=patches_image)

        return patches_image

class Image3dTilesDataset(Dataset):
    def __init__(
        self, 
        input_mask_pairs,
        transform=False,
        tile_depth=32,
        patch_size=(256, 256, 32),
        num_patches_per_tile=1, 
        max_overlap=0.0,
        norm=None,
    ):
        self.input_mask_pairs = input_mask_pairs
        self.transform = transform
        self.tile_depth = tile_depth
        self.patch_size = patch_size
        self.num_patches_per_tile = num_patches_per_tile
        self.max_overlap = max_overlap
        self.norm = norm

        # ---- Cache volumes and store valid indices for each tile/patch ----
        self.input_vols = []
        self.mask_vols = []
        self.valid_indices = []

        # ---- Precompute valid indices for all volumes ----
        for vol_idx, (input_path, mask_path, *_) in enumerate(self.input_mask_pairs):
            # ---- Load mask to query depth ----
            image, _ = getarray(input_path)
            mask, _ = getarray(mask_path)

            # Convert (D, H, W) -> (H, W, D) and store
            image = image.transpose(1, 2, 0)
            mask = mask.transpose(1, 2, 0)
            self.input_vols.append(image)
            self.mask_vols.append(mask)
            
            # ---- Query depth (H, W, D) ----
            depth = image.shape[2]

            # ---- Pre-padded on load; final check that depth is divisible by tile_depth ----
            if depth % self.tile_depth != 0:
                raise ValueError(
                    f'Volume depth {depth} is not divisible by tile_depth {self.tile_depth}. '
                )

            # ---- Compute number of tiles of tile_depth fit in volume ---- 
            num_tiles = depth // self.tile_depth
            
            # ---- For each tile, precompute the valid patch indices ----
            for tile_idx in range(num_tiles):
                for patch_idx in range(self.num_patches_per_tile):
                    self.valid_indices.append((vol_idx, tile_idx))
                    
        # ---- Log the number of precomputed valid indices (total number of patches) ----
        self.total_patches = len(self.valid_indices)
        
    def __len__(self):
        return self.total_patches

    def __getitem__(self, idx):
        # ---- Get the volume, tile, and patch index from the precomputed list ---- 
        vol_idx, tile_idx = self.valid_indices[idx]

        # ---- Load padded scan, vol_param1, and vol_param2 ----
        input_path, mask_path, vol_param1, vol_param2, _ = self.input_mask_pairs[vol_idx]
        image = self.input_vols[vol_idx]
        mask = self.mask_vols[vol_idx]
        
        # ---- Extract tile (smp3d uses dimension order H,W,D) ---- 
        slice_start = tile_idx * self.tile_depth
        slice_end = slice_start + self.tile_depth

        tile_image = image[:, :, slice_start:slice_end].astype(np.float32)
        tile_mask = mask[:, :, slice_start:slice_end]

        # ---- Apply random geometric transformation if enabled ---- 
        if self.transform:
            tile_image, tile_mask = self.random_augmentation(tile_image, tile_mask)

        # ---- Extract patches from the tile ---- 
        patches_image, patches_mask = self.random_crop(
            tile_image, 
            tile_mask, 
            self.num_patches_per_tile, 
            self.max_overlap,
            vol_param1,
            vol_param2,
            max_retries=100,
            norm_mode=self.norm,
        )

        # ---- Apply random intensity augmentation if transform is True ---- 
        if self.transform:
            patches_image = self.intensity_augment(patches_image)

        # ---- Convert image and mask to PyTorch tensors ---- 
        patches_image = torch.tensor(patches_image.astype(np.float32))
        patches_mask = torch.tensor(patches_mask, dtype=torch.long)

        # ---- Add channel dimension (B, 1, H, W, D) for each patch ---- 
        patches_image = patches_image.unsqueeze(1)

        return patches_image, patches_mask

    def random_augmentation(self, image, mask):
        # ---- Define the augmentation probabilities ---- 
        augmentations = [
            ('flip_horiz', 0.111),
            ('flip_vert', 0.111),
            ('flip_depth', 0.111),
            ('rotate', 0.5),
            ('none', 0.167)
        ]

        # ---- Choose an augmentation based on the given probabilities ---- 
        augmentation_type = random.choices(
            [aug[0] for aug in augmentations],
            [aug[1] for aug in augmentations],
            k=1
        )[0]

        # ---- Apply the chosen augmentation (dim: H,W,D) ---- 
        if augmentation_type == 'flip_vert':
            image = np.flip(image, axis=0)  # Flip vertically (along height)
            mask = np.flip(mask, axis=0)    # Flip vertically (along height)
        elif augmentation_type == 'flip_horiz':
            image = np.flip(image, axis=1)  # Flip horizontally (along width)
            mask = np.flip(mask, axis=1)    # Flip horizontally (along width)
        elif augmentation_type == 'flip_depth':
            image = np.flip(image, axis=2)  # Flip horizontally (along depth)
            mask = np.flip(mask, axis=2)    # Flip horizontally (along depth)
        elif augmentation_type == 'rotate':
            rotation_times = random.choice([1, 2, 3])  # 90, 180, or 270 degrees
            image = np.rot90(image, k=rotation_times, axes=[0, 1])  # Rotate along H and W
            mask = np.rot90(mask, k=rotation_times, axes=[0, 1])    # Rotate along H and W

        return image, mask

    def random_crop(
        self,
        image,
        mask,
        num_patches,
        max_overlap=0.3,
        vol_param1=None,
        vol_param2=None,
        max_retries=100,
        norm_mode=None,
    ):
        # ---- Get image and mask dimensions ---- 
        img_height, img_width, img_depth = image.shape
        patch_height, patch_width, patch_depth = self.patch_size

        if img_width < patch_width or img_height < patch_height or img_depth < patch_depth:
            raise ValueError(f'Tile size {image.shape} is smaller than patch size {self.patch_size}')

        patches_image = []
        patches_mask = []
        top_left_fronts = []

        for _ in range(num_patches):
            retries = 0
            while retries < max_retries:
                # Randomly select top-left corner coordinates
                top = random.randint(0, img_height - patch_height)
                left = random.randint(0, img_width - patch_width)
                front = random.randint(0, img_depth - patch_depth)

                # Check overlap with previous patches
                overlap_found = False
                for prev_top, prev_left, prev_front in top_left_fronts:
                    # Calculate 3D overlap area
                    overlap_height = max(
                        0,
                        min(top + patch_height, prev_top + patch_height) - max(top, prev_top)
                    )
                    overlap_width = max(
                        0,
                        min(left + patch_width, prev_left + patch_width) - max(left, prev_left)
                    )
                    overlap_depth = max(
                        0,
                        min(front + patch_depth, prev_front + patch_depth) - max(front, prev_front)
                    )
                    overlap_volume = overlap_height * overlap_width * overlap_depth
                    patch_volume = patch_width * patch_height * patch_depth

                    # If the overlap is more than the allowed threshold, skip this patch
                    if overlap_volume / patch_volume > max_overlap:
                        overlap_found = True
                        break

                if not overlap_found:
                    # No overlap issue, accept this patch
                    top_left_fronts.append((top, left, front))
                    break

                retries += 1

            # ---- If we've reached max retries, just accept the patch regardless of overlap ---- 
            if retries == max_retries:
                top_left_fronts.append((top, left, front))

            # ---- Slice the patches directly from numpy arrays ----
            image_patch = image[
                top:top + patch_height, 
                left:left + patch_width, 
                front:front + patch_depth
            ]
            mask_patch = mask[
                top:top + patch_height, 
                left:left + patch_width,
                front:front + patch_depth
            ]

            patches_image.append(image_patch)
            patches_mask.append(mask_patch)

        # ---- Stack patches (num_patches, patch_height, patch_width, patch_depth) ---- 
        patches_image = np.stack(patches_image)
        patches_mask = np.stack(patches_mask)

        # ---- Normalize (for all patches at once) ---- 
        if norm_mode == 'ZScore':
            patches_image = (patches_image - vol_param1) / (vol_param2 + 1e-8)
        elif norm_mode == 'minMax':
            patches_image = (patches_image - vol_param1) / (vol_param2 - vol_param1 + 1e-8)
        else:
            raise ValueError(f'Unknown normalization method: {norm_mode}')

        del top_left_fronts, image, mask

        return patches_image, patches_mask

    def intensity_augment(self, patches_image):
        clip_min, clip_max = {
            'ZScore': (-5.0, 5.0),
            'minMax': (0.0, 1.0),
        }.get(self.norm, (-np.inf, np.inf))

        if self.norm not in ('ZScore', 'minMax'):
            log_to_console(f'Unknown normalization mode. No clipping will be applied.', self.log_file_txt)

        # Randomly decide whether to apply shift and/or scale
        apply_shift = np.random.rand() < 0.4
        apply_scale = np.random.rand() < 0.4

        # Generate one random shift and scale value for the whole batch
        shift_val = np.random.uniform(-0.15, 0.15)
        scale_val = np.random.uniform(0.75, 1.25)

        if apply_shift:
            patches_image += shift_val

        if apply_scale:
            patches_image *= scale_val

        # Clip the values
        np.clip(patches_image, clip_min, clip_max, out=patches_image)

        return patches_image
