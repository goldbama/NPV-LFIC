# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import re
import numpy as np
from PIL import Image

class PNGReader():
    def __init__(self, src_folder, width, height):
        self.src_folder = src_folder
        self.width = width
        self.height = height

        self.png_files = sorted([f for f in os.listdir(src_folder) if f.lower().endswith('.png')])

        if not self.png_files:
            raise ValueError(f'No PNG files found in {src_folder}')

        first_file = self.png_files[0]
        if re.match(r'^\d{3}_\d{3}\.png$', first_file):
            # 000_000.png, 001_007.png, ...
            self.file_format = 'coordinate'
            print(f"Detected coordinate format: {first_file}")

            self.png_files.sort(key=lambda x: (
                int(x.split('_')[0]),
                int(x.split('_')[1].split('.')[0])
            ))
        elif 'im1.png' in self.png_files:
            self.file_format = 'im_padding_1'
        elif 'im00001.png' in self.png_files:
            self.file_format = 'im_padding_5'
        else:

            self.file_format = 'unknown'
            print(f"Unknown file naming convention, using alphabetical order. First file: {first_file}")

        self.current_file_index = 0
        self.eof = False
        self.image_cache = {}

    def get_all_image_files(self):
        return self.png_files

    def read_frame_by_name(self, filename):
        filepath = os.path.join(self.src_folder, filename)
        if not os.path.exists(filepath):
            return None

        rgb = Image.open(filepath).convert('RGB')
        rgb = np.asarray(rgb).astype('float32').transpose(2, 0, 1)
        rgb = rgb / 255.
        _, height, width = rgb.shape

        if height != self.height or width != self.width:
            raise ValueError(
                f'Image size mismatch: expected ({self.height}, {self.width}), got ({height}, {width}) for {filename}')

        return rgb

    def read_one_frame(self, src_format="rgb"):
        def _none_exist_frame():
            if src_format == "rgb":
                return None
            return None, None, None

        if self.eof or self.current_file_index >= len(self.png_files):
            self.eof = True
            return _none_exist_frame()

        current_file = self.png_files[self.current_file_index]
        filepath = os.path.join(self.src_folder, current_file)

        rgb = Image.open(filepath).convert('RGB')
        rgb = np.asarray(rgb).astype('float32').transpose(2, 0, 1)
        rgb = rgb / 255.
        _, height, width = rgb.shape

        if height != self.height or width != self.width:
            raise ValueError(f'Image size mismatch: expected ({self.height}, {self.width}), got ({height}, {width})')

        self.current_file_index += 1

        return rgb

    def read_frame_by_index(self, index):
        if index < 0 or index >= len(self.png_files):
            return None

        filename = self.png_files[index]
        return self.read_frame_by_name(filename)

    def get_frame_count(self):
        return len(self.png_files)

    def reset(self):
        self.current_file_index = 0
        self.eof = False

    def close(self):
        self.reset()