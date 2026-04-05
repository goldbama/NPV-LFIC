import os
import sys
import logging
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights

# =========================================================
# 1. Configuration
# =========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
LOG_FILE = os.path.join(CHECKPOINT_DIR, "test_run.log")
LOAD_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "checkpoint_best_31.pth")

TEST_DATASETS = [
    {
        'csv': os.path.join(BASE_DIR, 'csv', 'Best_HCI_Configs.csv'),
        'root': os.path.join(BASE_DIR, 'png', 'HCI')
    },
    {
        'csv': os.path.join(BASE_DIR, 'csv', 'Best_EPFL_Configs.csv'),
        'root': os.path.join(BASE_DIR, 'png', 'EPFL')
    },
]

IMAGE_SIZE = 416
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ALL_CONFIGS = ['HCI_5', 'HCI_9', 'HCI_13', 'HCI_25', 'HCI_41', 'HCI_81']
CONFIG_TO_ID = {cfg: i for i, cfg in enumerate(ALL_CONFIGS)}
ID_TO_CONFIG = {i: cfg for cfg, i in CONFIG_TO_ID.items()}
RATE_TO_ID = {'Q1': 0, 'Q2': 1, 'Q3': 2, 'Q4': 3}
NUM_RATES = 4
VIEW_FILES = ['004_004.png', '000_004.png', '008_004.png', '004_000.png', '004_008.png']

# =========================================================
# 2. Log
# =========================================================
def setup_logger(log_file):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if logger.hasHandlers(): return logger
    fh = logging.FileHandler(log_file, mode='a')
    fh.setFormatter(logging.Formatter("[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    logger.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter("[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    logger.addHandler(sh)
    return logger

logger = setup_logger(LOG_FILE)

# =========================================================
# 3. Tools
# =========================================================
def load_csv_robust(path):
    for enc in ['utf-8', 'gbk', 'gb18030', 'latin1']:
        try:
            return pd.read_csv(path, encoding=enc)
        except:
            continue
    raise RuntimeError(f"Cannot load {path}")

def prepare_dataframe(dataset_config_list, is_test=True):
    all_dfs = []

    for item in dataset_config_list:
        csv_path = item['csv']
        root_path = item['root']
        if not os.path.exists(csv_path):
            logger.warning(f"File not found: {csv_path}")
            continue

        try:
            df = load_csv_robust(csv_path)
            scene_name_col = None
            possible_names = ['Scene_Name', 'Scene', 'Name', 'scene_name', 'scene', 'name']
            for col in possible_names:
                if col in df.columns:
                    scene_name_col = col
                    break

            if scene_name_col is None:
                scene_name_col = df.columns[0]
                logger.warning(f"No standard scene name column found in {csv_path}, using {scene_name_col}")

            df = df.set_index(scene_name_col)
            df = df.reset_index().melt(
                id_vars=scene_name_col, var_name='Rate_Point', value_name='Best_Config'
            )
            df = df.rename(columns={scene_name_col: 'Scene_Name'})
            df = df.dropna(subset=['Best_Config'])
            df = df[df['Rate_Point'].isin(RATE_TO_ID.keys())]
            df['Original_Config'] = df['Best_Config']
            df['Best_Config'] = df['Best_Config'].apply(
                lambda x: f"HCI_{x.split('_')[-1]}"
            )
            df = df[df['Best_Config'].isin(ALL_CONFIGS)]
            df['img_root'] = root_path

            csv_path_lower = csv_path.lower()
            root_path_lower = root_path.lower()

            if 'hci_old' in csv_path_lower or 'hci_old' in root_path_lower:
                df['dataset'] = 'HCI_old'
                df['dataset_prefix'] = 'HCI_old'
            elif 'hci' in csv_path_lower:
                df['dataset'] = 'HCI'
                if 'train' in csv_path_lower or 'train' in root_path_lower:
                    df['dataset_prefix'] = 'HCI_train'
                else:
                    df['dataset_prefix'] = 'HCI'
            elif 'epfl' in csv_path_lower:
                df['dataset'] = 'EPFL'
                if 'train' in csv_path_lower or 'train' in root_path_lower:
                    df['dataset_prefix'] = 'EPFL_train'
                else:
                    df['dataset_prefix'] = 'EPFL'
            elif 'pinet' in csv_path_lower:
                df['dataset'] = 'PINET'
                df['dataset_prefix'] = 'PINET'
            else:
                df['dataset'] = 'Unknown'
                df['dataset_prefix'] = 'Unknown'

            if len(df) > 0:
                all_dfs.append(df)
                logger.info(f"Successfully loaded {len(df)} samples from {csv_path}")
        except Exception as e:
            logger.error(f"Error processing {csv_path}: {e}")
            continue

    return pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()

# =========================================================
# 4. Model
# =========================================================
class FeatureExtractor(nn.Module):
    def __init__(self, in_channels=3):
        super(FeatureExtractor, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, 3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, 3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.avgpool(x)
        return x.view(x.size(0), -1)

class LFParallaxNet(nn.Module):
    def __init__(self, num_classes=6, num_rates=4, input_channels=15):
        super().__init__()
        self.backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        old_conv = self.backbone.conv1
        new_conv = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            w_new = old_conv.weight.repeat(1, input_channels // 3, 1, 1) / (input_channels // 3)
            new_conv.weight.copy_(w_new)
        self.backbone.conv1 = new_conv
        feat_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.rate_embed = nn.Embedding(num_rates, 32)
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim + 32 + 256 + 256 + 1, 256), nn.BatchNorm1d(256), nn.ReLU(inplace=True),
            nn.Dropout(0.4), nn.Linear(256, num_classes)
        )
        self.texture_branch = FeatureExtractor(in_channels=3)
        self.geometry_branch = FeatureExtractor(in_channels=6)

    def forward(self, x, center_img, diff_img, disparity_scalar, rate_idx):
        features = self.backbone(x)
        features_tex = self.texture_branch(center_img)
        features_geo = self.geometry_branch(diff_img)
        rate_vec = self.rate_embed(rate_idx)
        feat_scalar = disparity_scalar.view(-1, 1)
        return self.classifier(torch.cat([features, features_tex, features_geo, rate_vec, feat_scalar], dim=1))

# =========================================================
# 5. Dataset
# =========================================================
class LFMultiViewTestDataset(Dataset):
    def __init__(self, df, crop_size=512):
        self.samples = df
        self.crop_size = crop_size
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        row = self.samples.iloc[idx]
        current_root = row['img_root']
        scene_name = row['Scene_Name']

        raw_imgs = []
        for f in VIEW_FILES:
            p = os.path.join(current_root, scene_name, f)
            try:
                img = Image.open(p).convert("RGB")
                raw_imgs.append(img)
            except Exception as e:
                logger.error(f"Error loading {p}: {e}")

        w, h = raw_imgs[0].size
        i = (h - self.crop_size) // 2
        j = (w - self.crop_size) // 2
        crop_box = (j, i, j + self.crop_size, i + self.crop_size)
        imgs = [img.crop(crop_box) for img in raw_imgs]

        transformed_imgs = [self.transform(img) for img in imgs]

        center_input = transformed_imgs[0]
        diff_h = torch.abs(transformed_imgs[3] - transformed_imgs[4])
        diff_v = torch.abs(transformed_imgs[1] - transformed_imgs[2])
        diff_input = torch.cat([diff_h, diff_v], dim=0)
        scalar_score = torch.mean(diff_input)

        x = torch.cat(transformed_imgs, dim=0)
        return x, center_input, diff_input, scalar_score, torch.tensor(RATE_TO_ID[row['Rate_Point']]), torch.tensor(CONFIG_TO_ID[row['Best_Config']])

# =========================================================
# 6. Test
# =========================================================
def run_test():
    logger.info(">>> Initializing Model for Testing...")
    model = LFParallaxNet(num_classes=len(ALL_CONFIGS), num_rates=NUM_RATES).to(DEVICE)

    if not os.path.exists(LOAD_MODEL_PATH):
        logger.error(f"Checkpoint not found: {LOAD_MODEL_PATH}")
        return

    logger.info(f"Loading weights from {LOAD_MODEL_PATH}...")
    ckpt = torch.load(LOAD_MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(ckpt['model_state'])
    model.eval()

    if not TEST_DATASETS:
        logger.warning("TEST_DATASETS is empty. No data to evaluate.")
        return

    logger.info(">>> Starting Test Evaluation...")
    total_correct, total_samples = 0, 0

    for item in TEST_DATASETS:
        csv_path = item['csv']
        csv_name = os.path.basename(csv_path)

        single_df = prepare_dataframe([item], is_test=True)
        if single_df.empty: continue

        test_loader = DataLoader(LFMultiViewTestDataset(single_df, crop_size=IMAGE_SIZE),
                                 batch_size=1, shuffle=False, num_workers=4)

        current_correct, current_total = 0, 0
        logger.info(f"\n" + "=" * 30 + f" Dataset: {csv_name} " + "=" * 30)
        stats_by_dataset = {}

        with torch.no_grad():
            for i, (imgs, center, diff, scalar, r_ids, labels) in enumerate(test_loader):
                imgs, r_ids, labels = imgs.to(DEVICE), r_ids.to(DEVICE), labels.to(DEVICE)
                center, diff, scalar = center.to(DEVICE), diff.to(DEVICE), scalar.to(DEVICE)

                logits = model(imgs, center, diff, scalar, r_ids)
                pred_idx = torch.argmax(logits, 1).item()

                gt_idx = labels.item()
                row = single_df.iloc[i]
                scene_name = row['Scene_Name']
                rate_name = row['Rate_Point']
                dataset_prefix = row['dataset_prefix']

                pred_suffix = ID_TO_CONFIG[pred_idx].split('_')[-1]
                display_pred = f"{dataset_prefix}_{pred_suffix}"
                display_gt = row['Original_Config']

                is_correct = (pred_idx == gt_idx)
                status = "✓" if is_correct else f"✗ (GT: {display_gt})"

                if dataset_prefix not in stats_by_dataset:
                    stats_by_dataset[dataset_prefix] = {'correct': 0, 'total': 0}

                stats_by_dataset[dataset_prefix]['total'] += 1
                if is_correct:
                    stats_by_dataset[dataset_prefix]['correct'] += 1

                logger.info(f"Scene: {scene_name:<27} | Rate: {rate_name} | Predict: {display_pred:<8} | {status}")

                if is_correct: current_correct += 1
                current_total += 1

        acc = current_correct / current_total if current_total > 0 else 0
        logger.info(f"--> [{csv_name}] Sub-Total Accuracy: {acc:.2%} ({current_correct}/{current_total})")

        for dataset_prefix, stats in stats_by_dataset.items():
            if stats['total'] > 0:
                dataset_acc = stats['correct'] / stats['total']
                logger.info(f"  [{dataset_prefix}] Accuracy: {dataset_acc:.2%} ({stats['correct']}/{stats['total']})")

        total_correct += current_correct
        total_samples += current_total

    final_acc = total_correct / total_samples if total_samples > 0 else 0

    logger.info("\n" + "!" * 80)
    logger.info(f"OVERALL TEST ACCURACY: {final_acc:.4%} ({total_correct}/{total_samples})")
    logger.info("!" * 80 + "\n")

if __name__ == "__main__":
    run_test()