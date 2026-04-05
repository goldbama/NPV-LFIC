import argparse
import yaml
from utils.experiment import *

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default='./cfgs/bim_vfi_demo.yaml', type=str)
    parser.add_argument('--load-root', default='data')
    parser.add_argument('--save-root', default='save')
    parser.add_argument('--name', '-n', default=None)
    parser.add_argument('--tag', default=None)
    parser.add_argument('--cudnn', action='store_true')
    parser.add_argument('--port-offset', '-p', type=int, default=0)
    parser.add_argument('--wandb-upload', '-w', action='store_true')

    parser.add_argument('--img0_path', default='./test_image1/EPFL/000_000.png', type=str)
    parser.add_argument('--img1_path', default='./test_image1/EPFL/000_008.png', type=str)
    parser.add_argument('--save_folder', default='./test_image1/EPFL', type=str)
    parser.add_argument('--output_name', default='000_004', type=str)

    args = parser.parse_args()

    return args


def make_cfg(args):
    with open(args.cfg, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    def translate_cfg_(d):
        for k, v in d.items():
            if isinstance(v, dict):
                translate_cfg_(v)
            elif isinstance(v, str):
                d[k] = v.replace('$load_root$', args.load_root)
    translate_cfg_(cfg)

    if args.name is None:
        exp_name = os.path.basename(args.cfg).split('.')[0].replace('_benchmark', '').replace('_demo', '')
    else:
        exp_name = args.name
    if args.tag is not None:
        exp_name += '_' + args.tag

    env = dict()
    env['exp_name'] = exp_name + '_' + cfg['exp_name']
    env['save_dir'] = os.path.join(args.save_root, env['exp_name'])
    env['tot_gpus'] = torch.cuda.device_count()
    env['cudnn'] = args.cudnn
    env['port'] = str(29600 + args.port_offset)
    env['wandb_upload'] = args.wandb_upload
    cfg['env'] = env

    cfg['img0_path'] = args.img0_path
    cfg['img1_path'] = args.img1_path
    cfg['save_folder'] = args.save_folder
    cfg['output_name'] = args.output_name

    return cfg


def main():
    args = parse_args()

    cfgs = make_cfg(args)

    init_experiment(cfgs)
    init_distributed_mode(cfgs)
    init_deterministic(cfgs['seed'])

    trainer = Trainer(cfgs)

    if cfgs['mode'] == 'train':
        trainer.train()
    elif cfgs['mode'] == 'validate':
        trainer.validate()
    elif cfgs['mode'] == 'benchmark':
        trainer.benchmark()
    elif cfgs['mode'] == 'demo':
        trainer.demo()


import torch
from trainer import Trainer

def load_model_once():
    args = parse_args()
    cfgs = make_cfg(args)
    init_experiment(cfgs)
    init_distributed_mode(cfgs)
    init_deterministic(cfgs['seed'])
    trainer = Trainer(cfgs)
    # trainer.demo()

    model = trainer.model
    return model

def run_inference(model, img0, img1, save_folder, output_name, ratio=2, from_array=False):
    from modules.models.inference_video import inference_demo
    import torch

    net = model.model

    with torch.no_grad():
        output = inference_demo(
            net,
            ratio,
            img0,
            img1,
            output_name,
            save_folder,
            from_array=from_array
        )

    return output



if __name__ == '__main__':
    main()
