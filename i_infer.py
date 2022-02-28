import argparse
import sys
import json

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms

import _init_paths
import models
import models.aslloss
from models.query2label import build_q2l
from utils.misc import clean_state_dict
from dataset.cocodataset import CoCoDataset


coco_label_list = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
    'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
    'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
    'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
    'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
    'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant',
    'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
    'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush']


def parser_args():
    parser = argparse.ArgumentParser(
        description='Query2Label for multilabel classification')
    parser.add_argument('--img_size', default=448, type=int,
                        help='image size. default(448)')
    parser.add_argument(
        '--images_path', default='/home/zhuhao/dataset/public/coco/val2017',
        type=str, help='Path of images')
    parser.add_argument(
        '--annos_file',
        default='/home/zhuhao/dataset/public/coco/annotations_trainval2017/annotations/instances_val2017.json',
        type=str, help='File of annotations')
    parser.add_argument(
        '--labels_file',
        default='/home/zhuhao/myCode/vsass/stream_index/image_label/query2labels/data/coco/val_label_vectors_coco14.npy',
        type=str, help='File of labels')
    parser.add_argument(
        '--config',
        default='/home/zhuhao/myModel/query2labels/Q2L-CvT_w24-384/config_new.json',
        type=str, help='config file')
    parser.add_argument(
        '--output',
        default='/home/zhuhao/myCode/vsass/stream_index/image_label/query2labels/det_result',
        type=str, help='path to output folder')
    parser.add_argument(
        '--model_file',
        default='/home/zhuhao/myModel/query2labels/Q2L-CvT_w24-384/checkpoint.pkl',
        type=str, help='Path of model')

    parser.add_argument('--num_class', default=80, type=int,
                        help="Number of classes.")
    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                        help='number of data loading workers (default: 8)')
    parser.add_argument('--eps', default=1e-5, type=float,
                        help='eps for focal loss (default: 1e-5)')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='use pre-trained model. default is False. ')
    parser.add_argument(
        '-b', '--batch_size', default=1, type=int,
        help='mini-batch size (default: 8), this is the total batch size of all GPUs')
    parser.add_argument(
        '-p', '--print_freq', default=10, type=int,
        help='print frequency (default: 10)')

    # distribution training
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=0, type=int,
                        help='node rank for distributed training')
    parser.add_argument("--local_rank", default=0, type=int,
                        help='local rank for DistributedDataParallel')
    parser.add_argument('--amp', action='store_true',
                        help='use mixture precision.')
    parser.add_argument('--dist_url', default='tcp://127.0.0.1:3451', type=str,
                        help='url used to set up distributed training')

    # Transformer
    parser.add_argument(
        '--position_embedding', default='sine', type=str, choices=('sine'),
        help="Type of positional embedding to use on top of the image features")
    parser.add_argument(
        '--keep_other_self_attn_dec', action='store_true',
        help='keep the other self attention modules in transformer decoders, which will be removed default.')
    parser.add_argument(
        '--keep_first_self_attn_dec', action='store_true',
        help='keep the first self attention module in transformer decoders, which will be removed default.')
    parser.add_argument(
        '--keep_input_proj', action='store_true',
        help="keep the input projection layer. Needed when the channel of image features is different from hidden_dim of Transformer layers.")

    args = parser.parse_args()

    # update parameters with pre-defined config file
    if args.config:
        with open(args.config, 'r') as f:
            cfg_dict = json.load(f)
        for k, v in cfg_dict.items():
            setattr(args, k, v)

    return args


def main():
    args = parser_args()

    # single process, useful for debugging
    torch.cuda.set_device(args.local_rank)
    print('[INFO]: distributed init (local_rank {}): {}'.format(
        args.local_rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(
        backend='nccl', init_method=args.dist_url, world_size=args.world_size,
        rank=args.rank)
    cudnn.benchmark = True

    print("[INFO]: Command: "+' '.join(sys.argv))
    print('[INFO]: world size: {}'.format(dist.get_world_size()))
    print('[INFO]: dist.get_rank(): {}'.format(dist.get_rank()))
    print('[INFO]: local_rank: {}'.format(args.local_rank))

    return main_worker(args)


def get_datasets(img_size, images_path, annos_file, labels_file):

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    test_data_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        normalize])

    val_dataset = CoCoDataset(
        image_dir=images_path,
        anno_path=annos_file,
        input_transform=test_data_transform,
        labels_path=labels_file,
    )

    print("[INFO]: len(val_dataset):", len(val_dataset))
    return val_dataset


def main_worker(args):
    # build model
    model = build_q2l(args)
    model = model.cuda()
    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[args.local_rank], broadcast_buffers=False)
    criterion = models.aslloss.AsymmetricLossOptimized(
        gamma_neg=args.gamma_neg, gamma_pos=args.gamma_pos,
        disable_torch_grad_focal_loss=True,
        eps=args.eps,
    )

    # optionally resume from a checkpoint
    print("[INFO]: => loading checkpoint '{}'".format(args.model_file))
    checkpoint = torch.load(
        args.model_file, map_location=torch.device(dist.get_rank()))
    state_dict = clean_state_dict(checkpoint['state_dict'])
    model.module.load_state_dict(state_dict, strict=True)
    del checkpoint
    del state_dict
    torch.cuda.empty_cache()

    # Data loading code
    img_size = args.img_size
    images_path = args.images_path
    annos_file = args.annos_file
    labels_file = args.labels_file
    val_dataset = get_datasets(img_size, images_path, annos_file, labels_file)
    assert args.batch_size // dist.get_world_size() == args.batch_size / \
        dist.get_world_size(), 'Batch size is not divisible by num of gpus.'
    val_sampler = torch.utils.data.distributed.DistributedSampler(
        val_dataset, shuffle=False)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size // dist.get_world_size(),
        shuffle=False, num_workers=args.workers, pin_memory=True,
        sampler=val_sampler)

    # for eval only
    validate(val_loader, model, criterion, args)
    return


@torch.no_grad()
def validate(val_loader, model, criterion, args):
    # switch to evaluate mode
    model.eval()

    output_file = args.output
    out_file = open(output_file, 'w')

    with torch.no_grad():
        for _, (images, target, name) in enumerate(val_loader):
            images = images.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            image_name = name[0]

            # compute output
            with torch.cuda.amp.autocast(enabled=args.amp):
                output = model(images)
                criterion(output, target)
                output_sm = nn.functional.sigmoid(output)

            # save some data
            _item = torch.cat(
                (output_sm.detach().cpu(), target.detach().cpu()), 1)

            _item = _item[:, :args.num_class]
            top_probs, top_labels = _item.softmax(dim=-1).topk(5, dim=-1)
            scores = top_probs[0]
            labels_k = top_labels[0]
            labels_k = [coco_label_list[int(index)] for index in labels_k]
            print(
                f"[INFO]: {image_name}: top_scores are {scores}, top_labels are {labels_k}")

            label_str = ','.join(labels_k)
            out_file.write(image_name + ":" + label_str + '\n')

        if dist.get_world_size() > 1:
            dist.barrier()

    out_file.close()


if __name__ == '__main__':
    main()
