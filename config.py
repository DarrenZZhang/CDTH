import argparse
parser = argparse.ArgumentParser(description='testDA')
parser.add_argument('--source_gpu_id', type=str, nargs='?', default='2', help="device id to run")
parser.add_argument('--target_gpu_id', type=str, nargs='?', default='1', help="device id to run")

parser.add_argument('--source_epoch', type=int, default=100, help="max iterations") # flickr:100, target:100, coco:20
parser.add_argument('--target_epoch', type=int, default=300, help="max iterations") # flickr:40, target:40, coco:300
parser.add_argument('--batch_size_s', type=int, default=256, help="batch_size")
parser.add_argument('--batch_size', type=int, default=256, help="batch_size")
parser.add_argument('--num_workers', type=int, default=0, help="number of workers")

parser.add_argument('--data_path', type=str, default='/data/WangBoWen/Dataset/')  # 数据集根路径

parser.add_argument('--s_dset', type=str, default='flickr25k', choices=['coco', 'nuswide', 'flickr25k'])  # 选择源域数据集
parser.add_argument('--t_dset', type=str, default='coco', choices=['coco', 'nuswide', 'flickr25k'])  # 选择目标域数据集

# parser.add_argument('--s_dset', type=str, default='flickr25k', choices=['coco', 'nuswide', 'flickr25k'])  # 选择源域数据集
# parser.add_argument('--t_dset', type=str, default='nuswide', choices=['coco', 'nuswide', 'flickr25k'])  # 选择目标域数据集

parser.add_argument('--seed', type=int, default=1234, help="random seed")  # 随机种子

parser.add_argument('--source_lr', type=float, default=0.0001)
parser.add_argument('--source_weight_decay', type=float, default=0.0005)
parser.add_argument('--target_lr', type=float, default=0.00005) # target, lr nus,coco:5e-5   flickr:1e-4
parser.add_argument('--target_weight_decay', type=float, default=0.00025) # target, weight_decay nus,coco:2.5e-4   flickr:5e-4

parser.add_argument('--a1', type=float, default=0.9) #  flickr:0.4, nus:0.01, coco:0.9
parser.add_argument('--a2', type=float, default=0.3) #  flickr:1, nus:0.7, coco:0.3
parser.add_argument('--TopK', type=float, default=100) #  flickr:50, nus:30, coco:100
parser.add_argument('--K_num', type=float, default=1000) #  flickr:3000, nus:3500, coco:1000
parser.add_argument('--T', type=float, default=30)  # flickr:80, nus:80, coco:30

parser.add_argument('--lamda1', type=float, default=1.7) #  flickr:0.4, nus:0.6, coco:1.7
parser.add_argument('--lamda2', type=float, default=0.1) #  flickr:1, nus:1, coco:0.1

args = parser.parse_args()






