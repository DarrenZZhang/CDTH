from config import args
from utils.dset import PreDataset
from torch.utils.data import DataLoader


dataset_root = args.data_path  # 数据集根路径
source = args.s_dset  # source 数据集
target = args.t_dset  # target 数据集
batch_size_s = args.batch_size_s
batch_size = args.batch_size
num_workers = args.num_workers


"""dataset"""
source_train_ds = PreDataset(data_path=dataset_root, data_split='train', dataname=source, flag='nosup')
source_test_ds = PreDataset(data_path=dataset_root, data_split='test', dataname=source, flag='nosup')
source_retrieval_ds = PreDataset(data_path=dataset_root, data_split='retrieval', dataname=source, flag='nosup')


target_train_ds = PreDataset(data_path=dataset_root, data_split='train', dataname=target, flag='target')
target_test_ds = PreDataset(data_path=dataset_root, data_split='test', dataname=target,  flag='target')
target_retrieval_ds = PreDataset(data_path=dataset_root, data_split='retrieval', dataname=target,  flag='target')



"""dataloader"""
source_train_dl = DataLoader(dataset=source_train_ds, batch_size=batch_size_s,
                             num_workers=num_workers, drop_last=True, shuffle=True)

source_test_dl = DataLoader(dataset=source_test_ds, batch_size=batch_size_s,
                             num_workers=num_workers, drop_last=False, shuffle=False)

source_retrieval_dl = DataLoader(dataset=source_retrieval_ds, batch_size=batch_size_s,
                             num_workers=num_workers, drop_last=False, shuffle=False)



target_train_dl = DataLoader(dataset=target_train_ds, batch_size=batch_size, shuffle=True,
                             num_workers=num_workers, drop_last=True)
target_test_dl = DataLoader(dataset=target_test_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, drop_last=False)
target_retrieval_dl = DataLoader(dataset=target_retrieval_ds, batch_size=batch_size, shuffle=False,
                                 num_workers=num_workers, drop_last=False)





if __name__ == '__main__':

    pass
