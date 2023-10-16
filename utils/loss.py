import torch
import torch.nn.functional as F

def cal_similarity(F_I, F_T, a1=0.1, a2=0.3):
    batch_size = F_I.size(0)

    knn_number = 3000
    scale = 4000

    F_I = F.normalize(F_I)
    S_I = F_I.mm(F_I.t())
    F_T = F.normalize(F_T)
    S_T = F_T.mm(F_T.t())

    S_pair = a1 * S_T + (1 - a1) * S_I

    pro = F_T.mm(F_T.t()) * a1 + F_I.mm(F_I.t()) * (1. - a1)
    size = batch_size
    top_size = knn_number
    m, n1 = pro.sort()
    pro[torch.arange(size).view(-1, 1).repeat(1, top_size).view(-1), n1[:, :top_size].contiguous().view(-1)] = 0.
    pro[torch.arange(size).view(-1), n1[:, -1:].contiguous().view(-1)] = 0.

    pro = pro / pro.sum(1).view(-1, 1)
    pro_dis = pro.mm(pro.t())

    pro_dis = pro_dis * scale  # ？？？[0,1]
    # pdb.set_trace()
    pro_dis = 2 * torch.sigmoid(pro_dis) - 1  # TODO 修改
    S_pair = 2 * torch.sigmoid(S_pair) - 1 + 0*torch.eye(S_pair.size(0))

    pro_dis = (pro_dis+pro_dis.t())/2
    S = (S_pair * (1 - a2) + pro_dis * a2)

    return S, S_pair, pro_dis



if __name__ == '__main__':
    pass