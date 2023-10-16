import numpy as np
import torch
import torch.nn.functional as F
from torch.backends import cudnn

def seed_everything(seed=1234):
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    cudnn.deterministic = True
    import os
    os.environ['PYTHONHASHSEED'] = str(seed)

def affinity_tag_multi(tag1: np.ndarray, tag2: np.ndarray):
    aff = np.matmul(tag1, tag2.T)
    affinity_matrix = np.float32(aff)
    affinity_matrix = 1 / (1 + np.exp(-affinity_matrix))
    affinity_matrix = 2 * affinity_matrix - 1
    return affinity_matrix

def affinity_clean_tag(img_list, tag_list, T=None):
    text_features = torch.tensor(tag_list).cuda()
    image_features = torch.tensor(img_list).cuda()

    # 选取参数最高的标签
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    similarity = (T * image_features @ text_features.T).softmax(dim=-1)  # 对图像描述和图像特征  100相当于是温度系数1/100

    similarity_norm = F.normalize(similarity, dim=-1)  # 5000×cls_num
    # similarity_norm = similarity
    values, indices = similarity_norm.sort()

    similarity_norm[torch.arange(5000).view(-1, 1).repeat(1, 1).view(-1), indices[:, -1].contiguous().view(-1)] = 1  # top1的位置赋1

    similarity_norm = torch.as_tensor(similarity_norm, dtype=torch.float32)
    aff = torch.matmul(similarity_norm, similarity_norm.T)
    # aff = torch.tensor(aff, dtype=torch.float32)

    aff = 1 / (1 + torch.exp(-aff))
    aff = 2 * aff - 1

    S = aff
    mask = torch.eye(5000, dtype=torch.bool).cuda()
    # S = torch.tensor(S)
    S = S.masked_fill(mask, value=torch.tensor(2))

    return S, similarity


def common_sample_detection(I_tr, T_tr,  imgNet_S, txtNet_S, L_tr=None, K_num = 0):

    image_features = torch.tensor(I_tr).cuda()
    text_features = torch.tensor(T_tr).cuda()
    L_tr = torch.tensor(L_tr, dtype=torch.float32).cuda()

    imgNet_S.eval()
    txtNet_S.eval()

    with torch.no_grad():

        im_f1, im_f2, im_f3, code_I = imgNet_S(image_features)
        txt_f1, txt_f2, txt_f3, code_T = txtNet_S(text_features)


    code_I_norm = F.normalize(code_I)
    code_T_norm = F.normalize(code_T)

    S_T_source = torch.matmul(code_T_norm, code_T_norm.T)
    S_I_source = torch.matmul(code_I_norm, code_I_norm.T)

    image_features_norm = F.normalize(image_features)
    text_features_norm = F.normalize(text_features)

    S_T_target = torch.matmul(text_features_norm, text_features_norm.T)
    S_I_target = torch.matmul(image_features_norm, image_features_norm.T)
    S_IT_target = torch.matmul(image_features_norm, text_features_norm.T)

    v_t_I, n_t_I = S_I_target.sort()

    # 用target图像计算真实际近邻样本， 比source和target图像偏差
    topsize = 5
    topk_v_s = S_I_source[torch.arange(5000).view(-1, 1).repeat(1, topsize).view(-1), n_t_I[:, -topsize-1:-1].contiguous().view(-1)].view(-1, topsize)  # 除了自身topk个近邻值(相对target在source中)
    topk_v_t = S_I_target[torch.arange(5000).view(-1, 1).repeat(1, topsize).view(-1), n_t_I[:, -topsize-1:-1].contiguous().view(-1)].view(-1, topsize)  # 除了自身topk个近邻值(相对target在source中)
    # 比source和target文本偏差
    topk_v_s = S_T_source[torch.arange(5000).view(-1, 1).repeat(1, topsize).view(-1), n_t_I[:, -topsize-1:-1].contiguous().view(-1)].view(-1, topsize)  # 除了自身topk个近邻值(相对target在source中)
    topk_v_t = S_T_target[torch.arange(5000).view(-1, 1).repeat(1, topsize).view(-1), n_t_I[:, -topsize-1:-1].contiguous().view(-1)].view(-1, topsize)  # 除了自身topk个近邻值(相对target在source中)



    topk_v_s_mean = torch.mean(topk_v_s, dim=-1) # ui
    topk_v_t_mean = torch.mean(topk_v_t, dim=-1) # vi

    topk_v_s_var = torch.var(topk_v_s, dim=-1)
    topk_v_t_var = torch.var(topk_v_t, dim=-1)

    topk_v_s_mean_high = torch.mean(topk_v_s_mean)
    topk_v_t_mean_high = torch.mean(topk_v_t_mean)

    topk_v_s_var_high = torch.mean(topk_v_s_var)
    topk_v_t_var_high = torch.mean(topk_v_t_var)


    # topk_v_s_maxmin = torch.max(topk_v_s, dim=-1)[0] - torch.min(topk_v_s, dim=-1)[0]
    # topk_v_t_maxmin = torch.max(topk_v_t, dim=-1)[0] - torch.min(topk_v_t, dim=-1)[0]

    abs_mean = torch.abs(topk_v_s_mean - topk_v_t_mean)
    abs_var = torch.abs(topk_v_s_var - topk_v_t_var)


    abs_mean_high = torch.abs(topk_v_t_mean_high-topk_v_s_mean_high)
    abs_var_high = torch.abs(topk_v_t_var_high - topk_v_s_var_high)


    train_index = torch.arange(S_T_source.size(0))


    common_select = (abs_mean < 1*abs_mean_high) * (abs_var < 1*abs_var_high) # 1可以调
    common_index = train_index[common_select]

    init_weight = torch.zeros_like(S_I_target)
    init_weight[common_index.view(-1, 1).repeat(1, len(common_index)).view(-1), common_index.repeat(len(common_index))] = 1 # 相似样本处为一

    S_I_source_common = S_I_source * init_weight
    S_T_source_common = S_T_source * init_weight

    S_I_source_common[S_I_source_common == 0] = 2
    S_T_source_common[S_T_source_common == 0] = 2

    # v_I, n_I = S_I_source_common.sort()
    # v_T, n_T = S_T_source_common.sort()


    v_I, n_I = S_I_target.sort()
    v_T, n_T = S_T_target.sort()

    K_num = K_num
    # S_I_target[torch.arange(5000).view(-1, 1).repeat(1, K_num).view(-1), n_I[:, :K_num].contiguous().view(-1)] = 0  # 最小的赋0
    # S_I_target[S_I_target != 0] = 1
    S_T_target[torch.arange(5000).view(-1, 1).repeat(1, K_num).view(-1), n_T[:, :K_num].contiguous().view(-1)] = 0  # 最小的赋0
    S_T_target[S_T_target != 0] = 1

    R_origin = init_weight * S_T_target

    return common_index, init_weight, R_origin, S_I_source, S_T_source, code_I_norm, code_T_norm



def cal_code_similarity(I_code, T_code):
    S_I = torch.matmul(I_code, I_code.T)
    S_T = torch.matmul(T_code, T_code.T)
    S_I_source_sig = 2 / (1 + torch.exp(-S_I)) - 1
    # S_T_source_sig = 2 / (1 + torch.exp(-S_T)) - 1
    S = S_I_source_sig
    return S

