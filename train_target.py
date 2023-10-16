from utils.util import *
from data import *
from utils.loss import *
from model.source_model import *
from model.target_model import *
import scipy.io as sio

from eval import *
import time
import os
os.environ["CUDA_VISIBLE_DEVICES"] = args.source_gpu_id

loss_l2 = torch.nn.MSELoss()
loss_cls = torch.nn.CrossEntropyLoss()


def freeze_bn(L):
    if isinstance(L, nn.BatchNorm1d):
        L.eval()


def train(code_len=64):

    # 设置随机种子
    seed_everything(seed=args.seed)
    # 定义模型
    imgNet_S = ImgNet_S(code_len=code_len).cuda().eval()
    txtNet_S = TxtNet_S(text_length=512, code_len=code_len).cuda().eval()

    imgNet_T = ImgNet_T(code_len=code_len).cuda().train()
    txtNet_T = TxtNet_T(text_length=512, code_len=code_len).cuda().train()

    imgNet_T_weight = imgNet_T.state_dict()
    txtNet_T_weight = txtNet_T.state_dict()

    if args.s_dset == 'flickr25k':
        # model_path = '/data/WangBoWen/model_file/flickr_clip_{}_noreconst_{}.pth'.format(code_len, epoch)
        model_path = '/data/WangBoWen/model_file/flickr_clip_{}_noreconst_99.pth'.format(code_len)

    if args.s_dset == 'nuswide':
        # model_path = '/data/WangBoWen/model_file/nus10_clip_{}_noreconst_{}.pth'.format(code_len, epoch)
        model_path = '/data/WangBoWen/model_file/nus10_clip_{}_noreconst_99.pth'.format(code_len)

    if args.s_dset == 'coco':
        # model_path = '/data/WangBoWen/model_file/coco_clip_{}_noreconst_{}.pth'.format(code_len, epoch)
        model_path = '/data/WangBoWen/model_file/coco_clip_{}_noreconst_19.pth'.format(code_len)


    if args.t_dset == 'flickr25k':
        tag_path = 'data/tags/flickr_tags_tr.mat'
    if args.t_dset == 'nuswide':
        tag_path = 'data/tags/nus_tags_tr.mat'
    if args.t_dset == 'coco':
        tag_path = 'data/tags/coco_tags_tr.mat'


    state = torch.load(model_path)
    imgNet_S_state = state['imgNet']
    txtNet_S_state = state['txtNet']

    imgNet_S.load_state_dict(state['imgNet'])
    txtNet_S.load_state_dict(state['txtNet'])



    common_index, init_weight, R_origin, S_I_source, S_T_source, code_I_norm, code_T_norm = common_sample_detection(I_tr=target_train_ds.images, T_tr=target_train_ds.texts,
                                                     imgNet_S=imgNet_S, txtNet_S=txtNet_S, L_tr=target_train_ds.labels, K_num=args.K_num)


    R_origin = torch.as_tensor(R_origin, dtype=torch.float32)


    S_fusion_source = cal_code_similarity(code_I_norm, code_T_norm)

    # 加载imgNet_S的部分参数
    for k in imgNet_S_state.keys():
        if k.startswith('img_encoder1'):
            imgNet_T_weight[k] = imgNet_S_state[k]

        if k.startswith('img_encoder2'):
            imgNet_T_weight[k] = imgNet_S_state[k]

        # if k.startswith('img_encoder3.0'):
        #     imgNet_T_weight[k] = imgNet_S_state[k]

        if k.startswith('imgHashing.0'):
            imgNet_T_weight[k] = imgNet_S_state[k]


    imgNet_T.load_state_dict(imgNet_T_weight)

    # 冻结部分层的参数
    params_I = []
    freeze_layer = ['img_encoder1', 'img_encoder2']

    for name, param in imgNet_T.named_parameters():
        if any(name.startswith(prefix) for prefix in freeze_layer):
            param.requires_grad = False
        else:
            params_I.append(param)

    # 加载txtNet_S的部分参数
    for k in txtNet_S_state.keys():
        if k.startswith('txt_encoder1'):
            txtNet_T_weight[k] = txtNet_S_state[k]

        if k.startswith('txt_encoder2'):
            txtNet_T_weight[k] = txtNet_S_state[k]

        # if k.startswith('txt_encoder3.0'):
        #     imgNet_T_weight[k] = txtNet_S_state[k]

        if k.startswith('txtHashing.0'):
            txtNet_T_weight[k] = txtNet_S_state[k]



    txtNet_T.load_state_dict(txtNet_T_weight)

    # 冻结部分层的参数
    params_T = []
    freeze_layer = ['txt_encoder1', 'txt_encoder2']

    for name, param in txtNet_T.named_parameters():
        if any(name.startswith(prefix) for prefix in freeze_layer):
            param.requires_grad = False
        else:
            params_T.append(param)

    # label embedding
    tags_feature = sio.loadmat(tag_path)['tags'][:args.TopK]

    # pseudo label
    aff_tag_label, _ = affinity_clean_tag(target_train_ds.images, tags_feature, T=args.T)  # [0,1]


    S_origin, _, _ = cal_similarity(torch.as_tensor(target_train_ds.images, dtype=torch.float32), torch.as_tensor(target_train_ds.texts, dtype=torch.float32), a1=args.a1, a2=args.a2)

    R = R_origin * S_origin.cuda()

    opt_T = torch.optim.Adam([{'params': filter(lambda p: p.requires_grad, txtNet_T.parameters())},
                              ], lr=args.target_lr, weight_decay=args.target_weight_decay)
    opt_I = torch.optim.Adam([{'params': filter(lambda p: p.requires_grad, imgNet_T.parameters())},
                              ], lr=args.target_lr, weight_decay=args.target_weight_decay)


    # hash码缓存
    I_buffer = torch.randn((5000, code_len)).cuda()
    T_buffer = torch.randn((5000, code_len)).cuda()

    torch.cuda.synchronize()
    start = time.time()
    # TODO-------------------------------------------------------------------------------------------
    for epoch in range(args.target_epoch):
        for i, (im_target, text_target, label_target, id_target) in enumerate(target_train_dl):
            im_target = im_target.cuda()
            text_target = text_target.cuda()
            all_1 = torch.rand(batch_size).fill_(1).cuda()

            img_f1, img_f2, img_f3, img_hash_t = imgNet_T(im_target)
            txt_f1, txt_f2, txt_f3, txt_hash_t = txtNet_T(text_target)

            I_buffer[id_target, :] = img_hash_t.detach()
            T_buffer[id_target, :] = txt_hash_t.detach()


            img_hash_norm = F.normalize(img_hash_t)
            txt_hash_norm = F.normalize(txt_hash_t)

            id_target = id_target.cuda()

            s_loss = loss_l2(img_hash_norm.mm(img_hash_norm.t()), aff_tag_label[id_target, :][:, id_target]) + \
                     loss_l2(txt_hash_norm.mm(txt_hash_norm.t()), aff_tag_label[id_target, :][:, id_target]) + \
                     loss_l2(img_hash_norm.mm(txt_hash_norm.t()), aff_tag_label[id_target, :][:, id_target])


            R_batch = R[id_target, :][:, id_target].cuda()

            s_loss_tf = 1 * loss_l2(img_hash_norm.mm(img_hash_norm.t()) * R_batch, R_batch) + \
                       1 * loss_l2(txt_hash_norm.mm(txt_hash_norm.t()) * R_batch, R_batch) + \
                       loss_l2(img_hash_norm.mm(txt_hash_norm.t()) * R_batch, R_batch)


            B = torch.sign((img_hash_t + txt_hash_t) / 2)
            b_loss = loss_l2(img_hash_t, B) + loss_l2(txt_hash_t, B)
            loss_all = 1 * s_loss + args.lamda1 * s_loss_tf + args.lamda2 * b_loss  # 0.4

            opt_T.zero_grad()
            opt_I.zero_grad()
            loss_all.backward()
            opt_T.step()
            opt_I.step()


        I_buffer_norm = F.normalize(I_buffer)
        T_buffer_norm = F.normalize(T_buffer)

        S_fusion_target = cal_code_similarity(I_buffer_norm, T_buffer_norm)
        R = torch.exp(S_fusion_source - S_fusion_target) * R_origin * S_origin.cuda()

        # print('...epoch: %3d, loss_all: %3.3f' % (epoch, loss_all.item()))
        print('...epoch: %3d, s_loss: %3.3f, s_loss_tf: %3.3f, b_loss: %3.3f' % (epoch, s_loss.item(), s_loss_tf.item(), b_loss.item()))

        if (epoch + 1) == args.target_epoch:

            end = time.time()
            time_all = end - start
            # print(time_all)

            mapi2t, mapt2i = eval_retrieval_target(imgNet=imgNet_T, txtNet=txtNet_T, test_dl=target_test_dl, retrival_dl=target_retrieval_dl, db_name='coco_f')
            print("test MAP: MAP_t(i->t) %3.4f, MAP_t(t->i) %3.4f" % (mapi2t, mapt2i))
            # break

        is_save = False
        if is_save:
            model_data = {
                "imgNet": imgNet_T.state_dict(),
                'txtNet': txtNet_T.state_dict(),
            }
            model_path = './model_file/{}_with_{}_{}bits.pth'.format(args.t_dset, args.s_dset, code_len)
            torch.save(model_data, model_path)


if __name__ == '__main__':

    for bit in [64]:
        train(code_len=bit)
    pass


