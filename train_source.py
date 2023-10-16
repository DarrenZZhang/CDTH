from utils.util import seed_everything, affinity_tag_multi
import os

# args 是参数
from data import source_train_dl, source_test_dl, source_retrieval_dl, args
from utils.loss import *
from model.source_model import *

from eval import *

os.environ["CUDA_VISIBLE_DEVICES"] = args.source_gpu_id
SEED = args.seed
# 设置随机种子
seed_everything(seed=SEED)

loss_l2 = torch.nn.MSELoss()
loss_cls = torch.nn.CrossEntropyLoss()


def freeze_bn(L):
    if isinstance(L, nn.BatchNorm1d):
        L.eval()


def train(code_len = 16):

    # nclass
    if args.s_dset == 'flickr25k':
        nclass = 24
    if args.s_dset == 'nuswide':
        nclass = 10
    if args.s_dset == 'coco':
        nclass = 80

    # 定义模型
    imgNet = ImgNet_S(code_len=code_len).cuda()
    txtNet = TxtNet_S(text_length=512, code_len=code_len).cuda()
    clsNet = Classifer_S(in_dim=512, nclass=nclass).cuda()

    opt_T = torch.optim.Adam([{'params': filter(lambda p: p.requires_grad, txtNet.parameters())},
                              ], lr=args.source_lr, weight_decay=args.source_weight_decay)

    opt_I = torch.optim.Adam([{'params': filter(lambda p: p.requires_grad, imgNet.parameters())},
                              ], lr=args.source_lr, weight_decay=args.source_weight_decay)

    opt_C = torch.optim.Adam([{'params': filter(lambda p: p.requires_grad, clsNet.parameters())},
                              ], lr=args.source_lr, weight_decay=args.source_weight_decay)

    for epoch in range(args.source_epoch):
        for i, (im_source, text_source, label_source, id_source) in enumerate(source_train_dl):
            im_source = im_source.cuda()
            text_source = text_source.cuda()
            label_source = label_source.cuda()

            # 源域有监督hash
            aff_label = affinity_tag_multi(label_source.cpu().numpy(), label_source.cpu().numpy())  # [0,1]
            aff_label = torch.as_tensor(aff_label, dtype=torch.float32).cuda() # 0-1之间

            img_f1, img_f2, img_f3, img_hash = imgNet(im_source)
            txt_f1, txt_f2, txt_f3, txt_hash = txtNet(text_source)

            img_pred = clsNet(img_f3)
            txt_pred = clsNet(txt_f3)


            img_hash_norm = F.normalize(img_hash)
            txt_hash_norm = F.normalize(txt_hash)

            B = torch.sign((img_hash + txt_hash) / 2)
            b_loss = loss_l2(img_hash, B) + loss_l2(txt_hash, B)

            s_loss = loss_l2(img_hash_norm.mm(img_hash_norm.t()), aff_label) + \
                     loss_l2(txt_hash_norm.mm(txt_hash_norm.t()), aff_label) + \
                     loss_l2(img_hash_norm.mm(txt_hash_norm.t()), aff_label)

            cls_loss = loss_cls(img_pred, label_source) + loss_cls(txt_pred, label_source)
            loss_s = s_loss + 1*cls_loss + 1*b_loss

            opt_T.zero_grad()
            opt_I.zero_grad()
            opt_C.zero_grad()
            loss_s.backward()
            opt_T.step()
            opt_I.step()
            opt_C.step()

        print('...epoch: %3d, loss_s: %3.3f, b_loss: %3.3f, s_loss:%3.3f, cls_loss:%3.3f' % (epoch, loss_s.item(), b_loss.item(), s_loss.item(), cls_loss.item()))
        if (epoch + 1) == args.source_epoch:
            mapi2t, mapt2i = eval_retrieval_source(imgNet=imgNet, txtNet=txtNet, test_dl=source_test_dl, retrival_dl=source_retrieval_dl)
            print("test MAP: MAP_t(i->t) %3.4f, MAP_t(t->i) %3.4f " % (mapi2t, mapt2i))
            # break
            print('------save start-----')
            model_data = {
                "imgNet": imgNet.state_dict(),
                'txtNet': txtNet.state_dict(),
            }

            if args.s_dset == 'flickr25k':
                model_path = './model_file/flickr_clip_{}_noreconst_{}.pth'.format(code_len, epoch)
            if args.s_dset == 'nuswide':
                model_path = './model_file/nus10_clip_{}_noreconst_{}.pth'.format(code_len, epoch)
            if args.s_dset == 'coco':
                model_path = './model_file/coco_clip_{}_noreconst_{}.pth'.format(code_len, epoch)

            # print(model_path)
            # torch.save(model_data, model_path)
            # print('------save-----')





    pass

if __name__ == '__main__':

    for bit in [64]:
        train(code_len=bit)

