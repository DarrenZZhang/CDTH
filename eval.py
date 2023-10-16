import torch
from utils.metric import calc_map_k


# target
def eval_retrieval_target(imgNet, txtNet, test_dl, retrival_dl, db_name=None):

    test_dl_dict = {'img_code': [], 'txt_code': [], 'label': []}
    retrieval_dl_dict = {'img_code': [], 'txt_code': [], 'label': []}
    imgNet.eval()
    txtNet.eval()
    with torch.no_grad():
        for i, (im_t, txt_t, label_t, id_target) in enumerate(test_dl):
            im_t = im_t.cuda()
            txt_t = txt_t.cuda()
            label_t = label_t.cuda()
            Im_f1, Im_f2, Im_f3, Im_code = imgNet(im_t)
            Txt_f1, Txt_f2, Txt_f3, Txt_code = txtNet(txt_t)



            Im_code = torch.sign(Im_code)
            Txt_code = torch.sign(Txt_code)

            test_dl_dict['img_code'].append(Im_code)
            test_dl_dict['txt_code'].append(Txt_code)
            test_dl_dict['label'].append(label_t)



        for i, (im_t, txt_t, label_t, id_target) in enumerate(retrival_dl):
            im_t_db = im_t.cuda()
            txt_t_db = txt_t.cuda()
            label_t_db = label_t.cuda()
            Im_f1, Im_f2, Im_f3, Im_code = imgNet(im_t_db)
            Txt_f1, Txt_f2, Txt_f3, Txt_code = txtNet(txt_t_db)

            Im_code = torch.sign(Im_code)
            Txt_code = torch.sign(Txt_code)

            retrieval_dl_dict['img_code'].append(Im_code)
            retrieval_dl_dict['txt_code'].append(Txt_code)
            retrieval_dl_dict['label'].append(label_t_db)

    query_img = torch.cat(test_dl_dict['img_code'], dim=0).cpu()
    query_txt = torch.cat(test_dl_dict['txt_code'], dim=0).cpu()
    query_label = torch.cat(test_dl_dict['label'], dim=0).cpu()

    retrieval_img = torch.cat(retrieval_dl_dict['img_code'], dim=0).cpu()
    retrieval_txt = torch.cat(retrieval_dl_dict['txt_code'], dim=0).cpu()
    retrieval_label = torch.cat(retrieval_dl_dict['label'], dim=0).cpu()

    # ## Save
    # _dict = {
    #     'I_db': np.array(retrieval_img),
    #     'I_te': np.array(query_img),
    #     'T_te': np.array(query_txt),
    #     'T_db': np.array(retrieval_txt),
    #     'L_te': np.array(query_label),
    #     'L_db': np.array(retrieval_label),
    # }
    # sava_path = './hashcode/' + db_name + '_' + str(bit) + '.mat'
    # sio.savemat(sava_path, _dict)

    # 计算i2t的map
    mapi2t = calc_map_k(query_img.cuda(), retrieval_txt.cuda(), query_label.cuda(), retrieval_label.cuda())
    # 计算t2i的map
    mapt2i = calc_map_k(query_txt.cuda(), retrieval_img.cuda(), query_label.cuda(), retrieval_label.cuda())

    imgNet.train()
    txtNet.train()
    return mapi2t.item(), mapt2i.item()

# source
def eval_retrieval_source(imgNet, txtNet, test_dl, retrival_dl):

    test_dl_dict = {'img_code': [], 'txt_code': [], 'label': []}
    retrieval_dl_dict = {'img_code': [], 'txt_code': [], 'label': []}
    imgNet.eval()
    txtNet.eval()
    with torch.no_grad():
        for i, (im_t, txt_t, label_t, id_target) in enumerate(test_dl):
            im_t = im_t.cuda()
            txt_t = txt_t.cuda()
            label_t = label_t.cuda()
            Im_f1, Im_f2, Im_f3, Im_code = imgNet(im_t)
            Txt_f1, Txt_f2, Txt_f3, Txt_code = txtNet(txt_t)

            Im_code = torch.sign(Im_code)
            Txt_code = torch.sign(Txt_code)

            test_dl_dict['img_code'].append(Im_code)
            test_dl_dict['txt_code'].append(Txt_code)
            test_dl_dict['label'].append(label_t)

        for i, (im_t, txt_t, label_t, id_target) in enumerate(retrival_dl):
            im_t_db = im_t.cuda()
            txt_t_db = txt_t.cuda()
            label_t_db = label_t.cuda()
            Im_f1, Im_f2, Im_f3, Im_code = imgNet(im_t_db)
            Txt_f1, Txt_f2, Txt_f3, Txt_code = txtNet(txt_t_db)

            Im_code = torch.sign(Im_code)
            Txt_code = torch.sign(Txt_code)

            retrieval_dl_dict['img_code'].append(Im_code)
            retrieval_dl_dict['txt_code'].append(Txt_code)
            retrieval_dl_dict['label'].append(label_t_db)


    query_img = torch.cat(test_dl_dict['img_code'], dim=0).cpu()
    query_txt = torch.cat(test_dl_dict['txt_code'], dim=0).cpu()
    query_label = torch.cat(test_dl_dict['label'], dim=0).cpu()

    retrieval_img = torch.cat(retrieval_dl_dict['img_code'], dim=0).cpu()
    retrieval_txt = torch.cat(retrieval_dl_dict['txt_code'], dim=0).cpu()
    retrieval_label = torch.cat(retrieval_dl_dict['label'], dim=0).cpu()

    # 计算i2t的map
    mapi2t = calc_map_k(query_img.cuda(), retrieval_txt.cuda(), query_label.cuda(), retrieval_label.cuda())
    # 计算t2i的map
    mapt2i = calc_map_k(query_txt.cuda(), retrieval_img.cuda(), query_label.cuda(), retrieval_label.cuda())

    # 计算i2i的map

    imgNet.train()
    txtNet.train()
    return mapi2t.item(), mapt2i.item()


