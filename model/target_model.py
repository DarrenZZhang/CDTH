import torch
import torch.nn as nn

class ImgNet_T(nn.Module):
    def __init__(self, code_len=64, nclass=21):
        super(ImgNet_T, self).__init__()

        self.img_encoder1 = nn.Sequential(
            # nn.Linear(4096, 4096),
            nn.Linear(512, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(),
        )

        self.img_encoder2 = nn.Sequential(
            nn.Linear(4096, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
        )

        self.img_encoder3 = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(p=0.5)
        )

        self.imgHashing = nn.Sequential(
            nn.Linear(512, code_len),
        )

        self.dropout = nn.Dropout(p=0.5)  # ?????
        self.relu = nn.ReLU()
        self.alpha = 1.0

    def get_weight(self, ):
        for name, param in self.img_encoder3[0].named_parameters():
            if name == 'weight':
                return param

    def get_weight2(self, ):
        for name, param in self.imgHashing[0].named_parameters():
            if name == 'weight':
                return param

    def forward(self, x):

        feat1 = self.img_encoder1(x)
        feat2 = self.img_encoder2(feat1)
        feat3 = self.img_encoder3(feat2)
        code = self.imgHashing(feat3)
        code = torch.tanh(code)
        return feat1, feat2, feat3, code


class TxtNet_T(nn.Module):
    def __init__(self, text_length=1386, code_len=64):
        super(TxtNet_T, self).__init__()
        self.txt_encoder1 = nn.Sequential(
            nn.Linear(text_length, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(),
        )

        self.txt_encoder2 = nn.Sequential(
            nn.Linear(4096, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
        )

        self.txt_encoder3 = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
        )

        self.txtHashing = nn.Sequential(
            nn.Linear(512, code_len),
        )

        self.dropout = nn.Dropout(p=0.5)  # ?????
        self.relu = nn.ReLU()
        self.alpha = 1.0

    def get_weight(self, ):
        for name, param in self.txt_encoder3[0].named_parameters():
            if name == 'weight':
                return param

    def get_weight2(self, ):
        for name, param in self.txtHashing[0].named_parameters():
            if name == 'weight':
                return param

    def forward(self, x):
        feat1 = self.txt_encoder1(x)
        feat2 = self.txt_encoder2(feat1)
        feat3 = self.txt_encoder3(feat2)
        code = self.txtHashing(feat3)
        code = torch.tanh(code)
        return feat1, feat2, feat3, code



class Classifer_T(nn.Module):
    def __init__(self, in_dim=512, nclass=24):
        super(Classifer_T, self).__init__()

        self.Classifier = nn.Sequential(
            nn.Linear(in_dim, nclass)
        )

    def forward(self, x):
        y_pred = self.Classifier(x)
        return y_pred
