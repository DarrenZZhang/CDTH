import torch.nn as nn
class ImgNet_S(nn.Module):
    def __init__(self, code_len=64):
        super(ImgNet_S, self).__init__()

        self.img_encoder1 = nn.Sequential(
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
            nn.BatchNorm1d(512),
            nn.ReLU(),
        )

        self.imgHashing = nn.Sequential(
            nn.Linear(512, code_len),
            nn.BatchNorm1d(code_len),
            nn.Tanh()
        )

        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()
        self.alpha = 1.0

    def get_weight(self, ):
        for name, param in self.img_encoder3[0].named_parameters():
            if name == 'weight':
                return param

    def forward(self, x):

        feat1 = self.img_encoder1(x)
        feat2 = self.img_encoder2(feat1)
        feat3 = self.img_encoder3(feat2)
        code = self.imgHashing(feat3)
        return feat1, feat2, feat3, code


class TxtNet_S(nn.Module):
    def __init__(self, text_length=1000, code_len=64):
        super(TxtNet_S, self).__init__()
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
            nn.BatchNorm1d(512),
            nn.ReLU(),
        )

        self.txtHashing = nn.Sequential(
            nn.Linear(512, code_len),
            nn.BatchNorm1d(code_len),
            nn.Tanh()
        )


        self.dropout = nn.Dropout(p=0.5)  # ?????
        self.relu = nn.ReLU()
        self.alpha = 1.0

    def forward(self, x):
        feat1 = self.txt_encoder1(x)
        feat2 = self.txt_encoder2(feat1)
        feat3 = self.txt_encoder3(feat2)
        code = self.txtHashing(feat3)
        return feat1, feat2, feat3, code


class Classifer_S(nn.Module):
    def __init__(self, in_dim=512, nclass=21):
        super(Classifer_S, self).__init__()

        self.Classifier = nn.Sequential(
            nn.Linear(in_dim, nclass)
        )

    def forward(self, x):
        y_pred = self.Classifier(x)
        return y_pred
