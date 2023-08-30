# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the \"License\");
# you may not use this file except in compliance with the License.\n",
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an \"AS IS\" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch.optim as optim
import torch.nn as nn
import torchvision.models as models


# Specify the Deep Neural model
class SequentialDNN(nn.Module):
    def __init__(self):
        """Defines a simple DNN for this template, it is recommended that you
        modify this to match your data / model needs.
        """
        super(SequentialDNN, self).__init__()

        self.sequential_model = nn.Sequential(
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
            nn.ReLU(),
            nn.Linear(4, 1),
        )

    def forward(self, x):
        return self.sequential_model(x)

def create(args, device):
    """
    Create the model, loss function, and optimizer to be used for the DNN

    Args:
      args: experiment parameters.
    """
    if args.model_type == "simsiam":
        #base_model = args.base_model
        base_model = models.__dict__[args.base_model]
        sequential_model = SimSiam(
            base_encoder=base_model, dim=args.dim, pred_dim=args.pred_dim
        ).to(device)
        criterion = nn.CosineSimilarity(dim=1).to(device)
        if args.optimizer == "adam":
            optimizer = optim.Adam(
                sequential_model.parameters(),
                lr=args.learning_rate,
                weight_decay=args.weight_decay,
            )
        else:
            raise NotImplementedError

    return sequential_model, criterion, optimizer


class SimSiam(nn.Module):
    """
    Build a SimSiam model.
    """

    def __init__(self, base_encoder, dim=128, pred_dim=16):
        """
        dim: feature dimension (default: 2048)
        pred_dim: hidden dimension of the predictor (default: 512)
        """
        super(SimSiam, self).__init__()

        # create the encoder
        # num_classes is the output fc dimension, zero-initialize last BNs
        self.encoder = base_encoder(num_classes=dim, zero_init_residual=True)
        self.encoder.conv1 = nn.Conv2d(
            1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
        )
        # build a 3-layer projector
        prev_dim = self.encoder.fc.weight.shape[1]
        self.encoder.fc = nn.Sequential(
            nn.Linear(prev_dim, prev_dim, bias=False),
            nn.BatchNorm1d(prev_dim),
            nn.ReLU(inplace=True),  # first layer
            nn.Linear(prev_dim, prev_dim, bias=False),
            nn.BatchNorm1d(prev_dim),
            nn.ReLU(inplace=True),  # second layer
            self.encoder.fc,
            nn.BatchNorm1d(dim, affine=False),
        )  # output layer
        self.encoder.fc[
            6
        ].bias.requires_grad = False  # hack: not use bias as it is followed by BN

        # build a 2-layer predictor
        self.predictor = nn.Sequential(
            nn.Linear(dim, pred_dim, bias=False),
            nn.BatchNorm1d(pred_dim),
            nn.ReLU(inplace=True),  # hidden layer
            nn.Linear(pred_dim, dim),
        )  # output layer

    def forward(self, x1, x2):
        """
        Input:
            x1: first views of images
            x2: second views of images
        Output:
            p1, p2, z1, z2: predictors and targets of the network
            See Sec. 3 of https://arxiv.org/abs/2011.10566 for detailed notations
        """

        # compute features for one view
        z1 = self.encoder(x1)  # NxC
        z2 = self.encoder(x2)  # NxC

        p1 = self.predictor(z1)  # NxC
        p2 = self.predictor(z2)  # NxC

        return p1, p2, z1.detach(), z2.detach()
    


