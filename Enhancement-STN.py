'''
Python program source code
for research article "Enhancement Spatial Transformer Networks for Text Classification"

Version 1.0
(c) Copyright 2020 Ka-Hou Chan <chankahou (at) ipm.edu.mo>

The python program source code is free software: you can redistribute
it and/or modify it under the terms of the GNU General Public License
as published by the Free Software Foundation, either version 3 of the License,
or (at your option) any later version.

The python program source code is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for
more details.

You should have received a copy of the GNU General Public License
along with the Kon package.  If not, see <http://www.gnu.org/licenses/>.
'''

import torch
import torchvision

#https://www.itl.nist.gov/iaui/vip/cs_links/EMNIST/Readme.txt
train_set = torchvision.datasets.EMNIST(root='./emnist/', split="mnist", train=True, transform=torchvision.transforms.ToTensor(), download=True)
#print(train_set.train_data.type(), train_set.train_data.shape) #.ByteTensor torch.Size([60000, 28, 28])
#print(train_set.train_labels.type(), train_set.train_labels.shape) #torch.LongTensor torch.Size([60000])
test_set = torchvision.datasets.EMNIST(root='./emnist/', split="mnist", train=False, transform=torchvision.transforms.ToTensor(), download=True)
#print(test_set.train_data.type(), test_set.train_data.shape) #torch.ByteTensor torch.Size([10000, 28, 28])
#print(test_set.train_labels.type(), test_set.train_labels.shape) #torch.LongTensor torch.Size([10000])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#####################################################################################

class SelfAdaptive(torch.nn.Module):
    def __init__(self, degree_poly):
        super(SelfAdaptive, self).__init__()
        self.perimeter = torch.arange(degree_poly, dtype=torch.float).to(device)
        self.linear = torch.nn.Linear(degree_poly, 1, bias=False)
        self.linear.weight.data.zero_()

    def forward(self, input):
        t = input.unsqueeze(-1)
        t = torch.cos(self.perimeter*torch.acos(t))
        t = self.linear(t).squeeze(-1)
        return t

class SinPN(torch.nn.Module):
    def __init__(self, N):
        super(SinPN, self).__init__()
        self.N = N

    def forward(self, input):
        return torch.sin(input) + self.N*input

#####################################################################################

class STN_CNN(torch.nn.Module):
    def __init__(self):
        super(STN_CNN, self).__init__()

        self.Translation = torch.nn.Sequential(
            SelfAdaptive(16),
            torch.nn.Linear(28*28, 2)
            )  #[tx, ty]
        self.Translation[-1].weight.data.zero_()
        self.Translation[-1].bias.data.zero_()

        self.Scaling = torch.nn.Sequential(
            SelfAdaptive(16),
            torch.nn.Linear(28*28, 2)
            )  #[sx, sy]
        self.Scaling[-1].weight.data.zero_()
        self.Scaling[-1].bias.data.fill_(1.0)

        self.Rotation = torch.nn.Sequential(
            SelfAdaptive(16),
            torch.nn.Linear(28*28, 1)
            )  #[a]
        self.Rotation[-1].weight.data.zero_()
        self.Rotation[-1].bias.data.fill_(0.0)

        self.Convolution = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=7, padding=3),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(32, 64, kernel_size=5, padding=2),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
            )

        self.FullyConnected = torch.nn.Sequential(
            torch.nn.Linear(64*7*7, 256),
            SinPN(2.0),
            torch.nn.Linear(256, 10)
            )

    def forward(self, input):

        Txy = self.Translation(input.view(-1, 28*28))
        matT = torch.tensor([[1.0,0.0,0.0, 0.0,1.0,0.0]], device=device).repeat(input.size(0), 1)
        matT[:,[2,5]] = Txy

        Sxy = self.Scaling(input.view(-1, 28*28))
        matS = torch.diag_embed(Sxy)

        A = self.Rotation(input.view(-1, 28*28))
        matR = torch.tensor([[1.0,0.0, 0.0,1.0]], device=device).repeat(input.size(0), 1)
        matR[:,[0,3]] = torch.cos(A)
        matR[:,[2]] = torch.sin(A)
        matR[:,[1]] = -matR[:,[2]]

        matRS = torch.matmul(matR.view(-1,2,2), matS)
        matRST = torch.matmul(matRS, matT.view(-1,2,3))

        grid = torch.nn.functional.affine_grid(matRST, input.size(), align_corners=True)
        transformed = torch.nn.functional.grid_sample(input, grid, align_corners=True)
 
        feature = self.Convolution(transformed)
        return self.FullyConnected(feature.view(-1, 64*7*7))

#####################################################################################

epoch = 0
nn = STN_CNN().to(device)
optimizer = torch.optim.Adagrad(nn.parameters())
#optimizer = torch.optim.Adam(nn.parameters())
print(nn)
print(optimizer)

train_itr = torch.utils.data.DataLoader(dataset=train_set, batch_size=100, shuffle=True)
test_itr = torch.utils.data.DataLoader(dataset=test_set, batch_size=100, shuffle=False)
loss_func = torch.nn.CrossEntropyLoss()

while (epoch < 100):
    torch.cuda.empty_cache()

    nn.train()
    with torch.enable_grad():
        for step, (train_data, train_label) in enumerate(train_itr):
            optimizer.zero_grad()

            train_data = train_data.to(device)
            train_label = train_label.to(device)
            label = nn(train_data)
            loss = loss_func(label, train_label)
            #print('Epoch: ', epoch, '| Train Loss: %.4f' % loss.data)

            loss.backward()
            optimizer.step()

    epoch += 1

    nn.eval()
    with torch.no_grad():
        accuracy = 0.0
        for step, (test_data, test_label) in enumerate(test_itr):
            test_data = test_data.to(device)
            test_label = test_label.to(device)
            label = nn(test_data)
            loss = loss_func(label, test_label)
            accuracy += (label.max(-1)[1] == test_label).sum()
        accuracy /= len(test_set)
        print('Epoch: ', epoch, '| Test Accuracy: %.4f' % accuracy)
