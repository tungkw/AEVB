import matplotlib.pyplot as plt
import torch
import dataset
from torch.utils.data import DataLoader
from AEVB_model import *

import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default="FreyFace", help="'FreyFace' or 'MINST'")
parser.add_argument('--data_path', type=str, default='./datasets/FreyFace/', help="path to the root of selected dataset")
parser.add_argument('--batch_size', type=int, default=100, help='batch size')
parser.add_argument('--latent_dim', type=int, default=10, help='dimension of latent parameter z')
parser.add_argument('--hidden_dim', type=int, default=200, help='dimension of hidden layer of MLP')
parser.add_argument('--learning_rate', type=float, default=0.01, help='learing rate')
parser.add_argument('--epoch', type=int, default=10000, help='iteration over the whole dataset')
args = parser.parse_args()

if __name__ == "__main__":

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)

    if args.data == 'FreyFace':
        dataset = dataset.FreyFaceDataset(args.data_path)
        model = AEVB(dataset.sample_dim, args.latent_dim, args.hidden_dim, data='FreyFace')
    elif args.data == 'MINST':
        dataset = dataset.MINSTDataset(args.data_path)
        model = AEVB(dataset.sample_dim, args.latent_dim, args.hidden_dim, data='MINST')
    else:
        print("wrong data name")
        exit(1)
    model.to(device=torch.cuda.current_device())
    print(model)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)


    '''initialize'''
    for para in model.parameters():
        torch.nn.init.normal_(para,0,0.01)


    '''train'''
    plt.figure()
    record = []
    
    optimizer = torch.optim.Adagrad(model.parameters(),lr=args.learning_rate)
    cnt = 0
    for k in range(args.epoch):
        for i,batch in enumerate(data_loader):
            if batch.size()[0] < args.batch_size:
                continue
            model.zero_grad()
            lower_bound = model(batch.to(device=device))

            loss = - lower_bound
            loss.backward()
            optimizer.step()

            cnt+=1
            if cnt % 1000 == 0:
                print(cnt, "lower bound:", lower_bound.item())
                record.append(lower_bound)

    plt.plot(record)
    plt.savefig(os.path.join("./output/", args.data+"_result.png"))
    # plt.show()

