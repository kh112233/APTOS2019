import torch

import pandas as pd

def tester(model, test_dataloader, verbose = 2, with_cuda: bool = True):

    cuda_condition = torch.cuda.is_available() and with_cuda
    device = torch.device("cuda:0" if cuda_condition else "cpu")

    outputs_list = torch.zeros([])
    sample_csv = pd.read_csv("./dataset/input/sample_submission.csv")

    model.cuda()
    model.eval()

    with torch.no_grad():
        for i, batch in enumerate(test_dataloader):
            images = batch
            images = images.type(torch.FloatTensor).cuda()

            outputs = model(images)
            outputs = outputs.squeeze(-1)
            if(i==0):
                outputs_list = outputs
            else:
                outputs_list = torch.cat([outputs_list, outputs], -1) 

    save_result(outputs_list, sample_csv)

def save_result(outputs_list, sample_csv):
    sample_csv["diagnosis"] = torch.round(outputs_list.cpu()).detach()
    sample_csv.to_csv("./outputs.csv", index=False)
