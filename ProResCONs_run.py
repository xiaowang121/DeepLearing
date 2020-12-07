# coding=utf-8
"""...................................................
                author:Xiaofei Wang
                date:2020.12.6
                Pytorch 1.6.1
                python 3.7
    ..................................................
"""
"""
ProResCONs's running program
"""
import os
import re
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable

from WMSADataTool import WContactDataset
from PRCs_net import PRCsNet

from WParamReporter import sortPredictedLab

"""
Select the version of the Pytorch running environment: GPU or CPU
"""
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
if not torch.cuda.is_available():
    print('There is no cuda.')
    pass

def test(model, input_file):

    print('On Testing Stage')
    model.eval()

    dataset = WContactDataset(input_file)

    for ind in range(dataset.__len__()):
        file_name = dataset.getfilename(ind)

        fea = dataset.getIthSampleFeaLab(ind)
        fea = torch.FloatTensor(fea)
        fea = Variable(fea).cuda()

        _, _, pred_lab = model(fea)

        sorted_pred_lab_dict = sortPredictedLab(pred_lab)

        keys = sorted_pred_lab_dict.keys()
        keys = list(keys)
        num = len(keys)
        values = sorted_pred_lab_dict.values()
        values = list(values)
        out_file = os.path.join(ProResCONs_outfile, file_name)
        with open(out_file, 'w') as W:
            for i in range(num):
                if values[i] != 0:
                    W.write(str(int(((str(keys[i]).replace("(","")).replace(")","")).split(",")[0])+1)+"  "+str(int(((str(keys[i]).replace("(","")).replace(")","")).split(",")[1])+1)+"  "+str(values[i]) + '\n')
    return pred_lab

if __name__ == '__main__':

###

    model_path = './model.ckpt'    ## Model location

    input_file = r'G:\CASP14\casp14_contact_prediction\msa\hh'   ## The input position of MSA
    ProResCONs_outfile = r'D:\123'   ## The output file location of the model
###

    model = PRCsNet().to(device)
    model.load_state_dict(torch.load(model_path))
    test(model, input_file)
    print(">>>>>>>>>>>>>>>Anything is OK !<<<<<<<<<<<<<<<<<<<<<")
