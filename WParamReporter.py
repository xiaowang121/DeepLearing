# -*- coding: UTF-8 -*-
"""...................................................
                author:Xiaofei Wang
                date:2020.12.6
                Pytorch 1.6.1
                python 3.7
    ..................................................
"""
"""
To sort the output contact map of the model
"""
import torch
import numpy as np

def sortPredictedLab(pred_lab):
    pred_lab = torch.squeeze(pred_lab, 0)
    pred_lab = torch.squeeze(pred_lab[0], 0)

    Row = pred_lab.shape[0]
    Column = pred_lab.shape[1]

    pred_lab = pred_lab.cuda().data.cpu().numpy()
    pred_dict = {}
    mask = np.ones((Row, Column))
    mask = np.triu(mask, 1)
    pred_lab = pred_lab * mask
    for i in range(Row):
        for j in range(Column):
            pred_dict[i, j] = pred_lab[i, j]

    sorted_pred_lab_dict = dict(sorted(pred_dict.items(), key=lambda item: item[1], reverse=True))
    return sorted_pred_lab_dict
