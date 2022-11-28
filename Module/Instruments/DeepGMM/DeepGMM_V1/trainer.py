import torch
import numpy as np
from .methods.toy_model_selection_method import ToyModelSelectionMethod as DeepGMM_True
from utils import set_seed, cat

def run(exp, data, train_dict, log, device, resultDir, others):
    set_seed(train_dict['seed'])
    print(f"Run {exp}/{train_dict['reps']}")

    data.cuda()
    data.double()

    train_dict['stage1_dim'] = train_dict['z_dim'] + train_dict['x_dim']
    train_dict['stage2_dim'] = train_dict['t_dim'] + train_dict['x_dim']
    train_dict['stage1_layers'] = [20, ]
    train_dict['stage2_layers'] = [20, 3]

    train_stage_1 = torch.cat([data.train.z, data.train.x],1).double()
    train_stage_2 = torch.cat([data.train.t, data.train.x],1).double()
    val_stage_1 = torch.cat([data.valid.z, data.valid.x],1).double()
    val_stage_2 = torch.cat([data.valid.t, data.valid.x],1).double()
    test_stage_2 = torch.cat([data.test.t, data.test.x],1).double()

    method = DeepGMM_True(train_dict, torch.cuda.is_available())
    method.fit(train_stage_2, train_stage_1, data.train.y.double(), val_stage_2, val_stage_1, data.valid.y.double(), g_dev=data.valid.v.double(), verbose=True)

    def estimation(data):
        input0 = torch.cat([data.t-data.t, data.x],1).double()
        point0 = method.predict(input0)

        inputt = torch.cat([data.t, data.x],1).double()
        pointt = method.predict(inputt)

        return point0, pointt

    return estimation