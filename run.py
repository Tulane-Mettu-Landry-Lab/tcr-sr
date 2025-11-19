import warnings
warnings.filterwarnings('ignore')
from utils import (
    DataModule, ModelLib, ModelModule, EvalMetrics
)
from utils.data import TCRXAIDataset
import json
import os
import argparse


def run_exp(configs, device=None, batch_size=None, num_workers=None, epoch=None):
    save_path = configs['model_configs']['training_config']['save_path']
    if device:
        configs['model_configs']['device'] = device
    if batch_size:
        configs['trainset_config']['batch_size'] = batch_size
        configs['testset_config']['batch_size'] = batch_size
    if num_workers:
        configs['trainset_config']['num_workers'] = num_workers
        configs['testset_config']['num_workers'] = num_workers
    if epoch:
        configs['model_configs']['training_config']['epoch'] = epoch
    os.makedirs(save_path, exist_ok=True)
    with open(os.path.join(save_path, 'config.json'), 'w') as f:
        json.dump(configs, f, indent=2)
    
    print('[ACT] Load Train Dataset')    
    traindm = DataModule.from_config(configs['trainset_config'])
    print('[ACT] Load Test Dataset')  
    testdm = DataModule.from_config(configs['testset_config'])
    regdm = None
    if 'regset_config' in configs:
        print('[ACT] Load Regularization Dataset')  
        regdm = DataModule.from_config(configs['regset_config'], dataset_class=TCRXAIDataset)
    print('[ACT] Build Model')  
    modelmodule = ModelModule.from_config(configs['model_configs'])
    reg_config = None
    if 'reg_config' in configs:
        print('[ACT] Build Regularization Config')
        reg_config = configs['reg_config']
        reg_config['dataloader'] = regdm.dataloader
    print('[ACT] Train Model') 
    modelmodule.train(traindm, resume=True, reg_config=reg_config)
    print('[ACT] Evaluate Test Dataset') 
    y_true, y_pred = modelmodule.test(testdm, weights=configs['test_config']['weights'])
    evalmetrics = EvalMetrics.from_config(configs['eval_config'])
    evalresults = evalmetrics.eval(y_true, y_pred, testdm)
    with open(os.path.join(save_path, 'eval.json'), 'w') as f:
        json.dump(evalresults, f, indent=2)
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Experiment")
    parser.add_argument("--config", type=str, required=True, help="Path to config file or config string")
    parser.add_argument("--device", type=str, default=None, help="Device for model")
    parser.add_argument("--batchsize", type=int, default=None, help="Batch size for train and test")
    parser.add_argument("--numworkers", type=int, default=None, help="Number of CPU workers for data loader")
    parser.add_argument("--epoch", type=int, default=None, help="Number of epoches")
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        configs = json.load(f)
    run_exp(
        configs,
        device=args.device,
        batch_size=args.batchsize,
        num_workers=args.numworkers,
        epoch=args.epoch,
    )