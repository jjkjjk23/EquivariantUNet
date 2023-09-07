import torch
import torch.nn.functional as F
from tqdm import tqdm
import wandb
from etrainerfunctions import dicescore, sampler

@torch.inference_mode()
def etest(model, dataloader, device, amp, run_id=None, epoch='Latest', wandb_project='HeLa EUNet', experiment_started=True, experiment=None, etransforms=[], **kwargs):
    model=model.to(device=device)
    model.eval()
    test_len = len(dataloader)
    dice=[0 for j in range(model.n_classes+1)]
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        eerror=0
        randerror=0
        for batch in tqdm(dataloader, total=test_len, desc='Testing round', unit='batch', leave=False):
            images, true_masks = batch
            if wandb_project=='Equivariant UNet':
                true_masks=true_masks.squeeze(1)-1
                true_masks=F.one_hot(true_masks.to(torch.int64), model.n_classes).permute(0, 3, 1, 2).to(dtype=torch.float32, device=device)
            if wandb_project=='HeLa EUNet':
                true_masks=(true_masks/torch.max(true_masks)).to(dtype=torch.float32, device=device)
            images=images.to(dtype=torch.float32, device=device)
            #print(true_masks.shape, true_masks, torch.max(true_masks), torch.min(true_masks))
            masks_pred = model(images).to(device=device)
            #print(dice)
            #print(dicescore(masks_pred,true_masks))
            try:
                dice=[dice[0]+1] + [dice[x]+dicescore(masks_pred,true_masks, num_classes=model.n_classes)[x-1] for x in range(1,len(dice))]
            except:
                print(true_masks.shape, masks_pred.shape, dice)
            eerror+=pointwise_equivariance_error(model, images, etransforms)
            randerror+=pointwise_equivariance_error(model, sampler((1,512,512),n=model.n, cuda=True), etransforms)
        if run_id is not None:
            if experiment_started==False:
                experiment = wandb.init(project=wandb_project, id=run_id, resume="must")
        experiment.log({
                    'Test Dice Score' : {
                        j : float(dice[j]/dice[0]) for j in range(1,len(dice))
                    },
                    'Test Equivariance Error' : eerror/dice[0],
                    'Random Equivariance Error' : randerror/dice[0]
            })
    model.train()
    return [float(dice[j]/dice[0]) for j in range(1,len(dice))]

#This is specifically for difference without taking logs
@torch.inference_mode()
def pointwise_equivariance_error(model, tensor, etransforms):
    model.eval()
    errors=[]
    for f in model.etransforms:
        stackedval = torch.cat((tensor, model(tensor)), dim=0)
        fval = f[0](stackedval)
        if model.Linf:
            errors.append(torch.max(torch.abs(torch.log(torch.sigmoid(model(fval[0:1])))-torch.log(torch.sigmoid(fval[1:2])))))
        else:
            errors.append(torch.mean(torch.abs(torch.log(torch.sigmoid(model(fval[0:1])))-torch.log(torch.sigmoid(fval[1:2])))))
    error = sum(errors)/len(errors)                                                          
    model.train()
    return error