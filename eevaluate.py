import torch
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
import wandb
from etrainerfunctions import dicescore, sampler, pointwise_equivariance_error, rotate



@torch.inference_mode()
def etest(model, dataloader, device, amp, run_id=None, epoch='Latest', wandb_project='HeLa EUNet', experiment_started=True, experiment=None, etransforms=[], angle=None, class_weights=None, **kwargs):
    if class_weights is None:
        class_weights=[1 for j in range(model.n_classes)]
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, device=device)) if model.n_classes > 1 else nn.BCEWithLogitsLoss()
    model=model.to(device=device)
    model.eval()
    test_len = len(dataloader)
    dice=[0 for j in range(model.n_classes+1)]
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        eerror=0
        randerror=0
        loss = 0
        for batch in tqdm(dataloader, total=test_len, desc=f'Testing round, angle = {angle}', unit='batch', leave=False):
            images, true_masks = batch
            filter_tensor = rotate(angle)(torch.full(true_masks.shape, 1)).to(device=device)
            if wandb_project=='Equivariant UNet':
                images = images.to(dtype = torch.float32, device = device)
                true_masks=true_masks.squeeze(1)-1 #This -1 thing messes it up if rotate adds 0. That's why you fill 1 for oxford
                #print(true_masks.shape, torch.max(true_masks), torch.min(true_masks), true_masks.tolist())
                true_masks=F.one_hot(true_masks.to(torch.int64), model.n_classes).permute(0, 3, 1, 2).to(dtype=torch.float32, device=device)
            if wandb_project=='HeLa EUNet':
                true_masks=(true_masks/torch.max(true_masks)).to(dtype=torch.float32, device=device)
            images=images.to(dtype=torch.float32, device=device)
            #print(true_masks.shape, true_masks, torch.max(true_masks), torch.min(true_masks))
            masks_pred = model(images).to(device=device)
            masks_pred = masks_pred*filter_tensor
            #print(dice)
            #print(dicescore(masks_pred,true_masks))
            try:
                dice=[dice[0]+1] + [dice[x]+dicescore(masks_pred,true_masks, num_classes=model.n_classes)[x-1] for x in range(1,len(dice))]
            except:
                print(true_masks.shape, masks_pred.shape, dice)
            eerror+=test_equivariance_error(model, images, etransforms)
            randerror+=test_equivariance_error(model, sampler(images.shape[1:],n=model.n, cuda=True), etransforms)
            loss += criterion(masks_pred, true_masks) + 1 - dicescore(masks_pred,true_masks, num_classes=model.n_classes,average_classes=True, round=False)
            if dice[0]%(len(dataloader)/3+2)==0:
                experiment.log({'Test images': wandb.Image(images[0].cpu()),
                                'Test masks': {
                                        'true': wandb.Image(true_masks[0].float().cpu()),
                                        'pred': wandb.Image((torch.round(torch.sigmoid(masks_pred[0]))*(255)).float().cpu())
                                    }
                               })
        if run_id is not None:
            if experiment_started==False:
                experiment = wandb.init(project=wandb_project, id=run_id, resume="must")
        experiment.log({
                    f' Angle {angle} Test Dice Score' : {
                        j : float(dice[j]/dice[0]) for j in range(1,len(dice))
                    },
                    'Test Equivariance Error' : eerror/dice[0],
                    'Random Equivariance Error' : randerror/dice[0],
                    f'Angle {angle} Test Loss' : loss/dice[0]
            })


    model.train()
    return [float(dice[j]/dice[0]) for j in range(1,len(dice))]

#This is specifically for difference without taking logs
@torch.inference_mode()
def test_equivariance_error(model, tensor, etransforms):
    model.eval()
    errors=[]
    for f in model.etransforms:
            errors.append(pointwise_equivariance_error(model, tensor, f))
    error = sum(errors)/len(errors)                                                          
    model.train()
    return error

def dicescore(pred,target, ignore_index=None, average_classes=None, class_dim=1, num_classes=None, round=True):
    assert pred.shape == target.shape
    
    pred = torch.transpose(pred, class_dim, -1)
    pred = torch.sigmoid(pred)
    
    target = torch.transpose(target, class_dim, -1)
    if num_classes !=1:
        pred = torch.argmax(pred, dim=-1)
        pred = F.one_hot(pred,  num_classes=num_classes)
    else:
        if round:
            pred=torch.round(pred)
    split_pred = torch.split(pred,1,dim=-1)
    split_target = torch.split(target, 1, dim=-1)
    tp = [(x*split_target[j]).sum() for j,x in enumerate(split_pred) if j!=ignore_index]
    fp = [(x*(1-split_target[j])).sum() for j,x in enumerate(split_pred) if j!=ignore_index]
    fn = [((1-x)*(split_target[j])).sum() for j,x in enumerate(split_pred) if j!=ignore_index]
    scores=[2*tp[j]/(2*tp[j]+fp[j]+fn[j]) if tp[j]+fp[j]+fn[j] != 0 else 1 for j in range(len(tp))]
    if len(scores)==0:
        return 0
    if average_classes == None:
        return scores
    elif average_classes == True:
        return sum(scores)/len(scores)
    elif average_classes == 'weighted':
        return sum([scores[j]*(split_target[j].sum()) for j in range(len(scores))])/target.sum()