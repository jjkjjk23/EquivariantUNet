import argparse
import logging
import os
import random
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import torchmetrics
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import numpy as np
import time
import math

import wandb
from evaluate import evaluate
from unet import UNet
from utils.data_loading import BasicDataset, CarvanaDataset
from utils.dice_score import dice_loss
from eevaluate import etest
import functools

import PIL

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class HeLaDataset(torch.utils.data.Dataset):
    def __init__(self, train_file, label_file, n_images, transform=None, target_transform=None):
        super().__init__()
        self.images=PIL.Image.open(train_file)
        self.images.load()
        self.labels=PIL.Image.open(label_file)
        self.labels.load()
        #self.tensor=torchvision.transforms.PILToTensor(PIL.Image.open(file))
        self.n_images=n_images
        self.transform=transform
        self.target_transform=target_transform
        
    def __getitem__(self, idx):
        self.images.seek(idx)
        self.labels.seek(idx)
        if self.transform:
            return self.transform(torch.unsqueeze(torch.from_numpy(np.array(self.images)),0)), self.target_transform(torch.unsqueeze( torch.from_numpy(np.array(self.labels)),0))
        else:
            return torch.unsqueeze(torch.from_numpy(np.array(self.images)),0), torch.unsqueeze(torch.from_numpy(np.array(self.labels)),0)

        #return self
    def __len__(self):
        return self.n_images


class CombinedDataset(torch.utils.data.Dataset):
    def __init__(self, datasets):
        self.datasets = datasets
        self.numused = [0 for j in self.datasets]
        self.notexhausted = datasets

    def __getitem__(self, idx):
        while self.datasets:
            dataset = random.choice(self.notexhausted)
            i=self.datasets.index(dataset)
            self.numused[i]+=1
            if self.numused[i]==len(dataset)-1:
                self.notexhausted.remove(dataset)
            if not self.notexhausted:
                self.notexhausted = self.datasets
            currindex=(idx-sum([num for j , num in enumerate(self.numused) if j!=i]))%len(dataset)
            return dataset.__getitem__(currindex)

        raise StopIteration
    #def __iter__(self):
        #return self
    def __len__(self):
        return sum([len(j) for j in self.datasets])
    def reset(self):
        self.notexhausted=self.datasets
        self.numused=[0 for j in self.datasets]
        
class ScaleJitteredDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset=dataset
    def __getitem__(self, idx):
        image, true_mask = self.dataset.__getitem__(idx)
        a=random.uniform(.25,4)
        image=torchvision.transforms.Resize((math.floor(224*a),math.floor(224*a)))(image)
        image=torchvision.transforms.Resize((224,224))(image)
        true_mask=torchvision.transforms.Resize((math.floor(224*a),math.floor(224*a)))(true_mask)
        true_mask=torchvision.transforms.Resize((224,224))(true_mask)
        return image, true_mask
    def __len__(self):
        return len(self.dataset)
    
class RandomAngleDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset=dataset
    def __getitem__(self, idx):
        image, true_mask = self.dataset.__getitem__(idx)
       
        angle=random.uniform(-10,10)
        image=torchvision.transforms.functional.rotate(image, angle)
        true_mask=torchvision.transforms.functional.rotate(true_mask, angle)
        return image, true_mask
    def __len__(self):
        return len(self.dataset)
    
deformation = torch.tensor([[0,0,0],[0,1,0],[0,0,0]]).to(device=device, dtype=torch.float32)
deformation=torch.unsqueeze(deformation, dim=0)
deformation= torchvision.transforms.Resize((512,512), interpolation=TF.InterpolationMode.BICUBIC)(deformation)
ydeformation =torch.full([1,512,512], 0, device=device, dtype=torch.float32)
deformation = torch.stack([deformation, ydeformation], dim=3)

def rotate(angle):
            return lambda inputs : torchvision.transforms.functional.rotate(inputs, angle)
        
def shift(x, shiftnum=1, axis=-1):
    x=torch.transpose(x, axis, -1)
    if shiftnum == 0:
        padded = x
    elif shiftnum > 0:
      #paddings = (0, shift, 0, 0, 0, 0)
        paddings = [0 for j in range(2*len(tuple(x.shape)))]
        paddings[1]=shiftnum
        paddings=tuple(paddings)
        padded = nn.functional.pad(x[..., shiftnum:], paddings)
    elif shiftnum < 0:
        #paddings = (-shift, 0, 0, 0, 0, 0)
        paddings = [0 for j in range(2*len(tuple(x.shape)))]
        paddings[0]=-shiftnum
        paddings=tuple(paddings)
        padded = nn.functional.pad(x[..., :shiftnum], paddings)
    else:
        raise ValueError
    return torch.transpose(padded, axis,-1)
        
def shifty(shiftnum, axis):
    return lambda x : shift(x, shiftnum, axis)
def deform(tensor):
    return TF.elastic_transform(tensor, deformation, TF.InterpolationMode.NEAREST, 0.0)
    
class RandomDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset=dataset
    def __getitem__(self, idx):
        image, true_mask = self.dataset.__getitem__(idx)
        transform_or_not = random.randint(0,1)
        #if transform_or_not == 0:
            #return image, true_mask
        stacked = torch.stack((image, true_mask), dim=0)
        angle=random.uniform(-10,10)
        shiftnum = random.randint(-6,6)
        axis = random.randint(-2,-1)
        
        functions = [shifty(shiftnum, axis), rotate(angle), torchvision.transforms.ElasticTransform(interpolation=TF.InterpolationMode.NEAREST)]
            
        randcombo = functools.reduce(compose, functions)
        stacked=randcombo(stacked)
        
        image=stacked[0]
        true_mask=stacked[1]
        return image, true_mask
    def __len__(self):
        return len(self.dataset)
    
    
#split must equal 'test' or 'trainval'
def config_data(etransforms=None, augmented=False, split='trainval', batch_size=1, **kwargs):
    totensor=torchvision.transforms.PILToTensor()
    resize=torchvision.transforms.Resize((224,224))
    resizedtensor=torchvision.transforms.Compose([resize,totensor])
    if kwargs['Oxford']:
        dataset0 = torchvision.datasets.OxfordIIITPet(root="C:\\Users\\jjkjj\\Equivariant\\", split=split, target_types='segmentation',
                                        download=True, transform=resizedtensor, target_transform=resizedtensor)
    if kwargs['HeLa']:
        dataset0= HeLaDataset(f"C:\\Users\\jjkjj\\Equivariant\\ISBI-2012-challenge\\{split.removesuffix('val')}-volume.tif",f"C:\\Users\\jjkjj\\Equivariant\\ISBI-2012-challenge\\{split.removesuffix('val')}-labels.tif", 30)
    
    if augmented == True:
        if kwargs['Oxford']:       
            transformed_datasets=[torchvision.datasets.OxfordIIITPet(root="C:\\Users\\jjkjj\\Equivariant\\", split=split, target_types='segmentation', download=True, transform=torchvision.transforms.Compose([resizedtensor,f[1]]), target_transform=torchvision.transforms.Compose([resizedtensor,f[0]])) for f in etransforms]
        if kwargs['HeLa']:  
            transformed_datasets=[HeLaDataset(f"C:\\Users\\jjkjj\\Equivariant\\ISBI-2012-challenge\\{split.removesuffix('val')}-volume.tif",f"C:\\Users\\jjkjj\\Equivariant\\ISBI-2012-challenge\\{split.removesuffix('val')}-labels.tif", 30, transform=f[1], target_transform=f[0]) for f in etransforms]

        all_datasets=[dataset0]+transformed_datasets
        dataset=CombinedDataset(all_datasets)
    
        
    elif augmented==False:
        dataset=dataset0
    elif augmented=='random':
        dataset = ScaleJitteredDataset(dataset0)
    elif augmented=='rangle':
        dataset = RandomAngleDataset(dataset0)
    elif augmented=='randcombo':
        dataset=RandomDataset(dataset0)
    
    # 2. Split into train / validation partitions
    if split=='trainval' and kwargs['Oxford']:
        n_val = 200
        n_train = len(dataset) - n_val
        train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))
        train_loader = DataLoader(train_set, shuffle=True, batch_size=batch_size)
        val_loader = DataLoader(val_set, shuffle=False, drop_last=True, batch_size=batch_size)
        return train_loader, val_loader
   
    return DataLoader(dataset, shuffle=True if split=='trainval' else False, batch_size=batch_size)

def train_model(
        model,
        device,
        train_loader,
        val_loader=None,
        epochs: int = 5,
        batch_size: int = 1,
        learning_rate: float = 1e-5,
        save_checkpoint: bool = True,
        amp: bool = False,
        weight_decay: float = 1e-8,
        momentum: float = 0.999,
        gradient_clipping: float = 1.0,
        class_weights=None,
        epochbreaks=False,
        augmented=False,
        debugging=False,
        **kwargs
):

    if class_weights is None:
        class_weights=[1 for j in range(model.n_classes)]
    # (Initialize logging)
    if debugging:
        print('This is running in debugging mode and is not logging with wandb')
    if not debugging:
        experiment = wandb.init(project=kwargs['wandb_project'], resume='allow', anonymous='must')
        experiment.config.update(
            dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
                 save_checkpoint=save_checkpoint, amp=amp,
                 equivariance_measure=model.equivariance_measure,
                 etransforms = model.etransforms,
                 equivariant = model.equivariant,
                 Linf =model.Linf,
                 eqweight=model.eqweight,
                 n=model.n,
                 class_weights=class_weights,
                 augmented = augmented,
                 test_augmented = kwargs['test augmented']
                )
        )

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Mixed Precision: {amp}
        #equivariance_measure: {model.equivariance_measure}
        transforms: {model.etransforms}
        equivariant: {model.equivariant}
        Linf: {model.Linf}
        eqweight: {model.eqweight}
        n: {model.n}
        augmented: {augmented}
    ''')

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.RMSprop(model.parameters(),
                              lr=learning_rate, weight_decay=weight_decay, momentum=momentum, foreach=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=25, eps=0)  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, device=device)) if model.n_classes > 1 else nn.BCEWithLogitsLoss()
    global_step = 0
    time0=time.time()
    # 5. Begin training
    for epoch in range(1, epochs + 1):
        timeepoch=time.time()
        #dice[0] is total dice score dice[0] is steps since dice was cleared
        dice=[0 for j in range(model.n_classes+1)]
        model.train()
        epoch_loss = 0
        with tqdm(total=len(train_loader)*batch_size, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            
            for batch in train_loader:
                #images, true_masks = batch['image'], batch['mask']
                images, true_masks = batch
                if kwargs['wandb_project']=='Equivariant UNet':
                    true_masks=true_masks.squeeze(1)-1
                    true_masks=F.one_hot(true_masks.to(torch.int64), model.n_classes).permute(0, 3, 1, 2).to(dtype=torch.float32, device=device)
                if kwargs['wandb_project']=='HeLa EUNet':
                    true_masks=(true_masks/torch.max(true_masks)).to(dtype=torch.float32, device=device)
                    images=images.to(dtype=torch.float32, device=device)

                assert images.shape[1] == model.n_channels, \
                    f'Network has been defined with {model.n_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                #images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                #true_masks = true_masks.to(device=device, dtype=torch.float32)

                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp): #, enabled=amp):
                    masks_pred = model(images).to(device=device)
                    f=random.choice(model.etransforms)
                    if kwargs['eqerror']==True:
                        randomval=sampler((1,512,512),n=model.n, cuda=True)
                        stackedval = torch.cat((randomval, model(randomval)), dim=0)
                        fval = f[0](stackedval)
                        if model.Linf == False:
                            equivariance_err=torch.mean(torch.abs(torch.log(torch.sigmoid(model(fval[0:1])))-torch.log(torch.sigmoid(fval[1:2]))))
                        else: equivariance_err=torch.max(torch.abs(torch.log(torch.sigmoid(model(fval[0:1])))-torch.log(torch.sigmoid(fval[1:2]))))
                            
                    else:
                        equivariance_err=-1

                    #print(masks_pred.shape, masks_pred)
                    if model.equivariant:
                        loss0 = criterion(masks_pred, true_masks) + 1 - sum(dicescore(masks_pred,true_masks, num_classes=model.n_classes, round=False))
                        """
                        loss += dice_loss(
                            F.softmax(masks_pred, dim=1).float(),
                            true_masks.float(),
                            multiclass=True
                        )
                        """
                        loss = loss0 + model.eqweight*equivariance_err
                    else:
                        loss = criterion(masks_pred, true_masks) + 1 - sum(dicescore(masks_pred,true_masks, num_classes=model.n_classes, round=False))
                        """
                        loss += dice_loss(
                                    F.softmax(masks_pred, dim=1).float(),
                                    true_masks.float(),
                                    multiclass=True
                        )
                        """

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                grad_scaler.step(optimizer)
                grad_scaler.update()
                
                pbar.update(images.shape[0])
                global_step += 1
                dice=[dice[0]+1] + [dice[x]+dicescore(masks_pred,true_masks, num_classes=model.n_classes)[x-1] for x in range(1,len(dice))]
                epoch_loss += loss.item()
                if not debugging:
                    experiment.log({
                        'train loss': loss0.item() if model.equivariant else loss.item(),
                        'step': global_step,
                        'epoch': epoch,
                        'equivariance error' : equivariance_err,
                        'Training Dice score' : {
                            j : float(dice[j]/dice[0]) for j in range(1,len(dice))
                        },
                        'Time into epoch': time.time()-timeepoch,
                        'Total time' : time.time()-time0
                    })
                pbar.set_postfix(**{'Cumulative Dice' : [float(dice[j]/dice[0]) for j in range(1,len(dice))], 'loss (batch)': loss.item()})
            
                # Evaluation round
                division_step = (len(train_loader) // (5))
                if division_step > 0:
                    if global_step % division_step == 0:
                        scheduler.step(sum(dice[1:]))
                        histograms = {}
                        if not debugging:
                            for tag, value in model.named_parameters():
                                tag = tag.replace('/', '.')
                                if not (torch.isinf(value) | torch.isnan(value)).any():
                                    histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                                if not (torch.isinf(value.grad) | torch.isnan(value.grad)).any():
                                    histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())
                        """
                        #val_score = evaluate(model, val_loader, device, amp)
                        #scheduler.step(torch.mean(val_score))
                        for batch in val_loader:
                            images, true_masks = batch
                            true_masks=true_masks.squeeze(1)-1
                            images=images.to(dtype=torch.float32, device=device)
                            #print(true_masks.shape, true_masks, torch.max(true_masks), torch.min(true_masks))
                            true_masks=F.one_hot(true_masks.to(torch.int64), model.n_classes).permute(0, 3, 1, 2).to(dtype=torch.float32, device=device)
                            #print(true_masks.shape, true_masks)
            
                            assert images.shape[1] == model.n_channels, \
                                f'Network has been defined with {model.n_channels} input channels, ' \
                                f'but loaded images have {images.shape[1]} channels. Please check that ' \
                                'the images are loaded correctly.'

                            #images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                            #true_masks = true_masks.to(device=device, dtype=torch.float32)

                            with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp): #, enabled=amp):
                                masks_pred = model(images).to(device=device)
                            dice=[dice[0]+1] + [dice[x]+dicescore(masks_pred,true_masks)[x-1] for x in range(1,len(dice))]
                        """
                        
                        #logging.info('Validation Dice score: {}'.format([float(dice[j]/dice[0])for j in range(1,len(dice))]))
                        #try:
                        if not debugging:
                            if kwargs['wandb_project']=='Equivariant UNet':
                                experiment.log({
                                    'learning rate': optimizer.param_groups[0]['lr'],
                                    #'validation Dice': {j : float(dice[j]/dice[0]) for j in range(1,len(dice))},
                                    'images': wandb.Image(images[0].cpu()),
                                    #'transformed' : {
                                        #f'Function {j+1}' : wandb.Image(f[0](images[0])) for j, f in enumerate(model.etransforms)
                                    #},
                                    'masks': {
                                        'true': wandb.Image(true_masks[0].float().cpu()),
                                        'pred': wandb.Image(masks_pred.argmax(dim=1)[0].float().cpu())
                                    },
                                    'Transformed mask' : wandb.Image((f[0](masks_pred))[0].float().cpu()),
                                    'Prediction for Transform' : wandb.Image(model(f[0](images[0:1])[0])[0].float().cpu()),
                                    'step': global_step,
                                    'epoch': epoch,
                                    **histograms
                                })
                            if kwargs['wandb_project']=='HeLa EUNet':
                                experiment.log({
                                    'learning rate': optimizer.param_groups[0]['lr'],
                                    #'validation Dice': {j : float(dice[j]/dice[0]) for j in range(1,len(dice))},
                                    'images': wandb.Image(images[0].cpu()),
                                    #'transformed' : {
                                        #f'Function {j+1}' : wandb.Image(f[0](images[0])) for j, f in enumerate(model.etransforms)
                                    #},
                                    'masks': {
                                        'true': wandb.Image(true_masks[0].float().cpu()),
                                        'pred': wandb.Image((torch.round(torch.sigmoid(masks_pred[0]))*(255)).float().cpu())
                                    },
                                    'Transformed mask' : wandb.Image((f[0](masks_pred)).float().cpu()),
                                    'Prediction for Transform' : wandb.Image(model(f[0](images[0:1]))[0].float().cpu()),
                                    'step': global_step,
                                    'epoch': epoch,
                                    **histograms
                                })
                        dice=[0 for j in range(len(dice))]
                        #except:
                            #pass
        
        if save_checkpoint:
            if not debugging:
                dir_checkpoint = Path(f'./{str(experiment.name)}_checkpoints/')
                Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
                state_dict = model.state_dict()
                #state_dict['mask_values'] = dataset.mask_values
                torch.save(state_dict, str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch)))
                logging.info(f'Checkpoint {epoch} saved!')
                if 'test_on_epoch_end' in kwargs:
                    if kwargs['test_on_epoch_end']:
                        etest(model, config_data(HeLa=True, Oxford=False, split='test', augmented=kwargs['test augmented'], **kwargs), device, amp, run_id=experiment.id, epoch=epoch, experiment_started=True, experiment=experiment, etransforms=model.etransforms)
        if epochbreaks:
            print('Taking a break')
            time.sleep(kwargs['break_length'])
                
            
def randomscale(tensor):
        a=random.uniform(.25,4)
        tensor=torchvision.transforms.Resize((math.floor(224*a),math.floor(224*a)))(tensor)
        return torchvision.transforms.Resize((224,224))(tensor)
    
def pad24(inputs):
        return torch.nn.functional.pad(inputs, (0,24,0,24), mode='constant', value=1)

def sampler(shape, bounds=None, n=100,cuda=False):
    shape=(n,)+shape
    if bounds==None:
        output= torch.from_numpy(np.random.default_rng().random(size=shape)).to(torch.float32)
    elif type(bounds)==list:
        bounds=np.array(bounds)
        bounds=np.broadcast_to(bounds,shape+(2,))
        output = torch.from_numpy(np.random.default_rng().random(size=shape)*(bounds[...,1]-bounds[...,0])+bounds[...,0]).to(torch.float32)
    if cuda==True:
        return output.cuda()
    else:
        return output

def integrate(integrand, shape, bounds, n=100, measure=1.0, error=False, cuda=False):
    values=integrand(sampler(shape, bounds, n, cuda=cuda))
    if error==True:
        return (measure*torch.mean(values, dim=0), measure*torch.std(values, dim=0)/math.sqrt(n))
    if error==False:
        return measure*torch.mean(values, dim=0)
    
def Linffn(integrand, shape, bounds, n=100, cuda=False):
    sampled_vals=sampler(shape, bounds, n, cuda=cuda)
    values=integrand(sampler(shape, bounds, n, cuda=cuda))
    return torch.max(values, dim=0).values

def compose(function2,function1):
    return lambda *args : function2(function1(*args))

def l2integrand(f):
    return lambda inputs : torch.sum((f[0](inputs)-f[1](inputs))**2)

def CEintegrand(f):
    return lambda inputs : torch.max(torch.abs(torch.log(f[0](inputs))-torch.log(f[1](inputs))))

def linfintegrand(f):
    return lambda inputs : torch.max(torch.abs(f[0](inputs)-f[1](inputs)))

def l1integrand(f):
    return lambda inputs : torch.mean(torch.abs(f[0](inputs)-f[1](inputs)))
    
def args2list(*args):
    if len(args)==1:
        return args
    else:
        return [arg for arg in args]
    
def expand0(tensor):
    return torch.unsqueeze(tensor, axis=0)

def e(inputs):
    return inputs

def zero(inputs):
    return 0*inputs

def shift(x, shift=1, axis=-1):
    x=torch.transpose(x, axis, -1)
    if shift == 0:
        padded = x
    elif shift > 0:
        #paddings = (0, shift, 0, 0, 0, 0)
        paddings = [0 for j in range(2*len(tuple(x.shape)))]
        paddings[1]=shift
        paddings=tuple(paddings)
        padded = nn.functional.pad(x[..., shift:], paddings)
    elif shift < 0:
        #paddings = (-shift, 0, 0, 0, 0, 0)
        paddings = [0 for j in range(2*len(tuple(x.shape)))]
        paddings[0]=-shift
        paddings=tuple(paddings)
        padded = nn.functional.pad(x[..., :shift], paddings)
    else:
        raise ValueError
    return torch.transpose(padded, axis,-1)
            
def model2func(model):
    return lambda inputs : model.forward(inputs)

def equivariance_error(model, eintegrand, functions, shape, bounds=None, n=50, cuda=False, Linf=False):
    equivariance_measures={'linf' : linfintegrand, 'l2' : l2integrand, 'l1' : l1integrand}
    if type(eintegrand)==str:
        eintegrand=equivariance_measures[eintegrand]
    if bounds != None:
        if type(bounds)==list:
            bounds=np.array(bounds)
        measure = np.prod((bounds[...,1]-bounds[...,0]))
    if bounds == None:
        measure=1
    realintegrand=lambda f: eintegrand([compose(f[0],model),compose(model,f[1])])
    if Linf==False:
        return [torch.nn.functional.relu(integrate(realintegrand(f), shape, bounds, n=n, measure=measure,cuda=cuda)-f[2]) for f in functions]
    else:
        return [torch.nn.functional.relu(Linffn(realintegrand(f), shape, bounds, n=n,cuda=cuda)-f[2]) for f in functions]

#Calculate dice score for one-hot encoded targets and predictions
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
    
        
        
