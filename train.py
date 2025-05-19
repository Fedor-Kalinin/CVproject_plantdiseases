import time
import os

import numpy as np
import torch
import torch.optim as optim
from torch import nn
from torch.autograd import Variable

import torchvision
import torchvision.models as models
import torch.utils.model_zoo as model_zoo
import torchvision.transforms as transforms
from torchvision import datasets

from itertools import accumulate
from functools import reduce

#configuration
model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
    'densenet121': 'https://download.pytorch.org/models/densenet121-241335ed.pth',
    'densenet169': 'https://download.pytorch.org/models/densenet169-6f0f7f60.pth',
    'densenet201': 'https://download.pytorch.org/models/densenet201-4c113574.pth',
    'densenet161': 'https://download.pytorch.org/models/densenet161-17b70270.pth',
    'inception_v3': 'https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth',
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',    
    'squeezenet1_0': 'https://download.pytorch.org/models/squeezenet1_0-a815701f.pth',
    'squeezenet1_1': 'https://download.pytorch.org/models/squeezenet1_1-f364aa15.pth',
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',    
}

model_names = model_urls.keys()

input_sizes = {
    'alexnet' : (224,224),
    'densenet': (224,224),
    'resnet' : (224,224),
    'inception' : (299,299),
    'squeezenet' : (224,224),
    'vgg' : (224,224)
}


models_to_test = ['squeezenet1_1']  

batch_size = 20
use_gpu = torch.cuda.is_available()


def diff_states(dict_canonical, dict_subset):
    names1, names2 = (list(dict_canonical.keys()), list(dict_subset.keys()))
    

    not_in_1 = [n for n in names1 if n not in names2]
    not_in_2 = [n for n in names2 if n not in names1]
    assert len(not_in_1) == 0
    assert len(not_in_2) == 0

    for name, v1 in dict_canonical.items():
        v2 = dict_subset[name]
        assert hasattr(v2, 'size')
        if v1.size() != v2.size():
            yield (name, v1)                

def load_defined_model(name, num_classes):
    
    model = models.__dict__[name](num_classes=num_classes)
    
    if name == 'densenet169':
        model = torchvision.models.DenseNet(num_init_features=64, growth_rate=32, \
                                            block_config=(6, 12, 32, 32), num_classes=num_classes)
        
    pretrained_state = model_zoo.load_url(model_urls[name])

    diff = [s for s in diff_states(model.state_dict(), pretrained_state)]
    print("Replacing the following state from initialized", name, ":", \
          [d[0] for d in diff])
    
    for name, value in diff:
        pretrained_state[name] = value
    
    assert len([s for s in diff_states(model.state_dict(), pretrained_state)]) == 0
    
    model.load_state_dict(pretrained_state)
    return model, diff


def filtered_params(net, param_list=None):
    def in_param_list(s):
        for p in param_list:
            if s.endswith(p):
                return True
        return False    

    params = net.named_parameters() if param_list is None \
    else (p for p in net.named_parameters() if in_param_list(p[0]))
    return params


def load_data(resize):

    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(max(resize)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(int(max(resize)/224*256)),
            transforms.CenterCrop(max(resize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    data_dir = 'plantvillage dataset'
    dsets = {x: datasets.ImageFolder(os.path.join(data_dir), data_transforms[x])
             for x in ['train', 'val']}
    dset_loaders = {x: torch.utils.data.DataLoader(dsets[x], batch_size=batch_size,
                                                   shuffle=True)
                    for x in ['train', 'val']}
    dset_sizes = {x: len(dsets[x]) for x in ['train', 'val']}
    dset_classes = dsets['train'].classes
    
    return dset_loaders['train'], dset_loaders['val']



def train(net, trainloader, param_list=None, epochs=15, use_gpu=True):

    def in_param_list(name):
        return any(name.endswith(p) for p in param_list)

    device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
    net.to(device)

    criterion = nn.CrossEntropyLoss().to(device)

    if param_list:
        for name, param in net.named_parameters():
            if not in_param_list(name):
                param.requires_grad = False

    trainable_params = (p for p in net.parameters() if p.requires_grad)

    optimizer = optim.SGD(trainable_params, lr=0.001, momentum=0.9)

    save_model(net)

    losses = []
    for epoch in range(epochs):
        running_loss = 0.0

        for i, (inputs, labels) in enumerate(trainloader, 0):
            inputs = inputs.to(device,   non_blocking=True)
            labels = labels.to(device,   non_blocking=True)

            optimizer.zero_grad()

            outputs = net(inputs)

            if isinstance(outputs, tuple):
                loss = sum(criterion(o, labels) for o in outputs)
            else:
                loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 30 == 29:
                avg_loss = running_loss / 30
                losses.append(avg_loss)
                print(f'[{epoch+1}, {i+1:5d}] loss: {avg_loss:.3f}')
                running_loss = 0.0
                save_model(net)

        save_model(net)

    print('Finished Training')
    return losses







def save_model(net):
    if not os.path.exists("saved_models/plant_village"):
        os.mkdir("saved_models/plant_village")
    state_dic = {'task_name': "Plant_Village", 'state_dict': net.state_dict()}
    filename = "./saved_models/plant_village/Plant_Village_saved_model_Squeeze_Net.pth.tar"
    torch.save(state_dic, filename)
    print("Model Saved")

def load_model(net):
    filename = "Plant_Village_saved_model_Squeeze_Net.pth.tar"      
    checkpoint = torch.load(filename)
    net.load_state_dict(checkpoint['state_dict'])
    return net.eval()


def train_stats(m, trainloader, param_list = None):
    stats = {}
    params = filtered_params(m, param_list)    
    counts = 0,0
    for counts in enumerate(accumulate((reduce(lambda d1,d2: d1*d2, p[1].size()) for p in params)) ):
        pass
    stats['variables_optimized'] = counts[0] + 1
    stats['params_optimized'] = counts[1]
    
    before = time.time()
    losses = train(m, trainloader, param_list=param_list)
    stats['training_time'] = time.time() - before

    stats['training_loss'] = losses[-1] if len(losses) else float('nan')
    stats['training_losses'] = losses
    
    return stats



def evaluate_stats(net, testloader, device):
    net.eval()  
    correct = 0
    total = 0

    start_time = time.time()
    with torch.no_grad():
        for images, labels in testloader:
            images = images.to(device,   non_blocking=True)
            labels = labels.to(device,   non_blocking=True)

            outputs = net(images)                        
            _, predicted = torch.max(outputs, dim=1)     

            total += labels.size(0)
            correct += (predicted == labels).sum().item()  

    accuracy = correct / total
    eval_time = time.time() - start_time

    print(f'Accuracy on test images: {accuracy:.4f}')
    net.train()

    return {
        'accuracy': accuracy,
        'eval_time': eval_time
    }




def train_eval(net, trainloader, testloader, param_list=None):
    print("Training..." if not param_list else "Retraining...")
    stats_train = train_stats(net, trainloader, param_list=param_list)
    
    print("Evaluating...")
    net = net.eval()
    stats_eval = evaluate_stats(net, testloader)
    
    return {**stats_train, **stats_eval}

stats = []
num_classes = 39
print("RETRAINING")

for name in models_to_test:
    print("")
    print("Targeting %s with %d classes" % (name, num_classes))
    print("------------------------------------------")
    model_pretrained, diff = load_defined_model(name, num_classes)
    final_params = [d[0] for d in diff]
    
    resize = [s[1] for s in input_sizes.items() if s[0] in name][0]
    print("Resizing input images to max of", resize)
    trainloader, testloader = load_data(resize)
    
    if use_gpu:
        print("Transfering models to GPU(s)")
        model_pretrained = torch.nn.DataParallel(model_pretrained).cuda()
        
    pretrained_stats = train_eval(model_pretrained, trainloader, testloader, final_params)
    pretrained_stats['name'] = name
    pretrained_stats['retrained'] = True
    pretrained_stats['shallow_retrain'] = True
    stats.append(pretrained_stats)
    
    print("")

exit() 

print("---------------------")
print("TRAINING from scratch")
for name in models_to_test:
    print("")    
    print("Targeting %s with %d classes" % (name, num_classes))
    print("------------------------------------------")
    model_blank = models.__dict__[name](num_classes=num_classes)

    resize = [s[1] for s in input_sizes.items() if s[0] in name][0]
    print("Resizing input images to max of", resize)
    trainloader, testloader = load_data(resize)
    
    if use_gpu:
        print("Transfering models to GPU(s)")
        model_blank = torch.nn.DataParallel(model_blank).cuda()    
        
    blank_stats = train_eval(model_blank, trainloader, testloader)
    blank_stats['name'] = name
    blank_stats['retrained'] = False
    blank_stats['shallow_retrain'] = False
    stats.append(blank_stats)
    
    print("")

t = 0.0
for s in stats:
    t += s['eval_time'] + s['training_time']
print("Total time for training and evaluation", t)
print("FINISHED")

print("RETRAINING deep")

for name in models_to_test:
    print("")
    print("Targeting %s with %d classes" % (name, num_classes))
    print("------------------------------------------")
    model_pretrained, diff = load_defined_model(name, num_classes)
    
    resize = [s[1] for s in input_sizes.items() if s[0] in name][0]
    print("Resizing input images to max of", resize)
    trainloader, testloader = load_data(resize)
    
    if use_gpu:
        print("Transfering models to GPU(s)")
        model_pretrained = torch.nn.DataParallel(model_pretrained).cuda()
        
    pretrained_stats = train_eval(model_pretrained, trainloader, testloader, None)
    pretrained_stats['name'] = name
    pretrained_stats['retrained'] = True
    pretrained_stats['shallow_retrain'] = False
    stats.append(pretrained_stats)
    
    print("")


import csv
with open('stats.csv', 'w') as csvfile:
    fieldnames = stats[0].keys()
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    for s in stats:
        writer.writerow(s)