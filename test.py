import time
import os
import numpy as np
import torch
import torchvision
import torchvision.models as models
import torch.utils.model_zoo as model_zoo
import torchvision.transforms as transforms
from torchvision import datasets
from functools import reduce
from itertools import accumulate


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

models_to_test = ['squeezenet1_1']
batch_size = 20
use_gpu = torch.cuda.is_available()
device = torch.device('cuda' if use_gpu else 'cpu')


def diff_states(dict_canonical, dict_subset):
    keys1, keys2 = dict_canonical.keys(), dict_subset.keys()
    missing1 = [k for k in keys1 if k not in keys2]
    missing2 = [k for k in keys2 if k not in keys1]
    assert not missing1 and not missing2, f"State key mismatch: {missing1 or missing2}"
    for name, v1 in dict_canonical.items():
        v2 = dict_subset[name]
        if v1.size() != v2.size():
            yield name, v1


def load_defined_model(name, num_classes):
    model = models.__dict__[name](num_classes=num_classes)
    if name == 'densenet169':
        model = torchvision.models.DenseNet(
            num_init_features=64, growth_rate=32,
            block_config=(6, 12, 32, 32), num_classes=num_classes
        )
    pretrained = model_zoo.load_url(model_urls[name])
    diff = list(diff_states(model.state_dict(), pretrained))
    if diff:
        print(f"Replacing unmatched layers in {name}: {[d[0] for d in diff]}")
        for key, val in diff:
            pretrained[key] = val
    model.load_state_dict(pretrained)
    return model


def load_data(resize):
    transforms_map = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(max(resize)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
        'val': transforms.Compose([
            transforms.Resize(int(max(resize) / 224 * 256)),
            transforms.CenterCrop(max(resize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
    }
    data_dir = 'plantvillage dataset'
    datasets_map = {x: datasets.ImageFolder(os.path.join(data_dir), transforms_map[x])
                    for x in ['train', 'val']}
    loaders = {
        x: torch.utils.data.DataLoader(
            datasets_map[x], batch_size=batch_size,
            shuffle=(x == 'train'), num_workers=4,
            pin_memory=use_gpu
        )
        for x in ['train', 'val']
    }
    return loaders['train'], loaders['val']


def load_model(net):
    path = 'saved_Models/plant_village/Plant_Village_saved_model_Squeeze_Net.pth.tar'
    checkpoint = torch.load(path, map_location=device)
    net.load_state_dict(checkpoint['state_dict'])
    return net.to(device)


def evaluate_stats(net, testloader):
    net.eval()
    correct = 0
    total = 0
    start = time.time()
    with torch.no_grad():
        for images, labels in testloader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = net(images)
            _, preds = torch.max(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    accuracy = correct / total if total else 0.0
    elapsed = time.time() - start
    print(f'Accuracy on test set: {accuracy:.4f}')
    return {'accuracy': accuracy, 'eval_time': elapsed}


def main():
    stats = []
    num_classes = 39
    print("Evaluating pretrained model")
    for name in models_to_test:
        print(f"\nModel: {name} with {num_classes} classes")
        model = load_defined_model(name, num_classes)
        model = torch.nn.DataParallel(model) if use_gpu else model
        model.to(device)
        _, testloader = load_data(
            [s for k, s in {'squeezenet': (224, 224)}.items() if k in name][0]
        )
        model = load_model(model)
        stats_eval = evaluate_stats(model, testloader)
        stats_eval.update({'name': name})
        stats.append(stats_eval)

    import csv
    with open('test_stats.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=stats[0].keys())
        writer.writeheader()
        writer.writerows(stats)

if __name__ == '__main__':
    main()
