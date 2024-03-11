import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import random_split

def get_dataset(dataset, augment, standardize, val_len):
    if dataset == "cifar100":
        data_mu = (0.5071, 0.4867, 0.4408)
        data_sigma = (0.2675, 0.2565, 0.2761)
        crop = 32
    elif dataset == "cifar10":
        data_mu = (0.4914, 0.4822, 0.4465)
        data_sigma = (0.2471, 0.2435, 0.2616)
        crop = 32
    elif dataset == "tinyimagenet":
        data_mu = (0.4802, 0.4481, 0.3975)
        data_sigma = (0.2302, 0.2265, 0.2202)
        crop = 64
    else:
        raise AssertionError(f"{dataset} not a supported dataset")

    transform_train = [transforms.ToTensor()]
    transform_test = [transforms.ToTensor()]
    if augment:
        transform_train += [transforms.RandomCrop(crop, padding=4),
                 transforms.RandomHorizontalFlip()]
    if standardize:
        transform_train += [transforms.Normalize(data_mu, data_sigma)]
        transform_test += [transforms.Normalize(data_mu, data_sigma)]
    transform_train = transforms.Compose(transform_train)
    transform_test = transforms.Compose(transform_test)
    if dataset == "cifar100":
        train_data = datasets.CIFAR100("./data/CIFAR100", train=True, transform=transform_train, download=True)
        test_data = datasets.CIFAR100("./data/CIFAR100", train=False, transform=transform_test, download=True)
    elif dataset == "cifar10":
        train_data = datasets.CIFAR10("./data/CIFAR10", train=True, transform=transform_train, download=True)
        test_data = datasets.CIFAR10("./data/CIFAR10", train=False, transform=transform_test, download=True)
    elif dataset == "tinyimagenet":
        train_data = datasets.ImageFolder('./data/tiny-imagenet-200/train', transform=transform_train)
        test_data = datasets.ImageFolder('./data/tiny-imagenet-200/val', transform=transform_test)
    else:
        raise AssertionError(f"{dataset} not a supported dataset")

    train_len = len(train_data)
    val_len = val_len
    train_len -= val_len
    train_data, val_data = random_split(train_data, [train_len, val_len])
    return train_data, val_data, test_data

def get_model(model, dataset, pooling):
    # We will not use max pooling as it is too expensive
    assert(pooling != "max")
    if dataset == "tinyimagenet":
        num_classes = 200
    elif dataset == "cifar100":
        num_classes = 100
    elif dataset == "cifar10":
        num_classes = 10
    else:
        raise AssertionError(f"{dataset} not a supported dataset")

    from PyTorch_CIFAR10.cifar10_models.resnet import resnet18, resnet34, resnet50
    from PyTorch_CIFAR10.cifar10_models.vgg import vgg11_bn, vgg16_bn
    if model == "resnet18":
        net = resnet18(pretrained=False, num_classes=num_classes, pooling=pooling)
    elif model == "resnet34":
        net = resnet34(pretrained=False, num_classes=num_classes, pooling=pooling)
    elif model == "resnet50":
        net = resnet50(pretrained=False, num_classes=num_classes, pooling=pooling)
    elif model == "vgg11_bn":
        net = vgg11_bn(pretrained=False, num_classes=num_classes, pooling=pooling)
    elif model == "vgg16_bn":
        net = vgg16_bn(pretrained=False, num_classes=num_classes, pooling=pooling)
    else:
        raise AssertionError()

    return net
