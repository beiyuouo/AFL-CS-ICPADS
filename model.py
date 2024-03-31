import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from loguru import logger


class CNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=10, img_size: int = 28, init_weights=True):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * (img_size // 4) * (img_size // 4), 64)
        self.fc2 = nn.Linear(64, num_classes)

        self.img_size = img_size

        if init_weights:
            self._initialize_weights()

    def _initialize_weights(self, init_type="kaiming"):
        logger.info("initialize weights with {}".format(init_type))

        if init_type == "kaiming":
            for module in self.modules():
                if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                    nn.init.kaiming_normal_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
        elif init_type == "zeros":
            for module in self.modules():
                if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                    nn.init.zeros_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
        else:
            raise ValueError("unknown init type: {}".format(init_type))

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 64 * (self.img_size // 4) * (self.img_size // 4))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class CNN4(nn.Module):
    def __init__(self, in_channels=1, num_classes=10, img_size: int = 28, init_weights=True):
        super(CNN4, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.dropout2 = nn.Dropout2d(0.25)
        self.fc1 = nn.Linear(128 * (img_size // 4) * (img_size // 4), 512)
        self.dropout3 = nn.Dropout2d(0.25)
        self.fc2 = nn.Linear(512, num_classes)

        self.img_size = img_size

        if init_weights:
            self._initialize_weights()

    def _initialize_weights(self, init_type="kaiming"):
        logger.info("initialize weights with {}".format(init_type))

        if init_type == "kaiming":
            for module in self.modules():
                if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                    nn.init.kaiming_normal_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
        elif init_type == "zeros":
            for module in self.modules():
                if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                    nn.init.zeros_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
        else:
            raise ValueError("unknown init type: {}".format(init_type))

    def forward(self, x):
        # 4 conv layers + 2 max pool
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, 2)
        x = self.dropout2(x)
        x = x.view(-1, 128 * (self.img_size // 4) * (self.img_size // 4))
        x = F.relu(self.fc1(x))
        x = self.dropout3(x)
        x = self.fc2(x)

        return F.log_softmax(x, dim=1)


class ResNet18(nn.Module):
    def __init__(self, in_channels=3, num_classes=10, img_size: int = 32, init_weights=True, pretrained=False, **kwargs):
        super(ResNet18, self).__init__()
        self.resnet18 = torchvision.models.resnet18(pretrained=pretrained)
        self.resnet18.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.resnet18.fc = nn.Linear(512, num_classes)

        self.img_size = img_size

        if init_weights:
            self._initialize_weights()

    def _initialize_weights(self, init_type="kaiming"):
        logger.info("initialize weights with {}".format(init_type))

        if init_type == "kaiming":
            for module in self.modules():
                if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                    nn.init.kaiming_normal_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
        elif init_type == "zeros":
            for module in self.modules():
                if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                    nn.init.zeros_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
        else:
            raise ValueError("unknown init type: {}".format(init_type))

    def forward(self, x):
        x = self.resnet18(x)
        return F.log_softmax(x, dim=1)


def get_model(model_name, **kwargs):
    if model_name == "cnn":
        return CNN(**kwargs)
    elif model_name == "cnn4":
        return CNN4(**kwargs)
    elif model_name == "resnet18":
        return ResNet18(**kwargs)
    else:
        raise ValueError("Unknown model name: {}".format(model_name))

def build_optimizer(optimizer_name, model_params, optimizer_hyperparams):
    if optimizer_name == "sgd":
        return torch.optim.SGD(
            model_params,
            lr=optimizer_hyperparams["lr"],
            momentum=optimizer_hyperparams["momentum"],
        )
    elif optimizer_name == "adam":
        return torch.optim.Adam(model_params, lr=optimizer_hyperparams["lr"])
    else:
        raise ValueError("Unknown optimizer name: {}".format(optimizer_name))
    
def build_lr_scheduler(lr_scheduler_name, optimizer, lr_scheduler_hyperparams):
    if lr_scheduler_name == "step_lr":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=lr_scheduler_hyperparams["step_size"],
            gamma=lr_scheduler_hyperparams["gamma"],
        )
    elif lr_scheduler_name == "multi_step_lr":
        return torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=lr_scheduler_hyperparams["milestones"],
            gamma=lr_scheduler_hyperparams["gamma"],
        )
    elif lr_scheduler_name == "exponential_lr":
        return torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=lr_scheduler_hyperparams["gamma"]
        )
    else:
        raise ValueError("Unknown lr scheduler name: {}".format(lr_scheduler_name))

def build_loss_function(loss_function_name):
    if loss_function_name == "nll_loss":
        return torch.nn.NLLLoss()
    elif loss_function_name == "cross_entropy":
        return torch.nn.CrossEntropyLoss()
    else:
        raise ValueError("Unknown loss function name: {}".format(loss_function_name))
    