import torch
from torch import nn
from copy import deepcopy


class LLL_Net(nn.Module):
    """Basic class for implementing networks"""

    def __init__(self, model, is_cifar, remove_existing_head=False):
        head_var = model.head_var
        assert type(head_var) == str
        assert not remove_existing_head or hasattr(
            model, head_var
        ), "Given model does not have a variable called {}".format(head_var)
        assert not remove_existing_head or type(getattr(model, head_var)) in [
            nn.Sequential,
            nn.Linear,
        ], "Given model's head {} does is not an instance of nn.Sequential or nn.Linear".format(
            head_var
        )
        super(LLL_Net, self).__init__()

        self.model = model
        last_layer = getattr(self.model, head_var)

        if remove_existing_head:
            if type(last_layer) == nn.Sequential:
                self.out_size = last_layer[-1].in_features
                # strips off last linear layer of classifier
                del last_layer[-1]
            elif type(last_layer) == nn.Linear:
                self.out_size = last_layer.in_features
                # converts last layer into identity
                # setattr(self.model, head_var, nn.Identity())
                # WARNING: this is for when pytorch version is <1.2
                setattr(self.model, head_var, nn.Sequential())
        else:
            self.out_size = last_layer.out_features

        model_type = str(type(model)).lower()

        if is_cifar:
            if "resnet" in model_type:
                self.modify_to_cifar_resnet()

            elif "convnext" in model_type:
                self.modify_to_cifar_convnext()

        self.heads = nn.ModuleList()
        self.task_cls = []
        self.task_offset = []
        self._initialize_weights()

    def modify_to_cifar_resnet(self):
        self.model.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=2, bias=False
        )
        self.model.maxpool = nn.Identity()

    def modify_to_cifar_convnext(self):
        def change_kernel_size_to(model, shape):
            for name, layer in model.named_modules():
                if not isinstance(layer, nn.Conv2d):
                    continue

                if layer.kernel_size == (7, 7):
                    new_kernel_size = shape  # Define new kernel size
                    layer.kernel_size = new_kernel_size

        self.model.features[0][0] = nn.Conv2d(
            3, 96, kernel_size=3, stride=1, padding=2, bias=False
        )
        change_kernel_size_to(self.model, (4, 4))

    def add_head(self, num_outputs):
        """Add a new head with the corresponding number of outputs. Also update the number of classes per task and the
        corresponding offsets
        """
        self.heads.append(nn.Linear(self.out_size, num_outputs))
        # we re-compute instead of append in case an approach makes changes to the heads
        self.task_cls = torch.tensor([head.out_features for head in self.heads])
        self.task_offset = torch.cat(
            [torch.LongTensor(1).zero_(), self.task_cls.cumsum(0)[:-1]]
        )

    def forward(self, x, return_features=False):
        """Applies the forward pass

        Simplification to work on multi-head only -- returns all head outputs in a list
        Args:
            x (tensor): input images
            return_features (bool): return the representations before the heads
        """
        x = self.model(x)
        assert len(self.heads) > 0, "Cannot access any head"
        y = []
        for head in self.heads:
            y.append(head(x))
        if return_features:
            return y, x
        else:
            return y

    def get_copy(self):
        """Get weights from the model"""
        return deepcopy(self.state_dict())

    def set_state_dict(self, state_dict):
        """Load weights into the model"""
        self.load_state_dict(deepcopy(state_dict))
        return

    def freeze_all(self):
        """Freeze all parameters from the model, including the heads"""
        for param in self.parameters():
            param.requires_grad = False

    def freeze_backbone(self):
        """Freeze all parameters from the main model, but not the heads"""
        for param in self.model.parameters():
            param.requires_grad = False

    def freeze_bn(self):
        """Freeze all Batch Normalization layers from the model and use them in eval() mode"""
        for m in self.model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def unfreeze_bn(self):
        """Freeze all Batch Normalization layers from the model and use them in eval() mode"""
        for m in self.model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.requires_grad = True

    def unfreeze_all(self):
        """Freeze all parameters from the model, including the heads"""
        for param in self.parameters():
            param.requires_grad = True

    def freeze_last_head(self):
        for param in self.heads[-1].parameters():
            param.requires_grad = False

    def unfreeze_last_head(self):
        for param in self.heads[-1].parameters():
            param.requires_grad = True

    def _initialize_weights(self):
        """Initialize weights using different strategies"""
        # TODO: add different initialization strategies
        pass
