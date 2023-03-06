import collections
from torch import nn
from timm import create_model


def get_model() -> nn.Sequential:
    net = create_model(
        "vit_tiny_patch16_224", pretrained=False, num_classes=0, in_chans=3
    )

    head = nn.Sequential(
        nn.BatchNorm1d(192),
        nn.Dropout(0.25),
        nn.Linear(192, 512, bias=False),
        nn.ReLU(inplace=True),
        nn.BatchNorm1d(512),
        nn.Dropout(0.5),
        nn.Linear(512, 37, bias=False),
    )

    return nn.Sequential(net, head)


def copy_weight(name, parameter, state_dict):
    """
    Takes in a layer `name`, model `parameter`, and `state_dict`
    and loads the weights from `state_dict` into `parameter`
    if it exists.
    """
    # Part of the body
    if name[0] == "0":
        name = name[:2] + "model." + name[2:]
    if name in state_dict.keys():
        input_parameter = state_dict[name]
        if input_parameter.shape == parameter.shape:
            parameter.copy_(input_parameter)
        else:
            print(f"Shape mismatch at layer: {name}, skipping")
    else:
        print(f"{name} is not in the state_dict, skipping.")


def apply_weights(
    input_model: nn.Module,
    input_weights: collections.OrderedDict,
    application_function: callable,
):
    """
    Takes an input state_dict and applies those weights to the `input_model`, potentially
    with a modifier function.

    Args:
        input_model (`nn.Module`):
            The model that weights should be applied to
        input_weights (`collections.OrderedDict`):
            A dictionary of weights, the trained model's `state_dict()`
        application_function (`callable`):
            A function that takes in one parameter and layer name from `input_model`
            and the `input_weights`. Should apply the weights from the state dict into `input_model`.
    """
    model_dict = input_model.state_dict()
    for name, parameter in model_dict.items():
        application_function(name, parameter, input_weights)
    input_model.load_state_dict(model_dict)
