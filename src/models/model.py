from collections import OrderedDict
import json
import os
import posixpath
import struct
import tempfile

import numpy as np
import requests
import torch
from torch import nn
import torch.nn.functional as F

BASE_DIR = os.path.join(tempfile.gettempdir(), "_posenet_weights")


def prepare_model_to_train(keypoint_num=16, model_version=50, device="cpu"):
    """Load model and weights by model version,
    put it to our device,
    and change the last layers
    """
    model = MobileNetV1(model_version)
    # Load weights
    checkpoint = MOBILENET_V1_CHECKPOINTS[model_version]
    state_dict = load_variables(checkpoint)
    model.load_state_dict(state_dict)

    model.to(device)

    model.heatmap = (
        nn.Conv2d(model.last_depth, keypoint_num + 1, 2, 2).double().to(device)
    )
    model.offset = (
        nn.Conv2d(model.last_depth, keypoint_num * 2, 2, 2).double().to(device)
    )
    return model


def _to_output_strided_layers(convolution_def, output_stride):
    """."""
    current_stride = 1
    rate = 1
    block_id = 0
    buff = []
    for c in convolution_def:
        conv_type = c[0]
        inp = c[1]
        outp = c[2]
        stride = c[3]
        if current_stride == output_stride:
            layer_stride = 1
            layer_rate = rate
            rate *= stride
        else:
            layer_stride = stride
            layer_rate = 1
            current_stride *= stride

        buff.append(
            {
                "block_id": block_id,
                "conv_type": conv_type,
                "inp": inp,
                "outp": outp,
                "stride": layer_stride,
                "rate": layer_rate,
                "output_stride": current_stride,
            }
        )
        block_id += 1
    return buff


def _get_padding(kernel_size, stride, dilation):
    """."""
    padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2
    return padding


class InputConv(nn.Module):
    """."""

    def __init__(self, inp, outp, k=3, stride=1, dilation=1):
        """."""
        super().__init__()
        self.conv = nn.Conv2d(
            inp,
            outp,
            k,
            stride,
            padding=_get_padding(k, stride, dilation),
            dilation=dilation,
        )

    def forward(self, x):
        """."""
        return F.relu6(self.conv(x))


class SeperableConv(nn.Module):
    """."""

    def __init__(self, inp, outp, k=3, stride=1, dilation=1):
        """."""
        super().__init__()
        self.depthwise = nn.Conv2d(
            inp,
            inp,
            k,
            stride,
            padding=_get_padding(k, stride, dilation),
            dilation=dilation,
            groups=inp,
        )
        self.pointwise = nn.Conv2d(inp, outp, 1, 1)

    def forward(self, x):
        """."""
        x = F.relu6(self.depthwise(x))
        x = F.relu6(self.pointwise(x))
        return x


MOBILENET_V1_CHECKPOINTS = {
    50: "mobilenet_v1_050",
    75: "mobilenet_v1_075",
    100: "mobilenet_v1_100",
    101: "mobilenet_v1_101",
}

MOBILE_NET_V1_100 = [
    (InputConv, 3, 32, 2),
    (SeperableConv, 32, 64, 1),
    (SeperableConv, 64, 128, 2),
    (SeperableConv, 128, 128, 1),
    (SeperableConv, 128, 256, 2),
    (SeperableConv, 256, 256, 1),
    (SeperableConv, 256, 512, 2),
    (SeperableConv, 512, 512, 1),
    (SeperableConv, 512, 512, 1),
    (SeperableConv, 512, 512, 1),
    (SeperableConv, 512, 512, 1),
    (SeperableConv, 512, 512, 1),
    (SeperableConv, 512, 1024, 2),
    (SeperableConv, 1024, 1024, 1),
]

MOBILE_NET_V1_75 = [
    (InputConv, 3, 24, 2),
    (SeperableConv, 24, 48, 1),
    (SeperableConv, 48, 96, 2),
    (SeperableConv, 96, 96, 1),
    (SeperableConv, 96, 192, 2),
    (SeperableConv, 192, 192, 1),
    (SeperableConv, 192, 384, 2),
    (SeperableConv, 384, 384, 1),
    (SeperableConv, 384, 384, 1),
    (SeperableConv, 384, 384, 1),
    (SeperableConv, 384, 384, 1),
    (SeperableConv, 384, 384, 1),
    (SeperableConv, 384, 384, 1),
    (SeperableConv, 384, 384, 1),
]

MOBILE_NET_V1_50 = [
    (InputConv, 3, 16, 2),
    (SeperableConv, 16, 32, 1),
    (SeperableConv, 32, 64, 2),
    (SeperableConv, 64, 64, 1),
    (SeperableConv, 64, 128, 2),
    (SeperableConv, 128, 128, 1),
    (SeperableConv, 128, 256, 2),
    (SeperableConv, 256, 256, 1),
    (SeperableConv, 256, 256, 1),
    (SeperableConv, 256, 256, 1),
    (SeperableConv, 256, 256, 1),
    (SeperableConv, 256, 256, 1),
    (SeperableConv, 256, 256, 1),
    (SeperableConv, 256, 256, 1),
]

GOOGLE_CLOUD_STORAGE_DIR = "https://storage.googleapis.com/tfjs-models/weights/posenet/"


def download_json(checkpoint, filename, base_dir):
    """."""
    url = posixpath.join(GOOGLE_CLOUD_STORAGE_DIR, checkpoint, filename)
    response = requests.get(url)
    data = json.loads(response.content)

    with open(os.path.join(base_dir, checkpoint, filename), "w") as outfile:
        json.dump(data, outfile)


def download_file(checkpoint, filename, base_dir):
    """."""
    url = posixpath.join(GOOGLE_CLOUD_STORAGE_DIR, checkpoint, filename)
    response = requests.get(url)
    f = open(os.path.join(base_dir, checkpoint, filename), "wb")
    f.write(response.content)
    f.close()


def download(checkpoint, base_dir="./weights/"):
    """."""
    save_dir = os.path.join(base_dir, checkpoint)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    download_json(checkpoint, "manifest.json", base_dir)

    f = open(os.path.join(save_dir, "manifest.json"), "r")
    json_dict = json.load(f)

    for x in json_dict:
        filename = json_dict[x]["filename"]
        print("Downloading", filename)
        download_file(checkpoint, filename, base_dir)


def to_torch_name(tf_name):
    """."""
    tf_name = tf_name.lower()
    tf_split = tf_name.split("/")
    tf_layer_split = tf_split[1].split("_")
    tf_variable_type = tf_split[2]
    if tf_variable_type == "weights" or tf_variable_type == "depthwise_weights":
        variable_postfix = ".weight"
    elif tf_variable_type == "biases":
        variable_postfix = ".bias"
    else:
        variable_postfix = ""

    if tf_layer_split[0] == "conv2d":
        torch_name = "features.conv" + tf_layer_split[1]
        if len(tf_layer_split) > 2:
            torch_name += "." + tf_layer_split[2]
        else:
            torch_name += ".conv"
        torch_name += variable_postfix
    else:
        if (
            tf_layer_split[0] in ["offset", "displacement", "heatmap"]
            and tf_layer_split[-1] == "2"
        ):
            torch_name = "_".join(tf_layer_split[:-1])
            torch_name += variable_postfix
        else:
            torch_name = ""

    return torch_name


def load_variables(checkpoint, base_dir=BASE_DIR):
    """."""
    manifest_path = os.path.join(base_dir, checkpoint, "manifest.json")
    if not os.path.exists(manifest_path):
        if (
            base_dir
            == "/usr/local/lib/python3.8/dist-packages/posenet/converter/config.yaml"
        ):
            print("yiii")

        print(
            "Weights for checkpoint %s are not downloaded. Downloading to %s ..."
            % (checkpoint, base_dir)
        )
        # from posenet.converter.wget import download
        download(checkpoint, base_dir)
        assert os.path.exists(manifest_path)

    manifest = open(manifest_path)
    variables = json.load(manifest)
    manifest.close()

    state_dict = {}
    for x in variables:
        torch_name = to_torch_name(x)
        if not torch_name:
            continue
        filename = variables[x]["filename"]
        byte = open(os.path.join(base_dir, checkpoint, filename), "rb").read()
        fmt = str(int(len(byte) / struct.calcsize("f"))) + "f"
        d = struct.unpack(fmt, byte)
        d = np.array(d, dtype=np.float32)
        shape = variables[x]["shape"]
        if len(shape) == 4:
            tpt = (2, 3, 0, 1) if "depthwise" in filename else (3, 2, 0, 1)
            d = np.reshape(d, shape).transpose(tpt)
        state_dict[torch_name] = torch.Tensor(d)

    return state_dict


def convert(model_id, model_dir, output_stride=4):
    """."""
    checkpoint_name = MOBILENET_V1_CHECKPOINTS[model_id]

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    state_dict = load_variables(checkpoint_name)
    m = MobileNetV1(model_id, output_stride=output_stride)
    m.load_state_dict(state_dict)
    checkpoint_path = os.path.join(model_dir, checkpoint_name) + ".pth"
    torch.save(m.state_dict(), checkpoint_path)


class MobileNetV1(nn.Module):
    """."""

    def __init__(self, model_id, output_stride=4):
        """."""
        super().__init__()

        assert model_id in MOBILENET_V1_CHECKPOINTS.keys()
        # How many classes for classification (in this case how many points we need to find)
        self.output_stride = output_stride

        # Choose how many layers we will use, and their sizes
        if model_id == 50:
            arch = MOBILE_NET_V1_50
        elif model_id == 75:
            arch = MOBILE_NET_V1_75
        else:
            arch = MOBILE_NET_V1_100

        # Get the settings for our version of model
        conv_def = _to_output_strided_layers(arch, output_stride)
        conv_list = [
            (
                "conv%d" % c["block_id"],
                c["conv_type"](
                    c["inp"], c["outp"], 3, stride=c["stride"], dilation=c["rate"]
                ),
            )
            for c in conv_def
        ]
        # Output size of last layer
        last_depth = conv_def[-1]["outp"]
        self.last_depth = last_depth
        self.features = nn.Sequential(OrderedDict(conv_list))

        self.heatmap = nn.Conv2d(last_depth, 17, 1, 1)
        self.offset = nn.Conv2d(last_depth, 34, 1, 1)

        self.displacement_fwd = nn.Conv2d(last_depth, 32, 1, 1)
        self.displacement_bwd = nn.Conv2d(last_depth, 32, 1, 1)
        self.drop = nn.Dropout2d(0)

        self.double()

    def forward(self, x):
        """."""
        x = self.features(x)
        # Classify blocks
        heatmap = torch.sigmoid(self.heatmap(x))
        # Predict coordinates within a block
        offset = self.offset(x)
        return heatmap, offset
