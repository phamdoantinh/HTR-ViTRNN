import os
import re
import sys
import argparse
from collections import OrderedDict

import torch
import torch.utils.data
from PIL import Image
from torchvision import transforms

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import utils
from data import dataset
from model import logit


def preprocess_image(image_path):
    image = Image.open(image_path).convert('L')
    transform_fn = transforms.Compose([
        transforms.Resize(tuple([64, 512])),
        transforms.ToTensor()
    ])
    image_tensor = transform_fn(image).unsqueeze(0)
    return image_tensor


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--nb_cls', type=int, default=80)
    parser.add_argument('--img-size', default=[512, 64], type=int, nargs='+')
    parser.add_argument('--data_path', type=str, default='/home/mpeclab/Documents/HTR_VT/data/iam/lines/')
    parser.add_argument('--pth_path', type=str, default='/home/mpeclab/Documents/HTR-ViTRNN/output/iam_1_layers_2048_dim/best_CER.pth')
    parser.add_argument('--train_data_list', type=str, default='/home/mpeclab/Documents/HTR_VT/data/iam/train.ln')
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--image_path', type=str, default='/home/mpeclab/Documents/HTR_VT/data/iam/lines/a01-000x-01.png')
    parser.add_argument('--num_layer_RNN', type=int, default=1)
    parser.add_argument("--hidden_dim_RNN", type=int, default=2048)

    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    model = logit.create_model(nb_cls=args.nb_cls, img_size=args.img_size[::-1], num_layer_RNN=args.num_layer_RNN, hidden_dim_RNN=args.hidden_dim_RNN)
    ckpt = torch.load(args.pth_path, map_location='cpu')

    model_dict = OrderedDict()
    pattern = re.compile('module.')
    for k, v in ckpt['state_dict_ema'].items():
        if re.search(pattern, k):
            model_dict[re.sub(pattern, '', k)] = v
        else:
            model_dict[k] = v

    model.load_state_dict(model_dict, strict=True)
    model = model.to(device)
    model.eval()

    train_dataset = dataset.myLoadDS(args.train_data_list, args.data_path, args.img_size)
    converter = utils.CTCLabelConverter(train_dataset.ralph.values())

    image_tensor = preprocess_image(args.image_path)
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        preds = model(image_tensor)
        preds = preds.float()
        preds_size = torch.IntTensor([preds.size(1)])
        preds = preds.permute(1, 0, 2).log_softmax(2)
        _, preds_index = preds.max(2)
        preds_index = preds_index.transpose(1, 0).contiguous().view(-1)
        preds_str = converter.decode(preds_index.data, preds_size.data)
        recognized_text = preds_str[0]

    print(f"Recognized_text: {recognized_text}")


if __name__ == '__main__':
    main()
