import os
import re
import sys
import argparse
from collections import OrderedDict

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import utils
from data import dataset
from model import visual


def preprocess_image(image_path):
    """
    Returns:
        image_tensor: tensor resized for model inference, shape (1, 1, 64, 512)
        image_pil   : original PIL image in grayscale, NOT resized
    """
    image_pil = Image.open(image_path).convert('L')

    transform_fn = transforms.Compose([
        transforms.Resize((64, 512)),
        transforms.ToTensor()
    ])

    image_tensor = transform_fn(image_pil).unsqueeze(0)
    return image_tensor, image_pil


def timestep_to_xcenter(t, T, W):
    return (t + 0.5) * W / T


def non_maximum_suppression_1d(indices, scores, min_distance=3):
    """
    Giữ các peak mạnh nhất sao cho khoảng cách timestep giữa 2 peak >= min_distance.

    indices: np.array of candidate peak positions
    scores : np.array of scores for all time steps
    """
    if len(indices) == 0:
        return np.array([], dtype=np.int64)

    # sort candidates by score descending
    order = sorted(indices.tolist(), key=lambda i: scores[i], reverse=True)

    selected = []
    for idx in order:
        keep = True
        for s in selected:
            if abs(idx - s) < min_distance:
                keep = False
                break
        if keep:
            selected.append(idx)

    selected = np.array(sorted(selected), dtype=np.int64)
    return selected


def find_clear_bright_timesteps(
    logits,
    min_class_id=5,
    score_mode="max_subclass",
    min_score_percentile=90.0,
    local_peak_only=True,
    min_distance=3
):
    """
    Chọn các timestep có activation sáng rõ rệt.

    logits: (T, C)

    score_mode:
        - "max_subclass":
            score(t) = max_{c >= min_class_id} logits[t, c]
            => hợp khi bạn muốn bắt mọi cột có điểm sáng rõ ở class >= 5
        - "winner_only":
            chỉ xét nếu class thắng của timestep >= min_class_id,
            score(t) = max logits[t]
            => chặt hơn

    local_peak_only:
        nếu True, chỉ lấy các local maxima theo thời gian

    min_distance:
        khoảng cách timestep tối thiểu giữa 2 peak sau NMS
    """
    T, C = logits.shape
    pred_class = logits.argmax(axis=1)

    if min_class_id >= C:
        raise ValueError(f"min_class_id={min_class_id} >= number of classes={C}")

    if score_mode == "max_subclass":
        sub_logits = logits[:, min_class_id:]   # (T, C-min_class_id)
        score = sub_logits.max(axis=1)
    elif score_mode == "winner_only":
        score = logits.max(axis=1)
    else:
        raise ValueError(f"Unsupported score_mode: {score_mode}")

    threshold = np.percentile(score, min_score_percentile)

    candidates = []
    for t in range(T):
        if score_mode == "winner_only":
            if pred_class[t] < min_class_id:
                continue
        else:
            pass

        if score[t] < threshold:
            continue

        if local_peak_only:
            left_ok = True if t == 0 else score[t] > score[t - 1]
            right_ok = True if t == T - 1 else score[t] >= score[t + 1]
            if not (left_ok and right_ok):
                continue

        candidates.append(t)

    candidates = np.array(candidates, dtype=np.int64)
    selected = non_maximum_suppression_1d(candidates, score, min_distance=min_distance)

    return selected, score, pred_class, threshold


def save_visualization(
    image_pil,
    logit_before,
    logit_after,
    save_path,
    recognized_text=None,
    min_class_id=5,
    score_mode="max_subclass",
    min_score_percentile=90.0,
    local_peak_only=True,
    min_distance=3
):
    """
    Vẽ:
    - ảnh gốc
    - heatmap before RNN
    - heatmap after RNN
    - vạch đứng chỉ tại các timestep sáng rõ rệt
    """

    img = np.array(image_pil)   # (H_orig, W_orig)
    H_orig, W_orig = img.shape

    T = logit_after.shape[0]
    C = logit_after.shape[1]

    peak_before_idx, score_before, pred_before, thr_before = find_clear_bright_timesteps(
        logit_before,
        min_class_id=min_class_id,
        score_mode=score_mode,
        min_score_percentile=min_score_percentile,
        local_peak_only=local_peak_only,
        min_distance=min_distance
    )

    peak_after_idx, score_after, pred_after, thr_after = find_clear_bright_timesteps(
        logit_after,
        min_class_id=min_class_id,
        score_mode=score_mode,
        min_score_percentile=min_score_percentile,
        local_peak_only=local_peak_only,
        min_distance=min_distance
    )

    x_before = [timestep_to_xcenter(t, T, W_orig) for t in peak_before_idx]
    x_after = [timestep_to_xcenter(t, T, W_orig) for t in peak_after_idx]

    fig = plt.figure(figsize=(18, 10))
    gs = GridSpec(
        3, 3,
        figure=fig,
        width_ratios=[40, 1.2, 8],
        height_ratios=[1.0, 1.0, 1.0],
        wspace=0.08,
        hspace=0.32
    )

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])   # before
    ax3 = fig.add_subplot(gs[2, 0])   # after

    cax2 = fig.add_subplot(gs[1, 1])
    cax3 = fig.add_subplot(gs[2, 1])

    tax = fig.add_subplot(gs[:, 2])
    tax.axis("off")

    tick_count = min(T, 10)
    xs = np.linspace(0, W_orig, tick_count)

    # -------------------------------------------------
    # 1) Original image
    # -------------------------------------------------
    ax1.imshow(
        img,
        cmap='gray',
        aspect='auto',
        extent=[0, W_orig, H_orig, 0]
    )

    title = "Original input image"
    if recognized_text is not None:
        title += f" | Recognized: {recognized_text}"
    ax1.set_title(title)

    ax1.set_ylabel("Image")
    ax1.set_yticks([])
    ax1.set_xticks(xs.astype(int))

    for x in xs:
        ax1.axvline(x, color='yellow', alpha=0.08, linewidth=0.8)

    for t, x in zip(peak_before_idx, x_before):
        ax1.axvline(x, color='deepskyblue', alpha=0.80, linewidth=1.1, linestyle='--')

    for t, x in zip(peak_after_idx, x_after):
        ax1.axvline(x, color='red', alpha=0.85, linewidth=1.2, linestyle='-')

    tax.plot([], [], color='deepskyblue', linestyle='--', linewidth=1.3,
             label=f'Before RNN clear peaks')
    tax.plot([], [], color='red', linestyle='-', linewidth=1.3,
             label=f'After RNN clear peaks')
    tax.legend(loc='upper right', fontsize=9)
    im2 = ax2.imshow(
        logit_before.T,
        aspect='auto',
        extent=[0, W_orig, C - 1, 0]
    )
    ax2.set_title(
        f"RAW logits before RNN"
    )
    ax2.set_ylabel("Class id")
    ax2.set_xticks(xs.astype(int))

    for x in xs:
        ax2.axvline(x, color='white', alpha=0.06, linewidth=0.8)

    for t, x in zip(peak_before_idx, x_before):
        ax2.axvline(x, color='deepskyblue', alpha=0.90, linewidth=1.0, linestyle='--')

    fig.colorbar(im2, cax=cax2)

    im3 = ax3.imshow(
        logit_after.T,
        aspect='auto',
        extent=[0, W_orig, C - 1, 0]
    )
    ax3.set_title(
        f"RAW logits after RNN"
    )
    ax3.set_ylabel("Class id")
    ax3.set_xticks(xs.astype(int))

    for x in xs:
        ax3.axvline(x, color='white', alpha=0.06, linewidth=0.8)

    for t, x in zip(peak_after_idx, x_after):
        ax3.axvline(x, color='red', alpha=0.92, linewidth=1.0, linestyle='-')

    fig.colorbar(im3, cax=cax3)

    summary = []
  
    tax.text(
        0.0, 1.0,
        "\n".join(summary),
        ha="left",
        va="top",
        fontsize=10
    )

    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)

    print(f"[INFO] Selection mode = {score_mode}")
    print(f"[INFO] Min class id = {min_class_id}")
    print(f"[INFO] Percentile threshold = {min_score_percentile}")
    print(f"[INFO] Local peak only = {local_peak_only}")
    print(f"[INFO] Min timestep distance = {min_distance}")
    print(f"[INFO] BEFORE timesteps: {peak_before_idx.tolist()}")
    print(f"[INFO] AFTER  timesteps: {peak_after_idx.tolist()}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--nb_cls', type=int, default=80)
    parser.add_argument('--img-size', default=[512, 64], type=int, nargs='+')
    parser.add_argument('--data_path', type=str, default='/home/mpeclab/Documents/HTR_VT/data/iam/lines/')
    parser.add_argument('--pth_path', type=str, default='/home/mpeclab/Documents/HTR-ViTRNN/output/iam_1_layers_2048_dim/best_CER.pth')
    parser.add_argument('--num_layer_RNN', type=int, default=1)
    parser.add_argument("--hidden_dim_RNN", type=int, default=2048)
    
    parser.add_argument('--train_data_list', type=str, default='/home/mpeclab/Documents/HTR_VT/data/iam/train.ln')
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--image_path', type=str, default='/home/mpeclab/Documents/HTR_VT/data/iam/lines/d01-104-01.png')
    parser.add_argument('--save_dir', type=str, default='./visual_debug_single')
    parser.add_argument('--save_file', type=str, default='d01-104-01.png')
    parser.add_argument('--state_key', type=str, default='state_dict_ema',
                        help='checkpoint key to load, e.g. state_dict_ema or state_dict')

    parser.add_argument('--min_class_id', type=int, default=2,
                        help='only consider class ids >= this value')
    parser.add_argument('--score_mode', type=str, default='max_subclass',
                        choices=['max_subclass', 'winner_only'],
                        help='how to compute brightness score per timestep')
    parser.add_argument('--min_score_percentile', type=float, default=70.0,
                        help='keep only very bright timesteps above this percentile')
    parser.add_argument('--local_peak_only', action='store_true',
                        help='keep only local maxima in time')
    parser.add_argument('--min_distance', type=int, default=3,
                        help='minimum timestep distance between selected peaks')

    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    model = visual.create_model(nb_cls=args.nb_cls, img_size=args.img_size[::-1], num_layer_RNN = args.num_layer_RNN, hidden_dim_RNN = args.hidden_dim_RNN)

    ckpt = torch.load(args.pth_path, map_location='cpu')
    if args.state_key not in ckpt:
        raise KeyError(f"Key '{args.state_key}' not found in checkpoint. Available keys: {list(ckpt.keys())}")

    raw_state = ckpt[args.state_key]

    model_dict = OrderedDict()
    pattern = re.compile(r'^module\.')
    for k, v in raw_state.items():
        if re.search(pattern, k):
            model_dict[re.sub(pattern, '', k)] = v
        else:
            model_dict[k] = v

    model.load_state_dict(model_dict, strict=True)
    model = model.to(device)
    model.eval()

    train_dataset = dataset.myLoadDS(args.train_data_list, args.data_path, args.img_size)
    converter = utils.CTCLabelConverter(train_dataset.ralph.values())

    image_tensor, image_pil = preprocess_image(args.image_path)
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        preds, debug_dict = model(image_tensor, return_debug=True)

        preds = preds.float()
        preds_size = torch.IntTensor([preds.size(1)])

        preds_log = preds.permute(1, 0, 2).log_softmax(2)
        _, preds_index = preds_log.max(2)
        preds_index = preds_index.transpose(1, 0).contiguous().view(-1)
        preds_str = converter.decode(preds_index.data, preds_size.data)
        recognized_text = preds_str[0]

    print(f"Recognized_text: {recognized_text}")

    logit_before = debug_dict["logit_before_raw"][0].detach().cpu().numpy()  # (T, C)
    logit_after = debug_dict["logit_after_raw"][0].detach().cpu().numpy()    # (T, C)

    out_path = os.path.join(args.save_dir, args.save_file)
    save_visualization(
        image_pil=image_pil,
        logit_before=logit_before,
        logit_after=logit_after,
        save_path=out_path,
        recognized_text=recognized_text,
        min_class_id=args.min_class_id,
        score_mode=args.score_mode,
        min_score_percentile=args.min_score_percentile,
        local_peak_only=args.local_peak_only,
        min_distance=args.min_distance
    )

    print(f"Saved figure to: {out_path}")


if __name__ == '__main__':
    main()