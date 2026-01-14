import os
import re
import json
import torch
import torch.utils.data
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt

from torch.utils.tensorboard import SummaryWriter
from functools import partial

import valid
from utils import utils
from utils import sam
from utils import option
from data import dataset
from model import HTR_VT

def parse_log(log_file):
    iters, train_loss = [], []
    val_iters, val_loss, cer, wer = [], [], [], []

    with open(log_file, "r", encoding="utf-8") as f:
        for line in f:
            m = re.search(r"Iter\s*:\s*(\d+).*training loss\s*:\s*([\d\.]+)", line)
            if m:
                iters.append(int(m.group(1)))
                train_loss.append(float(m.group(2)))

            m = re.search(
                r"Val\. loss\s*:\s*([\d\.]+)\s*CER\s*:\s*([\d\.]+)\s*WER\s*:\s*([\d\.]+)",
                line
            )
            if m and iters:
                val_iters.append(iters[-1])
                val_loss.append(float(m.group(1)))
                cer.append(float(m.group(2)))
                wer.append(float(m.group(3)))

    return iters, train_loss, val_iters, val_loss, cer, wer


def visualize_log(log_file, output_path, title):
    iters, train_loss, val_iters, val_loss, cer, wer = parse_log(log_file)

    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    lw = 2

    axs[0, 0].plot(iters, train_loss, linewidth=lw)
    axs[0, 0].set_title("Training Loss")

    axs[0, 1].plot(val_iters, val_loss, marker="o", linewidth=lw)
    axs[0, 1].set_title("Validation Loss")

    axs[1, 0].plot(val_iters, cer, marker="o", linewidth=lw)
    axs[1, 0].set_title("CER")

    axs[1, 1].plot(val_iters, wer, marker="o", linewidth=lw)
    axs[1, 1].set_title("WER")

    for ax in axs.flat:
        ax.set_xlabel("Iteration")
        ax.grid(True)

    plt.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=300)
    plt.close()



def compute_loss(args, model, image, batch_size, criterion, text, length):
    preds = model(image, args.mask_ratio, args.max_span_length, use_masking=True)
    preds = preds.float()
    preds_size = torch.IntTensor([preds.size(1)] * batch_size).cuda()
    preds = preds.permute(1, 0, 2).log_softmax(2)

    torch.backends.cudnn.enabled = False
    loss = criterion(preds, text.cuda(), preds_size, length.cuda()).mean()
    torch.backends.cudnn.enabled = True
    return loss


def train(batch_size):

    args = option.get_args_parser()
    torch.manual_seed(args.seed)


    args.train_bs = batch_size
    args.exp_name = f'{args.exp_name}_bs{args.train_bs}'

    args.save_dir = os.path.join(args.out_dir, args.exp_name)
    os.makedirs(args.save_dir, exist_ok=True)

    logger = utils.get_logger(args.save_dir)
    logger.info(json.dumps(vars(args), indent=4, sort_keys=True))
    writer = SummaryWriter(args.save_dir)

    model = HTR_VT.create_model(nb_cls=args.nb_cls, img_size=args.img_size[::-1],
                                num_layer_RNN=args.num_layers_RNN,
                                hidden_dim_RNN=args.hidden_dim_RNN)

    total_param = sum(p.numel() for p in model.parameters())
    logger.info('total_param is {}'.format(total_param))

    model.train()
    model = model.cuda()
    model_ema = utils.ModelEma(model, args.ema_decay)
    model.zero_grad()

    logger.info('Loading train loader...')
    train_dataset = dataset.myLoadDS(args.train_data_list, args.data_path, args.img_size)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.train_bs,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=args.num_workers,
                                               collate_fn=partial(dataset.SameTrCollate, args=args))
    train_iter = dataset.cycle_data(train_loader)

    logger.info('Loading val loader...')
    val_dataset = dataset.myLoadDS(args.val_data_list, args.data_path, args.img_size, ralph=train_dataset.ralph)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=args.val_bs,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=args.num_workers)

    optimizer = sam.SAM(model.parameters(), torch.optim.AdamW, lr=1e-7, betas=(0.9, 0.99), weight_decay=args.weight_decay)
    criterion = torch.nn.CTCLoss(reduction='none', zero_infinity=True)
    converter = utils.CTCLabelConverter(train_dataset.ralph.values())

    best_cer, best_wer = 1e+6, 1e+6
    train_loss = 0.0

    #### ---- train & eval ---- ####

    for nb_iter in range(1, args.total_iter):

        optimizer, current_lr = utils.update_lr_cos(nb_iter, args.warm_up_iter, args.total_iter, args.max_lr, optimizer)

        optimizer.zero_grad()
        batch = next(train_iter)
        image = batch[0].cuda()
        text, length = converter.encode(batch[1])
        batch_size = image.size(0)
        loss = compute_loss(args, model, image, batch_size, criterion, text, length)
        loss.backward()
        optimizer.first_step(zero_grad=True)
        compute_loss(args, model, image, batch_size, criterion, text, length).backward()
        optimizer.second_step(zero_grad=True)
        model.zero_grad()
        model_ema.update(model, num_updates=nb_iter / 2)
        train_loss += loss.item()

        if nb_iter % args.print_iter == 0:
            train_loss_avg = train_loss / args.print_iter

            logger.info(f'Iter : {nb_iter} \t LR : {current_lr:0.5f} \t training loss : {train_loss_avg:0.5f} \t ' )

            writer.add_scalar('./Train/lr', current_lr, nb_iter)
            writer.add_scalar('./Train/train_loss', train_loss_avg, nb_iter)
            train_loss = 0.0

        if nb_iter % args.eval_iter == 0:
            model.eval()
            with torch.no_grad():
                val_loss, val_cer, val_wer, preds, labels = valid.validation(model_ema.ema,
                                                                             criterion,
                                                                             val_loader,
                                                                             converter)

                if val_cer < best_cer:
                    logger.info(f'CER improved from {best_cer:.4f} to {val_cer:.4f}!!!')
                    best_cer = val_cer
                    checkpoint = {
                        'model': model.state_dict(),
                        'state_dict_ema': model_ema.ema.state_dict(),
                        'optimizer': optimizer.state_dict(),
                    }
                    torch.save(checkpoint, os.path.join(args.save_dir, 'best_CER.pth'))

                if val_wer < best_wer:
                    logger.info(f'WER improved from {best_wer:.4f} to {val_wer:.4f}!!!')
                    best_wer = val_wer
                    checkpoint = {
                        'model': model.state_dict(),
                        'state_dict_ema': model_ema.ema.state_dict(),
                        'optimizer': optimizer.state_dict(),
                    }
                    torch.save(checkpoint, os.path.join(args.save_dir, 'best_WER.pth'))

                logger.info(
                    f'Val. loss : {val_loss:0.3f} \t CER : {val_cer:0.4f} \t WER : {val_wer:0.4f} \t ')

                writer.add_scalar('./VAL/CER', val_cer, nb_iter)
                writer.add_scalar('./VAL/WER', val_wer, nb_iter)
                writer.add_scalar('./VAL/bestCER', best_cer, nb_iter)
                writer.add_scalar('./VAL/bestWER', best_wer, nb_iter)
                writer.add_scalar('./VAL/val_loss', val_loss, nb_iter)
                model.train()

    log_file = os.path.join('./output', args.exp_name, 'run.log')
    output_path = os.path.join('./output', args.exp_name, f'log_visualization_bs{batch_size}.png')
    visualize_log(log_file, output_path, title=f'Training Log Visualization (Batch Size: {batch_size})')

def main():
    batch_sizes = [2, 4, 6, 8, 16, 32]

    for bs in batch_sizes:
        print(f'Training with batch size: {bs}')
        train(bs)

if __name__ == '__main__':
    main()