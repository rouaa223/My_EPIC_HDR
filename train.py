import argparse
import random
import sys
import os
import skimage.color as color
import numpy as np
from matplotlib import pyplot as plt
import torch
import imageio
from torch.utils.data import DataLoader
import torch.optim as optim
from compressai.datasets.image import ImageFolder
from compressai.losses.rate_distortion import RateDistortionLoss
from compressai.optimizers import net_aux_optimizer
from compressai.zoo import image_models

tonemap = lambda x: (np.log(np.clip(x, 0, 1) * 5000 + 1) / np.log(5000 + 1) * 255).astype(np.uint8)


class AverageMeter:
    """Compute running average."""
    def __init__(self):
        self.val = self.avg = self.sum = self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def configure_optimizers(net, args):
    conf = {
        "net": {"type": "Adam", "lr": args.learning_rate},
        "aux": {"type": "Adam", "lr": args.learning_rate},
    }
    optimizer = net_aux_optimizer(net, conf)
    return optimizer["net"]


def train_one_epoch(model, criterion, train_dataloader, optimizer, epoch, clip_max_norm):
    model.train()
    device = next(model.parameters()).device

    for i, d in enumerate(train_dataloader):
        x = d['nlp_I'].to(device)
        hdr_l = d['hdr_l'].to(device)
        s_max = d['s_max'].to(device).to(torch.float32)
        hdr = d['hdr'].to(device)
        reference = d['reference'].to(device)

        optimizer.zero_grad()
        out_net1, out_net2 = model(x, s_max, hdr)
        out_criterion = criterion(out_net1, out_net2, hdr_l, reference, s_max)
        out_criterion["loss"].backward()

        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        optimizer.step()

        if i % 2 == 0:
            print(f"Train epoch {epoch}: [{i}/{len(train_dataloader)}] "
                  f"Loss: {out_criterion['loss'].item():.3f} | "
                  f"LDR loss: {out_criterion['ldr_loss'].item():.6f} | "
                  f"Bpp loss1: {out_criterion['bpp_loss1'].item():.2f} | "
                  f"HDR loss: {out_criterion['hdr_loss'].item():.6f} | "
                  f"Bpp loss2: {out_criterion['bpp_loss2'].item():.2f}")


def test_epoch(epoch, test_dataloader, criterion, path, model):
    model.eval()
    device = next(model.parameters()).device
    loss = AverageMeter()

    with torch.no_grad():
        for i, d in enumerate(test_dataloader):
            x = d['nlp_I'].to(device)
            hdr_l = d['hdr_l'].to(device)
            s_max = d['s_max'].to(device).to(torch.float32)
            hdr = d['hdr'].to(device)
            reference = d['reference'].to(device)
            hdr_name = d['hdr_name']

            out_net1, out_net2 = model(x, s_max, hdr)
            out_criterion = criterion(out_net1, out_net2, hdr_l, reference, s_max)

            # Reconstruction HDR
            hdr_recon = out_net2["hdr"]
            hdr_recon_max = torch.max(torch.max(torch.max(hdr_recon, 1)[0], 1)[0], 1)[0].unsqueeze(1).unsqueeze(1).unsqueeze(1)
            hdr_recon = hdr_recon / (hdr_recon_max + 1e-30)
            hdr_rgb = hdr_recon.squeeze().permute(1, 2, 0).cpu().numpy()
            rgb8_h_tm = tonemap(hdr_rgb / np.max(hdr_rgb))

            # Save images
            os.makedirs(os.path.join(path, 'hdr'), exist_ok=True)
            imageio.imwrite(os.path.join(path, 'hdr', hdr_name[0] + 'h_tm.png'), rgb8_h_tm)
            imageio.imwrite(os.path.join(path, 'hdr', hdr_name[0] + '_hdr.hdr'), hdr_rgb)

            loss.update(out_criterion["loss"])

    print(f"Test epoch {epoch}: Average loss: {loss.avg:.3f}")
    return loss.avg


def save_checkpoint(state, is_best, filename="checkpoint.pth.tar"):
    torch.save(state, filename)
    if is_best:
        torch.save(state["state_dict"], "checkpoint_best_loss.pth.tar")


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Training script for multi-quality HDR compression")
    parser.add_argument("-d", "--dataset", type=str, default='./train')
    parser.add_argument("--test_dataset", type=str, default='./test')
    parser.add_argument("--results_savepath", type=str, default='./output')
    parser.add_argument("-e", "--epochs", type=int, default=600)
    parser.add_argument("-lr", "--learning-rate", type=float, default=1e-4)
    parser.add_argument("-n", "--num_workers", type=int, default=8)
    parser.add_argument("--lambda", dest="lmbda", type=float, default=200)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--test-batch-size", type=int, default=1)
    parser.add_argument("--patch-size", type=int, nargs=2, default=(512, 512))
    parser.add_argument("--cuda", action='store_true')
    parser.add_argument("--save", action='store_true')
    parser.add_argument("--seed", type=int)
    parser.add_argument("--clip_max_norm", type=float, default=1.0)
    parser.add_argument("--checkpoint", type=str, default='')
    parser.add_argument("--qualities", type=str, default='1,2,3,4,5,6,7,8', help="Comma-separated list of qualities")
    parser.add_argument("--init-from-prev", action='store_true', help="Init each quality from previous checkpoint")
    args = parser.parse_args(argv)
    return args


def main(argv):
    args = parse_args(argv)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)

    train_dataset = ImageFolder(
        os.path.join(args.dataset, 'train.txt'),
        os.path.join(args.dataset, 'train'),
        train=True,
        test=False
    )
    test_dataset = ImageFolder(
        os.path.join(args.test_dataset, 'test.txt'),
        os.path.join(args.test_dataset, 'test'),
        train=False,
        test=True
    )

    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=(device == "cuda"),
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=(device == "cuda"),
    )

    qualities = [int(q) for q in args.qualities.split(',')]
    prev_checkpoint = None

    for q in qualities:
        print(f"\n🔹 Training quality={q}")

        net1 = image_models["mbt2018"](quality=q).to(device)
        net2 = image_models["ldr2hdr"](quality=q).to(device)
        net = image_models["end2end"](net1, net2).to(device)

        optimizer = configure_optimizers(net, args)
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[200, 400], gamma=0.1)
        criterion = RateDistortionLoss(lmbda=args.lmbda)

        if args.init_from_prev and prev_checkpoint is not None:
            net.load_state_dict(prev_checkpoint, strict=False)

        last_epoch = 0
        best_loss = float("inf")

        for epoch in range(last_epoch, args.epochs):
            train_one_epoch(net, criterion, train_dataloader, optimizer, epoch, args.clip_max_norm)
            lr_scheduler.step()

            if (epoch+1) % 50 == 0 or epoch == 0:
                os.makedirs(args.results_savepath, exist_ok=True)
                loss = test_epoch(epoch, test_dataloader, criterion, args.results_savepath, net)

                is_best = loss < best_loss
                best_loss = min(loss, best_loss)

                save_checkpoint(
                    {
                        "epoch": epoch,
                        "state_dict": net.state_dict(),
                        "loss": loss,
                        "optimizer": optimizer.state_dict(),
                        "lr_scheduler": lr_scheduler.state_dict(),
                    },
                    is_best,
                    filename=f"checkpoint_quality{q}.pth.tar"
                )

        prev_checkpoint = net.state_dict()


if __name__ == "__main__":
    main(sys.argv[1:])
