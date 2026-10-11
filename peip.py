import argparse
import os
import time
import random
import logging
import numpy as np
from tqdm import tqdm
from copy import deepcopy as dcopy

import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast

from replace import clip
from models.prompters import TokenPrompter, NullPrompter
from utils import *
from attacks import *
from func import clip_img_preprocessing, multiGPU_CLIP


device = "cuda" if torch.cuda.is_available() else "cpu"
def compute_prompt_sensitivity(model, images, prompt_variants, prompter):
    all_logits = []
    with torch.no_grad():
        for p_variant in prompt_variants:
            text_tokens = clip.tokenize(p_variant).to(device)
            logits, _, _, _ = multiGPU_CLIP(
                None, None, None, model,
                prompter(clip_img_preprocessing(images)),
                text_tokens=text_tokens, prompt_token=None
            )
            all_logits.append(F.softmax(logits, dim=-1))

    all_probs = torch.stack(all_logits, dim=1)
    sensitivity = all_probs.var(dim=1).max(dim=-1)[0]
    return sensitivity, all_probs.mean(dim=1)


def peip_purification(model, X, prompter, p_canonical, p_ensemble_probs,
                      epsilon=4/255., num_steps=3, lam=1.0, eta=1/255.):
    delta = torch.zeros_like(X).uniform_(-epsilon, epsilon)
    delta.requires_grad = True

    with torch.no_grad():
        text_tokens_can = clip.tokenize(p_canonical).to(device)
        logits_can, _, _, _ = multiGPU_CLIP(
            None, None, None, model,
            prompter(clip_img_preprocessing(X)),
            text_tokens=text_tokens_can
        )
        prob_can = F.softmax(logits_can, dim=-1)
        pseudo_label = torch.argmax(p_ensemble_probs - prob_can, dim=-1)

    for _ in range(num_steps):
        img_adv = prompter(clip_img_preprocessing(X + delta))
        logits_p0, _, _, _ = multiGPU_CLIP(
            None, None, None, model, img_adv,
            text_tokens=text_tokens_can
        )
        prob_p0 = F.softmax(logits_p0, dim=-1)
        loss_var = F.kl_div(prob_p0.log(), p_ensemble_probs, reduction='batchmean')
        loss_sem = F.cross_entropy(logits_p0, pseudo_label)
        loss_total = loss_var + lam * loss_sem
        grad = torch.autograd.grad(loss_total, delta)[0]
        delta.data = torch.clamp(delta.data - eta * grad.sign(), -epsilon, epsilon)
        delta.data = clamp(delta.data, lower_limit - X, upper_limit - X)

    return delta.data


def validate_peip(args, val_dataset_name, model, prompter):

    logging.info("Running PEIP Evaluation")
    for dataset_name in val_dataset_name:

        val_dataset, val_loader = load_val_dataset(args, dataset_name)
        canonical_prompt = get_canonical_prompt(dataset_name)
        prompt_variants = get_prompt_variants(dataset_name, M=args.prompt_num)
        top1_clean = AverageMeter('Clean Acc', ':6.2f')
        top1_rob = AverageMeter('Robust Acc', ':6.2f')
        for images, target in tqdm(val_loader):
            images = images.to(device)
            target = target.to(device)

            with torch.no_grad():
                output, _, _, _ = multiGPU_CLIP(
                    None, None, None, model,
                    prompter(clip_img_preprocessing(images)),
                    text_tokens=clip.tokenize(canonical_prompt).to(device)
                )
                acc = accuracy(output, target, topk=(1,))
                top1_clean.update(acc[0].item(), images.size(0))

            s_init, p_consensus = compute_prompt_sensitivity(
                model, images, prompt_variants, prompter
            )

            need_purify = s_init > args.tau_thres

            adv_images = attack_pgd(
                args, prompter, model,
                None, None, None,
                None,
                images, target,
                args.test_stepsize,
                args.test_numsteps,
                'l_inf',
                text_tokens=clip.tokenize(canonical_prompt).to(device),
                epsilon=args.test_eps,
                dataset_name=dataset_name
            )

            final_delta = torch.zeros_like(images)

            if need_purify.any():
                purified_delta = peip_purification(
                    model,
                    adv_images[need_purify],
                    prompter,
                    canonical_prompt,
                    p_consensus[need_purify],
                    epsilon=args.peip_eps,
                    num_steps=args.peip_steps,
                    lam=args.lam
                )
                final_delta[need_purify] = purified_delta

            with torch.no_grad():
                output, _, _, _ = multiGPU_CLIP(
                    None, None, None, model,
                    prompter(clip_img_preprocessing(adv_images + final_delta)),
                    text_tokens=clip.tokenize(canonical_prompt).to(device)
                )
                acc = accuracy(output, target, topk=(1,))
                top1_rob.update(acc[0].item(), images.size(0))

        logging.info(
            f"{dataset_name}: Clean {top1_clean.avg:.2f} | Robust {top1_rob.avg:.2f}"
        )


def parse_options():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='tinyImageNet')
    parser.add_argument('--root', type=str, default='./data')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=16)

    parser.add_argument('--test_eps', type=float, default=1.)
    parser.add_argument('--test_numsteps', type=int, default=10)
    parser.add_argument('--test_stepsize', type=float, default=1.)

    parser.add_argument('--tau_thres', type=float, default=0.16)
    parser.add_argument('--prompt_num', type=int, default=4)
    parser.add_argument('--peip_eps', type=float, default=4/255.)
    parser.add_argument('--peip_steps', type=int, default=3)
    parser.add_argument('--lam', type=float, default=1.0)

    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--outdir', type=str, default='PEIP_results')

    return parser.parse_args()


def main():

    args = parse_options()
    os.makedirs(args.outdir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    args.test_eps /= 255.
    args.test_stepsize /= 255.

    model, _ = clip.load('ViT-B/32', device, jit=False)
    for p in model.parameters():
        p.requires_grad = False

    convert_models_to_fp32(model)
    model = torch.nn.DataParallel(model).cuda()
    model.eval()
    prompter = NullPrompter()
    prompter = torch.nn.DataParallel(prompter).cuda()
    logging.info("Model loaded.")
    test_set = DATASETS
    validate_peip(args, test_set, model, prompter)


if __name__ == "__main__":
    main()