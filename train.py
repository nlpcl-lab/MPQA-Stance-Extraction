import os
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
from pytorch_pretrained_bert import BertAdam
from torch.optim.adamw import AdamW
import transformers
from model import Net

from data_load import ACE2005Dataset, pad, all_triggers, all_entities, all_postags, all_arguments, tokenizer
from utils import report_to_telegram, set_random_seed
from eval import eval


def train(model, iterator, optimizer, criterion, scheduler):
    model.train()
    for i, batch in enumerate(iterator):
        tokens_x_2d, entities_x_3d, postags_x_2d, triggers_y_2d, arguments_2d, seqlens_1d, head_indexes_2d, words_2d, triggers_2d = batch
        optimizer.zero_grad()
        trigger_logits, triggers_y_2d, trigger_hat_2d, argument_hidden, argument_keys = model.module.predict_triggers(tokens_x_2d=tokens_x_2d, entities_x_3d=entities_x_3d,
                                                                                                                      postags_x_2d=postags_x_2d, head_indexes_2d=head_indexes_2d,
                                                                                                                      triggers_y_2d=triggers_y_2d, arguments_2d=arguments_2d)

        trigger_logits = trigger_logits.view(-1, trigger_logits.shape[-1])
        trigger_loss = criterion(trigger_logits, triggers_y_2d.view(-1))

        # if len(argument_keys) > 0:
        #     argument_logits, arguments_y_1d, argument_hat_1d, argument_hat_2d = model.module.predict_arguments(argument_hidden, argument_keys, arguments_2d)
        #     argument_loss = criterion(argument_logits, arguments_y_1d)
        #     loss = trigger_loss + 2 * argument_loss
        #     if i == 0:
        #         print("=====sanity check for arguments======")
        #         print('arguments_y_1d:', arguments_y_1d)
        #         print("arguments_2d[0]:", arguments_2d[0]['events'])
        #         print("argument_hat_2d[0]:", argument_hat_2d[0]['events'])
        #         print("=======================")
        # else:
        #     loss = trigger_loss

        loss = trigger_loss

        nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        if i == 0:
            print("=====sanity check======")
            print("tokens_x_2d[0]:", tokenizer.convert_ids_to_tokens(
                tokens_x_2d[0])[:seqlens_1d[0]])
            print("entities_x_3d[0]:", entities_x_3d[0][:seqlens_1d[0]])
            print("postags_x_2d[0]:", postags_x_2d[0][:seqlens_1d[0]])
            print("head_indexes_2d[0]:", head_indexes_2d[0][:seqlens_1d[0]])
            print("triggers_2d[0]:", triggers_2d[0])
            print("triggers_y_2d[0]:", triggers_y_2d.cpu(
            ).numpy().tolist()[0][:seqlens_1d[0]])
            print('trigger_hat_2d[0]:', trigger_hat_2d.cpu(
            ).numpy().tolist()[0][:seqlens_1d[0]])
            print("seqlens_1d[0]:", seqlens_1d[0])
            print("arguments_2d[0]:", arguments_2d[0])
            print("=======================")

        if i % 10 == 0:  # monitoring
            print("step: {}, loss: {}".format(i, loss.item()))


if __name__ == "__main__":
    set_random_seed(42)
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=48)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--n_epochs", type=int, default=100)  # 50
    parser.add_argument("--logdir", type=str, default="single")
    parser.add_argument("--trainset", type=str, default="data/train.json")
    parser.add_argument("--devset", type=str, default="data/dev.json")
    parser.add_argument("--testset", type=str, default="data/test.json")

    parser.add_argument("--telegram_bot_token", type=str, default="")
    parser.add_argument("--telegram_chat_id", type=str, default="")

    hp = parser.parse_args()
    # torch.cuda.set_device(1)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = Net(
        device=device,
        trigger_size=len(all_triggers),
        entity_size=len(all_entities),
        all_postags=len(all_postags),
        argument_size=len(all_arguments)
    )
    if device == 'cuda':
        model = model.cuda()

    model = nn.DataParallel(model)

    train_dataset = ACE2005Dataset(hp.trainset)
    dev_dataset = ACE2005Dataset(hp.devset)
    test_dataset = ACE2005Dataset(hp.testset)

    samples_weight = train_dataset.get_samples_weight()
    # sampler = torch.utils.data.WeightedRandomSampler(
    #     samples_weight, len(samples_weight))
    sampler = data.RandomSampler
    train_iter = data.DataLoader(dataset=train_dataset,
                                 batch_size=hp.batch_size,
                                 sampler=data.RandomSampler(
                                     train_dataset),  # =sampler,
                                 num_workers=4,
                                 collate_fn=pad)
    dev_iter = data.DataLoader(dataset=dev_dataset,
                               batch_size=hp.batch_size,
                               shuffle=False,
                               num_workers=4,
                               collate_fn=pad)
    test_iter = data.DataLoader(dataset=test_dataset,
                                batch_size=hp.batch_size,
                                shuffle=False,
                                num_workers=4,
                                collate_fn=pad)

    # optimizer = BertAdam(model.parameters(), lr=hp.lr)
    optimizer = AdamW(
        model.parameters(),
        lr=hp.lr,
        weight_decay=0.01
    )
    total_train_step = hp.n_epochs * len(train_iter)
    scheduler = transformers.get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_train_step,
    )

    criterion = nn.CrossEntropyLoss(ignore_index=0)

    savedir = "1016eval"
    os.makedirs(savedir, exist_ok=True)

    for epoch in range(1, hp.n_epochs + 1):
        train(model, train_iter, optimizer, criterion, scheduler)

        fname = os.path.join(savedir, '{:02d}'.format(epoch))
        print("=========eval dev at epoch={}=========".format(epoch))
        metric_dev, p, r, f1 = eval(model, dev_iter, fname + '_dev')

        torch.save(
            model, savedir + "/model_{:02d}_{:.3f}_{:.3f}_{:.3f}.pt".format(epoch, p, r, f1))
