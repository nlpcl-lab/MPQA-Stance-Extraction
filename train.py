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
from torch.utils.tensorboard import SummaryWriter

from data_load import MPQADataset, pad, all_attitudes, all_entities, all_arguments
from utils import report_to_telegram, set_random_seed
from eval import eval


def train(model, iterator, optimizer, criterion, scheduler,writer, mode):

    model.train()
    for i, batch in enumerate(iterator):
        tokens_x_2d, entities_x_3d, attitudes_y_2d, arguments_2d, seqlens_1d, head_indexes_2d, words_2d, attitudes_2d = batch
        optimizer.zero_grad()

        attitude_logits, attitudes_y_2d, attitude_hat_2d, argument_hidden, argument_keys = model.module.predict_attitudes(tokens_x_2d=tokens_x_2d, entities_x_3d=entities_x_3d,
                                                                                                                      head_indexes_2d=head_indexes_2d,
                                                                                                                      attitudes_y_2d=attitudes_y_2d, arguments_2d=arguments_2d,  mode =mode)
        """
        print(attitudes_y_2d.shape)
        print(len(words_2d), len(words_2d[0]))
        print(attitudes_y_2d[1])
        print(words_2d[1])
        input()
        
        print(attitude_logits.shape)
        print(type(attitude_logits))

        print(attitude_logits.view(-1, attitude_logits.shape[-1]).shape)
        print(attitude_logits.view(-1, attitude_logits.shape[-1]))
        print(attitudes_y_2d.shape)
        print(attitudes_y_2d.view(-1).shape)
        """

        attitude_logits2 = attitude_logits.view(-1, attitude_logits.shape[-1])
        attitude_loss = criterion(attitude_logits2, attitudes_y_2d.view(-1))


        # if len(argument_keys) > 0:
        #     argument_logits, arguments_y_1d, argument_hat_1d, argument_hat_2d = model.module.predict_arguments(argument_hidden, argument_keys, arguments_2d)
        #     argument_loss = criterion(argument_logits, arguments_y_1d)
        #     loss = attitude_loss + 2 * argument_loss
        #     if i == 0:
        #         print("=====sanity check for arguments======")
        #         print('arguments_y_1d:', arguments_y_1d)
        #         print("arguments_2d[0]:", arguments_2d[0]['events'])
        #         print("argument_hat_2d[0]:", argument_hat_2d[0]['events'])
        #         print("=======================")
        # else:
        #     loss = attitude_loss

        loss = attitude_loss
        if i %100==0:
            writer.add_scalar("loss", loss, ((epoch-1) * 500 + i)/100)


        nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        if i == 0:
            print("=====sanity check======")

            print("entities_x_3d[0]:", entities_x_3d[0][:seqlens_1d[0]])
            print("head_indexes_2d[0]:", head_indexes_2d[0][:seqlens_1d[0]])
            print("attitudes_2d[0]:", attitudes_2d[0])
            print("attitudes_y_2d[0]:", attitudes_y_2d.cpu(
            ).numpy().tolist()[0][:seqlens_1d[0]])
            print('attitude_hat_2d[0]:', attitude_hat_2d.cpu(
            ).numpy().tolist()[0][:seqlens_1d[0]])
            print("seqlens_1d[0]:", seqlens_1d[0])
            print("arguments_2d[0]:", arguments_2d[0])
            print("=======================")

        if i % 10 == 0:  # monitoring
            print("step: {}, loss: {}".format(i, loss.item()))



if __name__ == "__main__":
    set_random_seed(42)
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--n_epochs", type=int, default=200)  # 50
    parser.add_argument("--logdir", type=str, default="single")
    parser.add_argument("--trainset", type=str, default="mpqa_parsed/train.json")
    parser.add_argument("--devset", type=str, default="mpqa_parsed/dev.json")
    parser.add_argument("--testset", type=str, default="mpqa_parsed/test.json")

    parser.add_argument("--model", type=str, default="BERT-base")

    hp = parser.parse_args()
    # torch.cuda.set_device(1)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    bertsize = 'base'
    if hp.model[:4] == "BERT":
        bertsize = hp.model.split("-")[1]
        hp.model = hp.model.split("-")[0]

    model = Net(
        device=device,
        attitude_size=len(all_attitudes),
        entity_size=len(all_entities),
        argument_size=len(all_arguments),
        bert_size = bertsize
    )
    if device == 'cuda':
        model = model.cuda()

    model = nn.DataParallel(model)

    train_dataset = MPQADataset(hp.trainset)
    dev_dataset = MPQADataset(hp.devset)
    test_dataset = MPQADataset(hp.testset)

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
    mode = hp.model
    savedir = "mpqa_eval_" + mode
    os.makedirs(savedir, exist_ok=True)

    writer = SummaryWriter()
    highest = 0
    for epoch in range(1, hp.n_epochs + 1):
        train(model, train_iter, optimizer, criterion, scheduler,writer, mode)

        fname = os.path.join(savedir, '{:02d}'.format(epoch))
        print("=========eval dev at epoch={}=========".format(epoch))
        metric_dev, strict, soft, loose = eval(model, dev_iter, fname + '_dev', mode)

        writer.add_scalar("strict F1", strict[-1], epoch)
        writer.add_scalar("soft F1", soft[-1], epoch)
        writer.add_scalar("loose F1", loose[-1], epoch)

        if strict[-1]+soft[-1]+loose[-1] > highest:
            highest = strict[-1]+soft[-1]+loose[-1]
            torch.save(
                model, savedir + "/model_{:02d}_{:.3f}_{:.3f}_{:.3f}.pt".format(epoch, strict[-1], soft[-1], loose[-1]))
