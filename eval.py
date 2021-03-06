import os
import argparse

import torch
import torch.nn as nn
from torch.utils import data

from model import Net
from sklearn import metrics
import numpy as np

from data_load import MPQADataset, pad, all_attitudes, all_entities, idx2attitude, all_arguments
from utils import calc_metric_strict, calc_metric_soft, calc_metric_loose, find_attitudes


def eval(model, iterator, fname, mode):
    import time
    start = time.time()
    model.eval()

    words_all, attitudes_all, attitudes_hat_all = [], [], []
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            tokens_x_2d, entities_x_3d, attitudes_y_2d, _, seqlens_1d, head_indexes_2d, words_2d, attitudes_2d = batch

            attitude_logits, attitudes_y_2d, attitude_hat_2d, argument_hidden, argument_keys = model.module.predict_attitudes(tokens_x_2d=tokens_x_2d, entities_x_3d=entities_x_3d,
                                                                                                                          head_indexes_2d=head_indexes_2d,
                                                                                                                          attitudes_y_2d=attitudes_y_2d, arguments_2d=_, mode =mode)

            words_all.extend(words_2d)

            attitudes_all.extend(attitudes_2d)
            attitudes_hat_all.extend(attitude_hat_2d.cpu().numpy().tolist())


    attitudes_true, attitudes_pred = [], []
    attitudes_true_report, attitudes_pred_report = [], []

    with open('temp', 'w') as fout:
        for i, (words, attitudes, attitudes_hat) in enumerate(zip(words_all, attitudes_all, attitudes_hat_all)):
            attitudes_hat = attitudes_hat[:len(words) - 1]
            attitudes_hat = [idx2attitude[hat] for hat in attitudes_hat]

            assert len(attitudes) == len(attitudes_hat), "len(attitudes)={}, len(attitudes_hat)={}".format(
                len(attitudes), len(attitudes_hat))

            # [(ith sentence, t_start, t_end, t_type_str)]
            attitudes_true.extend([(i, *item)
                                  for item in find_attitudes(attitudes)])
            attitudes_pred.extend([(i, *item)
                                  for item in find_attitudes(attitudes_hat)])

            # [(ith sentence, t_start, t_end, t_type_str, a_start, a_end, a_type_idx)]


            for w, t, t_h in zip(words[1:], attitudes, attitudes_hat):
                fout.write('{}\t{}\t{}\n'.format(w, t, t_h))

            fout.write("\n")

            attitudes_true_report.extend(attitudes)
            attitudes_pred_report.extend(attitudes_hat)

    # print(metrics.classification_report([attitude for attitude in attitudes_true_report], [attitude for attitude in attitudes_pred_report]))

    print('[Evaluation] test classification')
    attitude_p, attitude_r, attitude_f1 = calc_metric_strict(
        attitudes_true, attitudes_pred)
    print('Using Strict grading')
    print(
        'Precision={:.3f}\nRecall={:.3f}\nF1-score={:.3f}'.format(attitude_p, attitude_r, attitude_f1))
    attitude_p, attitude_r, attitude_f1 = calc_metric_strict(
        attitudes_true, attitudes_pred)
    print('Using Soft grading')
    attitude_p_so, attitude_r_so, attitude_f1_so = calc_metric_soft(
        attitudes_true, attitudes_pred)
    print(
        'Precision={:.3f}\nRecall={:.3f}\nF1-score={:.3f}'.format(attitude_p_so, attitude_r_so, attitude_f1_so))
    attitude_p_l, attitude_r_l, attitude_f1_l = calc_metric_loose(
        attitudes_true, attitudes_pred)

    print(
        'Precision={:.3f}\nRecall={:.3f}\nF1-score={:.3f}'.format(attitude_p_l, attitude_r_l, attitude_f1_l))
    print('Total processing time:{:.3f}sec'.format(time.time() - start))
    # print('[argument classification]')

    # print('P={:.3f}\tR={:.3f}\tF1={:.3f}'.format(argument_p, argument_r, argument_f1))
    print()
    print('[Evaluation] test identification')
    attitudes_true = [(item[0], item[1], item[2]) for item in attitudes_true]
    attitudes_pred = [(item[0], item[1], item[2]) for item in attitudes_pred]

    attitude_p_, attitude_r_, attitude_f1_ = calc_metric_strict(
        attitudes_true, attitudes_pred)
    print('Precision={:.3f}\nRecall={:.3f}\nF1-score={:.3f}'.format(
        attitude_p_, attitude_r_, attitude_f1_))
    print('Total processing time:{:.3f}sec'.format(time.time() - start))

    metric = '[attitude classification]\tP={:.3f}\nR={:.3f}\tF1={:.3f}\n'.format(
        attitude_p, attitude_r, attitude_f1)

    # metric += '[argument identification]\tP={:.3f}\tR={:.3f}\tF1={:.3f}\n'.format(argument_p_, argument_r_, argument_f1_)
    final = fname + \
        mode + ".Strict%.3f_Soft%.3f_Loose%.3f" % (attitude_f1,attitude_f1_so,attitude_f1_l)
    with open(final, 'w') as fout:
        result = open("temp", "r").read()
        fout.write("{}\n".format(result))
        # fout.write(metric)
    os.remove("temp")

    return metric, [attitude_p, attitude_r, attitude_f1], [attitude_p_so, attitude_r_so, attitude_f1_so], [attitude_p_l, attitude_r_l, attitude_f1_l]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", type=str, default="eval_log_dir")
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--testset", type=str, default="mpqa_parsed/test.json")
    parser.add_argument("--model_path", type=str,
                        default="eval_test.P0.80_R0.73_F0.76")
    parser.add_argument("--model", type=str, default="BERT_twice-large")

    hp = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # torch.cuda.set_device(1)

    if hp.model[:4] == 'BERT':
        test_dataset = MPQADataset(hp.testset, hp.model.split('-')[1])
    else:
        test_dataset = MPQADataset(hp.testset)

    test_iter = data.DataLoader(dataset=test_dataset,
                                batch_size=hp.batch_size,
                                shuffle=False,
                                num_workers=4,
                                collate_fn=pad)

    if not os.path.exists(hp.model_path):
        print('Warning: There is no model on the path:',
              hp.model_path, 'Please check the model_path parameter')

    model = torch.load(hp.model_path, map_location='cuda:0')

    if device == 'cuda':
        model = model.cuda()

    if not os.path.exists(hp.logdir):
        os.makedirs(hp.logdir)

    print("=========eval test=========")
    if hp.model[:4] == 'BERT':
        eval(model, test_iter, 'eval_test', hp.model.split('-')[0])
    else:
        eval(model, test_iter, 'eval_test', hp.model)
