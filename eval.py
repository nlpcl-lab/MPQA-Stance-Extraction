import os
import argparse

import torch
import torch.nn as nn
from torch.utils import data

from model import Net
from sklearn import metrics
import numpy as np

from data_load import ACE2005Dataset, pad, all_triggers, all_entities, idx2trigger, all_arguments
from utils import calc_metric, find_triggers


def eval(model, iterator, fname):
    import time
    start = time.time()
    model.eval()

    words_all, triggers_all, triggers_hat_all, arguments_all, arguments_hat_all = [], [], [], [], []
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            tokens_x_2d, entities_x_3d, triggers_y_2d, arguments_2d, seqlens_1d, head_indexes_2d, words_2d, triggers_2d = batch

            trigger_logits, triggers_y_2d, trigger_hat_2d, argument_hidden, argument_keys = model.module.predict_triggers(tokens_x_2d=tokens_x_2d, entities_x_3d=entities_x_3d,
                                                                                                                          head_indexes_2d=head_indexes_2d,
                                                                                                                          triggers_y_2d=triggers_y_2d, arguments_2d=arguments_2d)

            words_all.extend(words_2d)

            triggers_all.extend(triggers_2d)
            triggers_hat_all.extend(trigger_hat_2d.cpu().numpy().tolist())
            arguments_all.extend(arguments_2d)

            if len(argument_keys) > 0:
                argument_logits, arguments_y_1d, argument_hat_1d, argument_hat_2d = model.module.predict_arguments(
                    argument_hidden, argument_keys, arguments_2d)
                arguments_hat_all.extend(argument_hat_2d)
            else:
                batch_size = len(arguments_2d)
                argument_hat_2d = [{'events': {}} for _ in range(batch_size)]
                arguments_hat_all.extend(argument_hat_2d)

    triggers_true, triggers_pred, arguments_true, arguments_pred = [], [], [], []
    triggers_true_report, triggers_pred_report = [], []

    with open('temp', 'w') as fout:
        for i, (words, triggers, triggers_hat, arguments, arguments_hat) in enumerate(zip(words_all, triggers_all, triggers_hat_all, arguments_all, arguments_hat_all)):
            triggers_hat = triggers_hat[:len(words) - 1]
            triggers_hat = [idx2trigger[hat] for hat in triggers_hat]

            assert len(triggers) == len(triggers_hat), "len(triggers)={}, len(triggers_hat)={}".format(
                len(triggers), len(triggers_hat))

            # [(ith sentence, t_start, t_end, t_type_str)]
            triggers_true.extend([(i, *item)
                                  for item in find_triggers(triggers)])
            triggers_pred.extend([(i, *item)
                                  for item in find_triggers(triggers_hat)])

            # [(ith sentence, t_start, t_end, t_type_str, a_start, a_end, a_type_idx)]
            for trigger in arguments['events']:
                t_start, t_end, t_type_str = trigger
                for argument in arguments['events'][trigger]:
                    a_start, a_end, a_type_idx = argument
                    arguments_true.append(
                        (i, t_start, t_end, t_type_str, a_start, a_end, a_type_idx))

            for trigger in arguments_hat['events']:
                t_start, t_end, t_type_str = trigger
                for argument in arguments_hat['events'][trigger]:
                    a_start, a_end, a_type_idx = argument
                    arguments_pred.append(
                        (i, t_start, t_end, t_type_str, a_start, a_end, a_type_idx))

            for w, t, t_h in zip(words[1:], triggers, triggers_hat):
                fout.write('{}\t{}\t{}\n'.format(w, t, t_h))
            # fout.write('#arguments#{}\n'.format(arguments['events']))
            # fout.write('#arguments_hat#{}\n'.format(arguments_hat['events']))
            fout.write("\n")

            triggers_true_report.extend(triggers)
            triggers_pred_report.extend(triggers_hat)

    # print(metrics.classification_report([trigger for trigger in triggers_true_report], [trigger for trigger in triggers_pred_report]))

    print('[Evaluation] test classification')
    trigger_p, trigger_r, trigger_f1 = calc_metric(
        triggers_true, triggers_pred)
    print(
        'Precision={:.3f}\nRecall={:.3f}\nF1-score={:.3f}'.format(trigger_p, trigger_r, trigger_f1))
    print('Total processing time:{:.3f}sec'.format(time.time() - start))
    # print('[argument classification]')
    # argument_p, argument_r, argument_f1 = calc_metric(arguments_true, arguments_pred)
    # print('P={:.3f}\tR={:.3f}\tF1={:.3f}'.format(argument_p, argument_r, argument_f1))
    print()
    print('[Evaluation] test identification')
    triggers_true = [(item[0], item[1], item[2]) for item in triggers_true]
    triggers_pred = [(item[0], item[1], item[2]) for item in triggers_pred]

    trigger_p_, trigger_r_, trigger_f1_ = calc_metric(
        triggers_true, triggers_pred)
    print('Precision={:.3f}\nRecall={:.3f}\nF1-score={:.3f}'.format(
        trigger_p_, trigger_r_, trigger_f1_))
    print('Total processing time:{:.3f}sec'.format(time.time() - start))

    # print('[argument identification]')
    # arguments_true = [(item[0], item[1], item[2], item[3], item[4], item[5]) for item in arguments_true]
    # arguments_pred = [(item[0], item[1], item[2], item[3], item[4], item[5]) for item in arguments_pred]
    # argument_p_, argument_r_, argument_f1_ = calc_metric(arguments_true, arguments_pred)
    # print('P={:.3f}\tR={:.3f}\tF1={:.3f}'.format(argument_p_, argument_r_, argument_f1_))

    metric = '[trigger classification]\tP={:.3f}\nR={:.3f}\tF1={:.3f}\n'.format(
        trigger_p, trigger_r, trigger_f1)
    # metric += '[argument classification]\tP={:.3f}\tR={:.3f}\tF1={:.3f}\n'.format(argument_p, argument_r, argument_f1)
    metric += '[trigger identification]\tP={:.3f}\tR={:.3f}\tF1={:.3f}\n'.format(
        trigger_p_, trigger_r_, trigger_f1_)

    # metric += '[argument identification]\tP={:.3f}\tR={:.3f}\tF1={:.3f}\n'.format(argument_p_, argument_r_, argument_f1_)
    final = fname + \
        ".P%.2f_R%.2f_F%.2f" % (trigger_p_, trigger_r_, trigger_f1_)
    with open(final, 'w') as fout:
        result = open("temp", "r").read()
        fout.write("{}\n".format(result))
        # fout.write(metric)
    os.remove("temp")
    return metric, trigger_p_, trigger_r_, trigger_f1_


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", type=str, default="logdir")
    parser.add_argument("--batch_size", type=int, default=48)
    parser.add_argument("--testset", type=str, default="data/test.json")
    parser.add_argument("--model_path", type=str,
                        default="eval_test.P0.80_R0.73_F0.76")

    hp = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # torch.cuda.set_device(1)

    test_dataset = ACE2005Dataset(hp.testset)

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
    eval(model, test_iter, 'eval_test')
