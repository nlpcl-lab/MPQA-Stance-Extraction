import numpy as np
import torch
from torch.utils import data
import json

from consts import NONE, PAD, CLS, SEP, UNK, ATTITUDES, ARGUMENTS, ENTITIES
from utils import build_vocab
from pytorch_pretrained_bert import BertTokenizer

# init vocab
all_attitudes, attitude2idx, idx2attitude = build_vocab(ATTITUDES)
all_entities, entity2idx, idx2entity = build_vocab(ENTITIES)
all_arguments, argument2idx, idx2argument = build_vocab(
    ARGUMENTS, BIO_tagging=False)






class MPQADataset(data.Dataset):
    def __init__(self, fpath, bert_size='base'):
        self.sent_li, self.entities_li, self.attitudes_li, self.arguments_li = [], [], [], []

        bert_string = 'bert-{}-uncased'.format(bert_size)
        self.tokenizer = BertTokenizer.from_pretrained(bert_string, do_lower_case=True, never_split=(PAD, CLS, SEP, UNK))

        with open(fpath, 'r') as f:
            data = json.load(f)
            assert isinstance(data, list)
            for item_index, item in enumerate(data):

                words = item['words']
                entities = [[NONE] for _ in range(len(words))]
                attitudes = [NONE] * len(words)

                arguments = {
                    'candidates': [
                        # ex. (5, 6, "entity_type_str"), ...
                    ],
                    'events': {
                        # ex. (1, 3, "trigger_type_str"): [(5, 6, "argument_role_idx"), ...]
                    },
                }
                """
                for entity_mention in item['golden-entity-mentions']:
                    arguments['candidates'].append(
                        (entity_mention['start'], entity_mention['end'], entity_mention['entity-type']))

                    for i in range(entity_mention['start'], entity_mention['end']):
                        entity_type = entity_mention['entity-type']
                        if i == entity_mention['start']:
                            entity_type = 'B-{}'.format(entity_type)
                        else:
                            entity_type = 'I-{}'.format(entity_type)

                        if len(entities[i]) == 1 and entities[i][0] == NONE:
                            entities[i][0] = entity_type
                        else:
                            entities[i].append(entity_type)
                """

                for att in item['attitudes']:
                    if att['source'] != "w":
                        continue
                    for i in range(att['trigger']['start'], att['trigger']['end']):
                        att_type = att['att_type']
                        if att_type not in ATTITUDES:
                            continue
                        if i == att['trigger']['start']:
                            attitudes[i] = 'B-{}'.format(att_type)
                        else:
                            attitudes[i] = 'I-{}'.format(att_type)

                    event_key = (
                        att['trigger']['start'], att['trigger']['end'], att['att_type'])
                    arguments['events'][event_key] = []
                    """
                    for argument in event_mention['arguments']:
                        role = argument['role']
                        if role.startswith('Time'):
                            role = role.split('-')[0]
                        arguments['events'][event_key].append(
                            (argument['start'], argument['end'], argument2idx[role]))
                    """
                # self.sent_li.append([CLS] + words + [SEP])
                # self.entities_li.append([[PAD]] + entities + [[PAD]])
                self.sent_li.append([CLS] + words)
                self.entities_li.append([[PAD]] + entities)

                self.attitudes_li.append(attitudes)
                self.arguments_li.append(arguments)

    def __len__(self):
        return len(self.sent_li)

    def __getitem__(self, idx): # gets sentence idx, tokenizes all, and returns the information of one sentence
        words, entities, attitudes, arguments = self.sent_li[idx], self.entities_li[
            idx], self.attitudes_li[idx], self.arguments_li[idx]

        # We give credits only to the first piece.
        tokens_x, entities_x, is_heads = [], [], []
        for w, e in zip(words, entities): # per word
            tokens = self.tokenizer.tokenize(w) if w not in [CLS, SEP] else [w]
            tokens_xx = self.tokenizer.convert_tokens_to_ids(tokens)

            if w in [CLS, SEP]:
                is_head = [0]
            else:
                is_head = [1] + [0] * (len(tokens) - 1)

            e = [e] + [[PAD]] * (len(tokens) - 1)  # <PAD>: no decision
            e = [[entity2idx[entity] for entity in entities] for entities in e]

            tokens_x.extend(tokens_xx), entities_x.extend(e), is_heads.extend(is_head)

        attitudes_y = [attitude2idx[t] for t in attitudes]
        head_indexes = []
        for i in range(len(is_heads)):
            if is_heads[i]:
                head_indexes.append(i)

        seqlen = len(tokens_x)

        return tokens_x, entities_x, attitudes_y, arguments, seqlen, head_indexes, words, attitudes

    def get_samples_weight(self):
        samples_weight = []
        for attitudes in self.attitudes_li:
            not_none = False
            for attitude in attitudes:
                if attitude != NONE:
                    not_none = True
                    break
            if not_none:
                samples_weight.append(5.0)
            else:
                samples_weight.append(1.0)
        return np.array(samples_weight)


def pad(batch):
    tokens_x_2d, entities_x_3d, attitudes_y_2d, arguments_2d, seqlens_1d, head_indexes_2d, words_2d, attitudes_2d = list(
        map(list, zip(*batch)))
    maxlen = np.array(seqlens_1d).max()

    for i in range(len(tokens_x_2d)):
        tokens_x_2d[i] = tokens_x_2d[i] + [0] * (maxlen - len(tokens_x_2d[i]))
        head_indexes_2d[i] = head_indexes_2d[i] + \
            [0] * (maxlen - len(head_indexes_2d[i]))
        attitudes_y_2d[i] = attitudes_y_2d[i] + \
            [attitude2idx[PAD]] * (maxlen - len(attitudes_y_2d[i]))
        entities_x_3d[i] = entities_x_3d[i] + [[entity2idx[PAD]]
                                               for _ in range(maxlen - len(entities_x_3d[i]))]

    return tokens_x_2d, entities_x_3d, \
        attitudes_y_2d, arguments_2d, \
        seqlens_1d, head_indexes_2d, \
        words_2d, attitudes_2d
