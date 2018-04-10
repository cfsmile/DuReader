#! /usr/bin/env python
# -*- coding: utf-8 -*-

import json
from collections import Counter


def load_dataset(data_path, train=True):
    """
    Loads the dataset
    Args:
        data_path: the data file to load
        train: if this process is train or test
    """
    with open(data_path) as fin:
        data_set = []
        for lidx, line in enumerate(fin):
            sample = json.loads(line.strip())
            if train:
                #  if length of answer_spans is zero, or

                if len(sample['answer_spans']) == 0:
                    continue
                #  if length of answer_spans is larger than
                #  paragraph itself, execute 'continue', which
                #  abandons the rest code of this loop.
                #  These two kinds of train sample won't be
                #  viewed as legal sample and will be excluded.
                if sample['answer_spans'][0][1] >= 500:  # self.max_p_len:
                    continue

            if 'answer_docs' in sample:
                sample['answer_passages'] = sample['answer_docs']

            sample['question_tokens'] = sample['segmented_question']

            sample['passages'] = []
            for d_idx, doc in enumerate(sample['documents']):
                if train:
                    most_related_para = doc['most_related_para']
                    sample['passages'].append(
                        {'passage_tokens': doc['segmented_paragraphs'][most_related_para],
                         'is_selected': doc['is_selected']}
                    )
                else:
                    para_infos = []
                    for para_tokens in doc['segmented_paragraphs']:
                        question_tokens = sample['segmented_question']
                        common_with_question = Counter(para_tokens) & Counter(question_tokens)
                        correct_preds = sum(common_with_question.values())
                        if correct_preds == 0:
                            recall_wrt_question = 0
                        else:
                            recall_wrt_question = float(correct_preds) / len(question_tokens)
                        para_infos.append((para_tokens, recall_wrt_question, len(para_tokens)))
                    para_infos.sort(key=lambda x: (-x[1], x[2]))
                    fake_passage_tokens = []
                    for para_info in para_infos[:1]:
                        fake_passage_tokens += para_info[0]
                    sample['passages'].append({'passage_tokens': fake_passage_tokens})
            data_set.append(sample)

    with open('tmp.txt', 'w') as fout:
        for data in data_set:
            fout.write(json.dumps(data, ensure_ascii=False) + '\n')
    return


def main():
    load_dataset("split_zd_json00_aa", train=True)


if __name__ == '__main__':
    main()
