# -*- coding: utf-8 -*-
'''
Evaluation script for CMRC 2018
version: v5 - special
Note: 
v5 - special: Evaluate on SQuAD-style CMRC 2018 Datasets
v5: formatted output, add usage description
v4: fixed segmentation issues
'''
from __future__ import print_function
import pdb
import nltk
from collections import Counter, OrderedDict
import string
import re
import argparse
import json
import sys
reload(sys)
sys.setdefaultencoding('utf8')

# split Chinese with English


def mixed_segmentation(in_str, rm_punc=False):
    in_str = str(in_str).decode('utf-8').lower().strip()
    segs_out = []
    temp_str = ""
    sp_char = ['-', ':', '_', '*', '^', '/', '\\', '~', '`', '+', '=',
               '，', '。', '：', '？', '！', '“', '”', '；', '’', '《', '》', '……', '·', '、',
               '「', '」', '（', '）', '－', '～', '『', '』']
    for char in in_str:
        if rm_punc and char in sp_char:
            continue
        if re.search(ur'[\u4e00-\u9fa5]', char) or char in sp_char:
            if temp_str != "":
                ss = nltk.word_tokenize(temp_str)
                segs_out.extend(ss)
                temp_str = ""
            segs_out.append(char)
        else:
            temp_str += char

    # handling last part
    if temp_str != "":
        ss = nltk.word_tokenize(temp_str)
        segs_out.extend(ss)

    return segs_out


# remove punctuation

... [truncated] ...
f1_score = 100.0 * f1 / total_count
em_score = 100.0 * em / total_count
return f1_score, em_score, total_count, skip_count


def calc_f1_score(answers, prediction):
    f1_scores = []
    for ans in answers:
        ans_segs = mixed_segmentation(ans, rm_punc=True)
        prediction_segs = mixed_segmentation(prediction, rm_punc=True)
        lcs, lcs_len = find_lcs(ans_segs, prediction_segs)
        if lcs_len == 0:
            f1_scores.append(0)
            continue
        precision = 1.0*lcs_len/len(prediction_segs)
        recall = 1.0*lcs_len/len(ans_segs)
        f1 = (2*precision*recall)/(precision+recall)
        f1_scores.append(f1)
    return max(f1_scores)


def calc_em_score(answers, prediction):
    em = 0
    for ans in answers:
        ans_ = remove_punctuation(ans)
        prediction_ = remove_punctuation(prediction)
        if ans_ == prediction_:
            em = 1
            break
    return em


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluation Script for CMRC 2018')
    parser.add_argument('dataset_file', help='Official dataset file')
    parser.add_argument('prediction_file', help='Your prediction File')
    args = parser.parse_args()
    ground_truth_file = json.load(open(args.dataset_file, 'rb'))
    prediction_file = json.load(open(args.prediction_file, 'rb'))
    F1, EM, TOTAL, SKIP = evaluate(ground_truth_file, prediction_file)
    AVG = (EM+F1)*0.5
    output_result = OrderedDict()
    output_result['AVERAGE'] = '%.3f' % AVG
    output_result['F1'] = '%.3f' % F1
    output_result['EM'] = '%.3f' % EM
    output_result['TOTAL'] = TOTAL
    output_result['SKIP'] = SKIP
    output_result['FILE'] = args.prediction_file
    print(json.dumps(output_result))
