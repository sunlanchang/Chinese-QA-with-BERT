# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Run BERT on SQuAD 1.1 and SQuAD 2.0."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import json
import math
import os
import random
import modeling
import optimization
import tokenization
import six
import tensorflow as tf
import numpy
import pdb

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('train_url', None, 'None help')
flags.DEFINE_string('data_url', None, 'No help')
flags.DEFINE_string('num_gpus', None, 'No help')

# Required parameters
flags.DEFINE_string(
    "bert_config_file", None,
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written.")

# Other parameters
flags.DEFINE_string("train_file", None,
                    "SQuAD json for training. E.g., train-v1.1.json")

flags.DEFINE_string(
    "predict_file", None,
    "SQuAD json for predictions. E.g., dev-v1.1.json or test-v1.1.json")

flags.DEFINE_string(
    "eval_file", None,
    "SQuAD json for predictions. E.g., dev-v1.1.json or test-v1.1.json")

flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_bool(
    "do_lower_case", False,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_integer(
    "max_seq_length", 384,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_integer(
    "doc_stride", 128,
    "When splitting up a long document into chunks, how much stride to "
    "take between chunks.")

flags.DEFINE_integer(
    "max_query_length", 64,
    "The maximum number of tokens for the question. Questions longer than "
    "this will be truncated to this length.")

flags.DEFINE_bool("do_train", False, "Whether to run training.")

flags.DEFINE_bool("do_predict", False, "Whether to run eval on the dev set.")

flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")

flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")

flags.DEFINE_integer("predict_batch_size", 8,
                     "Total batch size for predictions.")

flags.DEFINE_float("learning_rate", 5e-5,
                   "The initial learning rate for Adam.")

flags.DEFINE_float("num_train_epochs", 3.0,
                   "Total number of training epochs to perform.")

flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_integer(
    "n_best_size", 20,
    "The total number of n-best predictions to generate in the "
    "nbest_predictions.json output file.")

flags.DEFINE_integer(
    "max_answer_length", 30,
    "The maximum length of an answer that can be generated. This is needed "
    "because the start and end predictions are not conditioned on one another.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

tf.flags.DEFINE_string(
    "tpu_name", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

tf.flags.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")

flags.DEFINE_bool(
    "verbose_logging", False,
    "If true, all of the warnings related to data processing will be printed. "
    "A number of warnings are expected for a normal SQuAD evaluation.")

flags.DEFINE_integer("rand_seed", 12345, "set random seed")

# set random seed (i don't know whether it works or not)
numpy.random.seed(int(FLAGS.rand_seed))
tf.set_random_seed(int(FLAGS.rand_seed))

#


class SquadExample(object):
    """A single training/test example for simple sequence classification.

       For examples without an answer, the start and end position are -1.
    """

    def __init__(self,
                 qas_id,
                 question_text,
                 doc_tokens,
                 orig_answer_text=None,
                 start_position=None,
                 end_position=None):
        self.qas_id = qas_id
        self.question_text = question_text
        self.doc_tokens = doc_tokens
        self.orig_answer_text = orig_answer_text
        self.start_position = start_position
        self.end_position = end_position

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        s += "qas_id: %s" % (tokenization.printable_text(self.qas_id))
        s += ", question_text: %s" % (
            tokenization.printable_text(self.question_text))
        s += ", doc_tokens: [%s]" % (" ".join(self.doc_tokens))
        if self.start_position:
            s += ", start_position: %d" % (self.start_position)
        if self.start_position:
            s += ", end_position: %d" % (self.end_position)
        return s


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 unique_id,
                 example_index,
                 doc_span_index,
                 tokens,
                 token_to_orig_map,
                 token_is_max_context,
                 input_ids,
                 input_mask,
                 segment_ids,
                 input_span_mask,
                 start_position=None,
                 end_position=None):
        self.unique_id = unique_id
        self.example_index = example_index
        self.doc_span_index = doc_span_index
        self.tokens = tokens
        self.token_to_orig_map = token_to_orig_map
        self.token_is_max_context = token_is_max_context
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.input_span_mask = input_span_mask
        self.start_position = start_position
        self.end_position = end_position

#


def customize_tokenizer(text, do_lower_case=False):
    tokenizer = tokenization.BasicTokenizer(do_lower_case=do_lower_case)
    temp_x = ""
    text = tokenization.convert_to_unicode(text)
    for c in text:
        if tokenizer._is_chinese_char(ord(c)) or tokenization._is_punctuation(c) or tokenization._is_whitespace(c) or tokenization._is_control(c):
            temp_x += " " + c + " "
        else:
            temp_x += c
    if do_lower_case:
        temp_x = temp_x.lower()
    return temp_x.split()

#


class ChineseFullTokenizer(object):
    """Runs end-to-end tokenziation."""

    def __init__(self, vocab_file, do_lower_case=False):
        self.vocab = tokenization.load_vocab(vocab_file)
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        self.wordpiece_tokenizer = tokenization.WordpieceTokenizer(
            vocab=self.vocab)
        self.do_lower_case = do_lower_case

    def tokenize(self, text):
        split_tokens = []
        for token in customize_tokenizer(text, do_lower_case=self.do_lower_case):
            for sub_token in self.wordpiece_tokenizer.tokenize(token):
                split_tokens.append(sub_token)

        return split_tokens

    def convert_tokens_to_ids(self, tokens):
        return tokenization.convert_by_vocab(self.vocab, tokens)

    def convert_ids_to_tokens(self, ids):
        return tokenization.convert_by_vocab(self.inv_vocab, ids)

#


def read_squad_examples(input_file, is_training):
    """Read a SQuAD json file into a list of SquadExample."""
    with tf.gfile.Open(input_file, "r") as reader:
        input_data = json.load(reader)["data"]

    #
    examples = []
    for entry in input_data:
        for paragraph in entry["paragraphs"]:
            paragraph_text = paragraph["context"]
            raw_doc_tokens = customize_tokenizer(
                paragraph_text, do_lower_case=FLAGS.do_lower_case)
            doc_tokens = []
            char_to_word_offset = []
            prev_is_whitespace = True

            k = 0
            temp_word = ""
            for c in paragraph_text:
                if tokenization._is_whitespace(c):
                    char_to_word_offset.append(k-1)
                    continue
                else:
                    temp_word += c
                    char_to_word_offset.append(k)
                if FLAGS.do_lower_case:
                    temp_word = temp_word.lower()
                if temp_word == raw_doc_tokens[k]:
                    doc_tokens.append(temp_word)
                    temp_word = ""
                    k += 1

            assert k == len(raw_doc_tokens)

            for qa in paragraph["qas"]:
                qas_id = qa["id"]
                question_text = qa["question"]
                start_position = None
                end_position = None
                orig_answer_text = None

                if is_training:
                    answer = qa["answers"][0]
                    orig_answer_text = answer["text"]

                    if orig_answer_text not in paragraph_text:
                        tf.logging.warning("Could not find answer")
                    else:
                        answer_offset = paragraph_text.index(orig_answer_text)
                        answer_length = len(orig_answer_text)
                        start_position = char_to_word_offset[answer_offset]
                        end_position = char_to_word_offset[answer_offset +
                                                           answer_length - 1]

                        # Only add answers where the text can be exactly recovered from the
                        # document. If this CAN'T happen it's likely due to weird Unicode
                        # stuff so we will just skip the example.
                        #
                        # Note that this means for training mode, every example is NOT
                        # guaranteed to be preserved.
                        actual_text = "".join(
                            doc_tokens[start_position:(end_position + 1)])
                        cleaned_answer_text = "".join(
                            tokenization.whitespace_tokenize(orig_answer_text))
                        if FLAGS.do_lower_case:
                            cleaned_answer_text = cleaned_answer_text.lower()
                        if actual_text.find(cleaned_answer_text) == -1:
                            tf.logging.warning(
                                "Could not find answer: '%s' vs. '%s'", actual_text, cleaned_answer_text)
                            continue

                example = SquadExample(
                    qas_id=qas_id,
                    question_text=question_text,
                    doc_tokens=doc_tokens,
                    orig_answer_text=orig_answer_text,
                    start_position=start_position,
                    end_position=end_position)
                examples.append(example)
    tf.logging.info("**********read_squad_examples complete!**********")

    return examples


def convert_examples_to_features(examples, tokenizer, max_seq_length,
                                 doc_stride, max_query_length, is_training,
                                 output_fn):
    """Loads a data file into a list of `InputBatch`s."""

    unique_id = 1000000000
    tokenizer = ChineseFullTokenizer(
        vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

    import ipdb
    ipdb.set_trace()
    for (example_index, example) in enumerate(examples):
        query_tokens = tokenizer.tokenize(example.question_text)

        if len(query_tokens) > max_query_length:
            query_tokens = query_tokens[0:max_query_length]

        tok_to_orig_index = []
        orig_to_tok_index = []
        all_doc_tokens = []
        for (i, token) in enumerate(example.doc_tokens):
            orig_to_tok_index.append(len(all_doc_tokens))
            sub_tokens = tokenizer.tokenize(token)
            for sub_token in sub_tokens:
                tok_to_orig_index.append(i)
                all_doc_tokens.append(sub_token)

        tok_start_position = None
        tok_end_position = None
        if is_training:
            tok_start_position = orig_to_tok_index[example.start_position]
            if example.end_position < len(example.doc_tokens) - 1:
                tok_end_position = orig_to_tok_index[example.end_position + 1] - 1
            else:
                tok_end_position = len(all_doc_tokens) - 1
            (tok_start_position, tok_end_position) = _improve_answer_span(
                all_doc_tokens, tok_start_position, tok_end_position, tokenizer,
                example.orig_answer_text)

        # The -3 accounts for [CLS], [SEP] and [SEP]
        max_tokens_for_doc = max_seq_length - len(query_tokens) - 3

        # We can have documents that are longer than the maximum sequence length.
        # To deal with this we do a sliding window approach, where we take chunks
        # of the up to our max length with a stride of `doc_stride`.
        _DocSpan = collections.namedtuple(  # pylint: disable=invalid-name
            "DocSpan", ["start", "length"])
        doc_spans = []
        start_offset = 0
        while start_offset < len(all_doc_tokens):
            length = len(all_doc_tokens) - start_offset
            if length > max_tokens_for_doc:
                length = max_tokens_for_doc
            doc_spans.append(_DocSpan(start=start_offset, length=length))
            if start_offset + length == len(all_doc_tokens):
                break
            start_offset += min(length, doc_stride)

        for (doc_span_index, doc_span) in enumerate(doc_spans):
            tokens = []
            token_to_orig_map = {}
            token_is_max_context = {}
            segment_ids = []
            input_span_mask = []
            tokens.append("[CLS]")
            segment_ids.append(0)
            input_span_mask.append(1)
            for token in query_tokens:
                tokens.append(token)
                segment_ids.append(0)
                input_span_mask.append(0)
            tokens.append("[SEP]")
            segment_ids.append(0)
            input_span_mask.append(0)

            for i in range(doc_span.length):
                split_token_index = doc_span.start + i
                token_to_orig_map[len(
                    tokens)] = tok_to_orig_index[split_token_index]

                is_max_context = _check_is_max_context(doc_spans, doc_span_index,
                                                       split_token_index)
                token_is_max_context[len(tokens)] = is_max_context
                tokens.append(all_doc_tokens[split_token_index])
                segment_ids.append(1)
                input_span_mask.append(1)
            tokens.append("[SEP]")
            segment_ids.append(1)
            input_span_mask.append(0)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            while len(input_ids) < max_seq_length:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)
                input_span_mask.append(0)

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length
            assert len(input_span_mask) == max_seq_length

            start_position = None
            end_position = None
            if is_training:
                # For training, if our document chunk does not contain an annotation
                # we throw it out, since there is nothing to predict.
                doc_start = doc_span.start
                doc_end = doc_span.start + doc_span.length - 1
                out_of_span = False
                if not (tok_start_position >= doc_start and
                        tok_end_position <= doc_end):
                    out_of_span = True
                if out_of_span:
                    start_position = 0
                    end_position = 0
                else:
                    doc_offset = len(query_tokens) + 2
                    start_position = tok_start_position - doc_start + doc_offset
                    end_position = tok_end_position - doc_start + doc_offset

            if example_index < 3:
                tf.logging.info("*** Example ***")
                tf.logging.info("unique_id: %s" % (unique_id))
                tf.logging.info("example_index: %s" % (example_index))
                tf.logging.info("doc_span_index: %s" % (doc_span_index))
                tf.logging.info("tokens: %s" % " ".join(
                    [tokenization.printable_text(x) for x in tokens]))
                tf.logging.info("token_to_orig_map: %s" % " ".join(
                    ["%d:%d" % (x, y) for (x, y) in six.iteritems(token_to_orig_map)]))
                tf.logging.info("token_is_max_context: %s" % " ".join([
                    "%d:%s" % (x, y) for (x, y) in six.iteritems(token_is_max_context)
                ]))
                tf.logging.info("input_ids: %s" %
                                " ".join([str(x) for x in input_ids]))
                tf.logging.info(
                    "input_mask: %s" % " ".join([str(x) for x in input_mask]))
                tf.logging.info(
                    "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
                tf.logging.info(
                    "input_span_mask: %s" % " ".join([str(x) for x in input_span_mask]))
                if is_training:
                    answer_text = " ".join(
                        tokens[start_position:(end_position + 1)])
                    tf.logging.info("start_position: %d" % (start_position))
                    tf.logging.info("end_position: %d" % (end_position))
                    tf.logging.info(
                        "answer: %s" % (tokenization.printable_text(answer_text)))
            '''
            unique_id：是问题唯一的id
            print(unique_id)如下：
            1000000000

            example_index：输入模型的example的index
            print(example_index)如下：
            0

            doc_span_index：该feature在doc_span的索引，如果一个文本很长，那么肯定要对其进行截取成若干片段转成doc_span,
            doc_span里面的各个片段都会装进各个feature里面，所以一个feature就会有一个doc_span_index。
            print(doc_span_index)如下：
            0

            token_to_orig_map：tokens里面每一个token在原始doc_token的索引。
            print(token_to_orig_map)如下：
            {22: 0, 23: 1, 24: 2, 25: 3, 26: 4, 27: 5, 28: 6, 29: 7, 30: 8, 31: 9, 32: 10, 33: 11, 34: 12, 35: 13, 36: 14, 37: 15, 38: 16, 39: 17, 40: 18, 41: 19, 42: 20, 43: 21, 44: 22, 45: 23, 46: 24, 47: 25, 48: 26, 49: 27, 50: 28, 51: 29, 52: 30, 53: 31, 54: 32, 55: 33, 56: 34, 57: 35, 58: 36, 59: 37, 60: 38, 61: 39, 62: 40, 63: 41, 64: 42, 65: 43, 66: 44, 67: 45, 68: 46, 69: 47, 70: 48, 71: 49, 72: 50, 73: 51, 74: 52, 75: 53, 76: 54, 77: 55, 78: 56, 79: 57, 80: 58, 81: 59, 82: 60, 83: 61, 84: 62, 85: 63, 86: 64, 87: 65, 88: 66, 89: 67, 90: 68, 91: 69, 92: 70, 93: 71, 94: 72, 95: 73, 96: 74, 97: 75, 98: 76, 99: 77, 100: 78, 101: 79, 102: 80, 103: 81, 104: 82, 105: 83, 106: 84, 107: 85, 108: 86, 109: 87, 110: 88, 111: 89, 112: 90, 113: 91, 114: 92, 115: 93, 116: 94, 117: 95, 118: 96, 119: 97, 120: 98, 121: 99, 122: 100, 123: 101, 124: 102, 125: 103, 126: 104, 127: 105, 128: 106, 129: 107, 130: 108, 131: 109, 132: 110, 133: 111, 134: 112, 135: 113, 136: 114, 137: 115, 138: 116, 139: 117, 140: 118, 141: 119, 142: 120, 143: 121, 144: 122, 145: 123, 146: 124, 147: 125, 148: 126, 149: 127, 150: 128, 151: 129, 152: 130, 153: 131, 154: 132, 155: 133, 156: 134, 157: 135, 158: 136, 159: 137, 160: 138, 161: 139, 162: 140, 163: 141, 164: 142, 165: 143, 166: 144, 167: 145, 168: 146, 169: 147, 170: 148, 171: 149, 172: 150, 173: 151, 174: 152, 175: 153, 176: 154, 177: 155, 178: 156, 179: 157, 180: 158, 181: 159, 182: 160, 183: 161, 184: 162, 185: 163, 186: 164, 187: 165, 188: 166, 189: 167, 190: 168, 191: 169, 192: 170, 193: 171, 194: 172, 195: 173, 196: 174, 197: 175, 198: 176, 199: 177, 200: 178, 201: 179, 202: 180, 203: 181, 204: 182, 205: 183, 206: 184, 207: 185, 208: 186, 209: 187, 210: 188, 211: 189, 212: 190, 213: 191, 214: 192, 215: 193, 216: 194, 217: 195, 218: 196, 219: 197, 220: 198, 221: 199, 222: 200, 223: 201, 224: 202, 225: 203, 226: 204, 227: 205, 228: 206, 229: 207, 230: 208, 231: 209, 232: 210, 233: 211, 234: 212, 235: 213, 236: 214, 237: 215, 238: 216, 239: 217, 240: 218, 241: 219, 242: 220, 243: 221, 244: 222, 245: 223, 246: 224, 247: 225, 248: 226, 249: 227, 250: 228, 251: 229, 252: 230, 253: 231, 254: 232, 255: 233, 256: 234, 257: 235, 258: 236, 259: 237, 260: 238, 261: 239, 262: 240, 263: 241, 264: 242, 265: 243, 266: 244, 267: 245, 268: 246, 269: 247, 270: 248, 271: 249, 272: 250, 273: 251, 274: 252, 275: 253, 276: 254, 277: 255, 278: 256, 279: 257, 280: 258, 281: 259, 282: 260, 283: 261, 284: 262, 285: 263, 286: 264, 287: 265, 288: 266, 289: 267, 290: 268, 291: 269, 292: 270, 293: 271, 294: 272, 295: 273, 296: 274, 297: 275, 298: 276, 299: 277, 300: 278, 301: 279, 302: 280, 303: 281, 304: 282, 305: 283, 306: 284, 307: 285, 308: 286, 309: 287, 310: 288, 311: 289, 312: 290, 313: 291, 314: 292, 315: 293, 316: 294, 317: 295, 318: 296, 319: 297, 320: 298, 321: 299, 322: 300, 323: 301, 324: 302, 325: 303, 326: 304, 327: 305, 328: 306, 329: 307, 330: 308, 331: 309, 332: 310, 333: 311, 334: 312, 335: 313, 336: 314, 337: 315, 338: 316, 339: 317, 340: 318, 341: 319, 342: 320, 343: 321, 344: 322, 345: 323}

            tokens：就是模型需要的数据，由context和question组成。
            print(toens)如下：
            ['[CLS]', '为', '什', '么', '鄂', '图', '曼', '军', '队', '纷', '纷', '丢', '弃', '阵', '地', '向', '内', '陆', '退', '却', '？', '[SEP]', 
            '英', '法', '两', '国', '投', '入', '战', '役', '共', '计', '62', '艘', '战', '舰', '以', '及', '大', '量', '辅', '助', '船', '只', '，',
             '并', '指', '定', '英', '国', '皇', '家', '海', '军', '地', '中', '海', '舰', '队', '司', '令', '萨', '克', '维', '尔', '·', '卡', '登', 
            '上', '将', '负', '责', '指', '挥', '这', '次', '战', '役', '。', '舰', '队', '从', '1915', '年', '2', '月', '19', '日', '起', '开', '始',
            '炮', '轰', '达', '达', '尼', '尔', '海', '峡', '。', '3', '月', '18', '日', '，', '16', '艘', '军', '舰', '企', '图', '强', '行', '闯',
            '入', '狭', '窄', '的', '海', '峡', '通', '道', '，', '4', '艘', '军', '舰', '触', '发', '水', '雷', '，', '舰', '只', '慌', '忙', '撤',
            '退', '。', '在', '陆', '地', '上', '，', '鄂', '图', '曼', '军', '队', '因', '遭', '受', '突', '然', '袭', '击', '，', '纷', '纷', '丢',
            '弃', '阵', '地', '向', '内', '陆', '退', '却', '，', '英', '国', '突', '击', '部', '队', '在', '没', '有', '遇', '到', '抵', '抗', '的',
            '情', '况', '下', '冲', '上', '海', '岸', '。', '德', '国', '军', '事', '顾', '问', '奥', '托', '·', '冯', '·', '桑', '德', '斯', '洞',
            '悉', '对', '方', '计', '划', '登', '陆', '加', '里', '波', '利', '，', '火', '速', '调', '动', '军', '队', '至', '战', '区', '。', 
            '鄂', '图', '曼', '军', '队', '掘', '壕', '坚', '守', '，', '依', '据', '半', '岛', '复', '杂', '的', '地', '形', '建', '立', '了',
            '强', '大', '的', '防', '御', '体', '系', '，', '又', '在', '该', '地', '集', '结', '炮', '兵', '部', '队', '。', '在', '英', '法',
            '军', '队', '准', '备', '扩', '大', '战', '果', '时', '，', '隐', '蔽', '在', '阵', '地', '中', '的', '鄂', '图', '曼', '士', '兵',
            '一', '起', '开', '火', '，', '把', '正', '在', '攀', '登', '悬', '崖', '的', '英', '军', '打', '的', '措', '手', '不', '及', '。',
            '3', '月', '3', '日', '，', '联', '军', '的', '首', '轮', '登', '陆', '行', '动', '宣', '告', '失', '败', '，', '卡', '登', '上', 
            '将', '因', '伤', '送', '回', '英', '国', '。', '[SEP]']

            input_ids：tokens转化为token index作为模型的输入。
            print(input_ids)
            {22: True, 23: True, 24: True, 25: True, 26: True, 27: True, 28: True, 29: True, 30: True, 31: True, 32: True, 33: True, 34: True, 35: True, 36: True, 37: True, 38: True, 39: True, 40: True, 41: True, 42: True, 43: True, 44: True, 45: True, 46: True, 47: True, 48: True, 49: True, 50: True, 51: True, 52: True, 53: True, 54: True, 55: True, 56: True, 57: True, 58: True, 59: True, 60: True, 61: True, 62: True, 63: True, 64: True, 65: True, 66: True, 67: True, 68: True, 69: True, 70: True, 71: True, 72: True, 73: True, 74: True, 75: True, 76: True, 77: True, 78: True, 79: True, 80: True, 81: True, 82: True, 83: True, 84: True, 85: True, 86: True, 87: True, 88: True, 89: True, 90: True, 91: True, 92: True, 93: True, 94: True, 95: True, 96: True, 97: True, 98: True, 99: True, 100: True, 101: True, 102: True, 103: True, 104: True, 105: True, 106: True, 107: True, 108: True, 109: True, 110: True, 111: True, 112: True, 113: True, 114: True, 115: True, 116: True, 117: True, 118: True, 119: True, 120: True, 121: True, 122: True, 123: True, 124: True, 125: True, 126: True, 127: True, 128: True, 129: True, 130: True, 131: True, 132: True, 133: True, 134: True, 135: True, 136: True, 137: True, 138: True, 139: True, 140: True, 141: True, 142: True, 143: True, 144: True, 145: True, 146: True, 147: True, 148: True, 149: True, 150: True, 151: True, 152: True, 153: True, 154: True, 155: True, 156: True, 157: True, 158: True, 159: True, 160: True, 161: True, 162: True, 163: True, 164: True, 165: True, 166: True, 167: True, 168: True, 169: True, 170: True, 171: True, 172: True, 173: True, 174: True, 175: True, 176: True, 177: True, 178: True, 179: True, 180: True, 181: True, 182: True, 183: True, 184: True, 185: True, 186: True, 187: True, 188: True, 189: True, 190: True, 191: True, 192: True, 193: True, 194: True, 195: True, 196: True, 197: True, 198: True, 199: True, 200: True, 201: True, 202: True, 203: True, 204: True, 205: True, 206: True, 207: True, 208: True, 209: True, 210: True, 211: True, 212: True, 213: True, 214: True, 215: True, 216: True, 217: True, 218: True, 219: True, 220: True, 221: True, 222: True, 223: True, 224: True, 225: True, 226: True, 227: True, 228: True, 229: True, 230: True, 231: True, 232: True, 233: True, 234: True, 235: True, 236: True, 237: True, 238: True, 239: True, 240: True, 241: True, 242: True, 243: True, 244: True, 245: True, 246: True, 247: True, 248: True, 249: True, 250: True, 251: True, 252: True, 253: True, 254: True, 255: True, 256: True, 257: True, 258: True, 259: True, 260: True, 261: True, 262: True, 263: True, 264: True, 265: True, 266: True, 267: True, 268: True, 269: True, 270: True, 271: True, 272: True, 273: True, 274: True, 275: True, 276: True, 277: True, 278: True, 279: True, 280: True, 281: True, 282: True, 283: True, 284: True, 285: True, 286: True, 287: True, 288: True, 289: True, 290: True, 291: True, 292: True, 293: True, 294: True, 295: True, 296: True, 297: True, 298: True, 299: True, 300: True, 301: True, 302: True, 303: True, 304: True, 305: True, 306: True, 307: True, 308: True, 309: True, 310: True, 311: True, 312: True, 313: True, 314: True, 315: True, 316: True, 317: True, 318: True, 319: True, 320: True, 321: True, 322: True, 323: True, 324: True, 325: True, 326: True, 327: True, 328: True, 329: True, 330: True, 331: True, 332: True, 333: True, 334: True, 335: True, 336: True, 337: True, 338: True, 339: True, 340: True, 341: True, 342: True, 343: True, 344: True, 345: True}

            segment_ids：将question、context和padding部分划分开。
            print(segment_ids)
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

            input_span_mask：将输入模型部分和padding部分划分开。
            print(input_span_mask)
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            
            start_position与end_position：answer的位置。
            print(start_position, end_position)
            150,155
            '''
            feature = InputFeatures(
                unique_id=unique_id,
                example_index=example_index,
                doc_span_index=doc_span_index,
                tokens=tokens,
                token_to_orig_map=token_to_orig_map,
                token_is_max_context=token_is_max_context,
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                input_span_mask=input_span_mask,
                start_position=start_position,
                end_position=end_position)

            # Run callback
            output_fn(feature)

            unique_id += 1


def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer,
                         orig_answer_text):
    """Returns tokenized answer spans that better match the annotated answer."""

    # The SQuAD annotations are character based. We first project them to
    # whitespace-tokenized words. But then after WordPiece tokenization, we can
    # often find a "better match". For example:
    #
    #   Question: What year was John Smith born?
    #   Context: The leader was John Smith (1895-1943).
    #   Answer: 1895
    #
    # The original whitespace-tokenized answer will be "(1895-1943).". However
    # after tokenization, our tokens will be "( 1895 - 1943 ) .". So we can match
    # the exact answer, 1895.
    #
    # However, this is not always possible. Consider the following:
    #
    #   Question: What country is the top exporter of electornics?
    #   Context: The Japanese electronics industry is the lagest in the world.
    #   Answer: Japan
    #
    # In this case, the annotator chose "Japan" as a character sub-span of
    # the word "Japanese". Since our WordPiece tokenizer does not split
    # "Japanese", we just use "Japanese" as the annotation. This is fairly rare
    # in SQuAD, but does happen.
    tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))

    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = " ".join(doc_tokens[new_start:(new_end + 1)])
            if text_span == tok_answer_text:
                return (new_start, new_end)

    return (input_start, input_end)


def _check_is_max_context(doc_spans, cur_span_index, position):
    """Check if this is the 'max context' doc span for the token."""

    # Because of the sliding window approach taken to scoring documents, a single
    # token can appear in multiple documents. E.g.
    #  Doc: the man went to the store and bought a gallon of milk
    #  Span A: the man went to the
    #  Span B: to the store and bought
    #  Span C: and bought a gallon of
    #  ...
    #
    # Now the word 'bought' will have two scores from spans B and C. We only
    # want to consider the score with "maximum context", which we define as
    # the *minimum* of its left and right context (the *sum* of left and
    # right context will always be the same, of course).
    #
    # In the example the maximum context for 'bought' would be span C since
    # it has 1 left context and 3 right context, while span B has 4 left context
    # and 0 right context.
    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span.start + doc_span.length - 1
        if position < doc_span.start:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span.start
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + \
            0.01 * doc_span.length
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index


#
def create_model(bert_config, is_training, input_ids, input_mask, segment_ids, input_span_mask,
                 use_one_hot_embeddings):
    """Creates a classification model."""
    model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings)

    final_hidden = model.get_sequence_output()

    final_hidden_shape = modeling.get_shape_list(final_hidden, expected_rank=3)
    batch_size = final_hidden_shape[0]
    seq_length = final_hidden_shape[1]
    hidden_size = final_hidden_shape[2]

    output_weights = tf.get_variable(
        "cls/squad/output_weights", [2, hidden_size],
        initializer=tf.truncated_normal_initializer(stddev=0.02))

    output_bias = tf.get_variable(
        "cls/squad/output_bias", [2], initializer=tf.zeros_initializer())

    final_hidden_matrix = tf.reshape(final_hidden,
                                     [batch_size * seq_length, hidden_size])
    logits = tf.matmul(final_hidden_matrix, output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)

    logits = tf.reshape(logits, [batch_size, seq_length, 2])
    logits = tf.transpose(logits, [2, 0, 1])

    unstacked_logits = tf.unstack(logits, axis=0)

    (start_logits, end_logits) = (unstacked_logits[0], unstacked_logits[1])

    # apply output mask
    adder = (1.0 - tf.cast(input_span_mask, tf.float32)) * -10000.0
    start_logits += adder
    end_logits += adder

    return (start_logits, end_logits)


#
def model_fn_builder(bert_config, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings):
    """Returns `model_fn` closure for TPUEstimator."""

    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""

        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info("  name = %s, shape = %s" %
                            (name, features[name].shape))

        unique_ids = features["unique_ids"]
        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        input_span_mask = features["input_span_mask"]

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        (start_logits, end_logits) = create_model(
            bert_config=bert_config,
            is_training=is_training,
            input_ids=input_ids,
            input_mask=input_mask,
            segment_ids=segment_ids,
            input_span_mask=input_span_mask,
            use_one_hot_embeddings=use_one_hot_embeddings)

        tvars = tf.trainable_variables()

        initialized_variable_names = {}
        scaffold_fn = None
        if init_checkpoint:
            (assignment_map, initialized_variable_names
             ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
            if use_tpu:

                def tpu_scaffold():
                    tf.train.init_from_checkpoint(
                        init_checkpoint, assignment_map)
                    return tf.train.Scaffold()

                scaffold_fn = tpu_scaffold
            else:
                tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        tf.logging.info("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            tf.logging.info("  name = %s, shape = %s%s",
                            var.name, var.shape, init_string)

        output_spec = None
        if mode == tf.estimator.ModeKeys.TRAIN:
            seq_length = modeling.get_shape_list(input_ids)[1]

            def compute_loss(logits, positions):
                on_hot_pos = tf.one_hot(
                    positions, depth=seq_length, dtype=tf.float32)
                log_probs = tf.nn.log_softmax(logits, axis=-1)
                loss = - \
                    tf.reduce_mean(tf.reduce_sum(
                        on_hot_pos * log_probs, axis=-1))
                return loss

            start_positions = features["start_positions"]
            end_positions = features["end_positions"]

            start_loss = compute_loss(start_logits, start_positions)
            end_loss = compute_loss(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

            train_op = optimization.create_optimizer(
                total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)

            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                train_op=train_op,
                scaffold_fn=scaffold_fn)
        elif mode == tf.estimator.ModeKeys.PREDICT:
            start_logits = tf.nn.log_softmax(start_logits, axis=-1)
            end_logits = tf.nn.log_softmax(end_logits, axis=-1)

            predictions = {
                "unique_ids": unique_ids,
                "start_logits": start_logits,
                "end_logits": end_logits,
            }
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode, predictions=predictions, scaffold_fn=scaffold_fn)
        else:
            raise ValueError(
                "Only TRAIN and PREDICT modes are supported: %s" % (mode))

        return output_spec

    return model_fn


def input_fn_builder(input_file, seq_length, is_training, drop_remainder):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    name_to_features = {
        "unique_ids": tf.FixedLenFeature([], tf.int64),
        "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
        "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "input_span_mask": tf.FixedLenFeature([seq_length], tf.int64),
    }

    if is_training:
        name_to_features["start_positions"] = tf.FixedLenFeature([], tf.int64)
        name_to_features["end_positions"] = tf.FixedLenFeature([], tf.int64)

    def _decode_record(record, name_to_features):
        """Decodes a record to a TensorFlow example."""
        example = tf.parse_single_example(record, name_to_features)

        # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
        # So cast all int64 to int32.
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            example[name] = t

        return example

    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]

        # For training, we want a lot of parallel reading and shuffling.
        # For eval, we want no shuffling and parallel reading doesn't matter.
        d = tf.data.TFRecordDataset(input_file)
        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)

        d = d.apply(
            tf.contrib.data.map_and_batch(
                lambda record: _decode_record(record, name_to_features),
                batch_size=batch_size,
                drop_remainder=drop_remainder))

        return d

    return input_fn


RawResult = collections.namedtuple("RawResult",
                                   ["unique_id", "start_logits", "end_logits"])


def write_predictions(all_examples, all_features, all_results, n_best_size,
                      max_answer_length, do_lower_case, output_prediction_file,
                      output_nbest_file):
    """Write final predictions to the json file and log-odds of null if needed."""
    tf.logging.info("Writing predictions to: %s" % (output_prediction_file))
    tf.logging.info("Writing nbest to: %s" % (output_nbest_file))

    example_index_to_features = collections.defaultdict(list)
    for feature in all_features:
        example_index_to_features[feature.example_index].append(feature)

    unique_id_to_result = {}
    for result in all_results:
        unique_id_to_result[result.unique_id] = result

    _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
        "PrelimPrediction",
        ["feature_index", "start_index", "end_index", "start_logit", "end_logit"])

    all_predictions = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()

    for (example_index, example) in enumerate(all_examples):
        features = example_index_to_features[example_index]
        prelim_predictions = []

        for (feature_index, feature) in enumerate(features):  # multi-trunk
            result = unique_id_to_result[feature.unique_id]
            start_indexes = _get_best_indexes(result.start_logits, n_best_size)
            end_indexes = _get_best_indexes(result.end_logits, n_best_size)
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # We could hypothetically create invalid predictions, e.g., predict
                    # that the start of the span is in the question. We throw out all
                    # invalid predictions.
                    if start_index >= len(feature.tokens):
                        continue
                    if end_index >= len(feature.tokens):
                        continue
                    if start_index not in feature.token_to_orig_map:
                        continue
                    if end_index not in feature.token_to_orig_map:
                        continue
                    if not feature.token_is_max_context.get(start_index, False):
                        continue
                    if end_index < start_index:
                        continue
                    length = end_index - start_index + 1
                    if length > max_answer_length:
                        continue
                    prelim_predictions.append(
                        _PrelimPrediction(
                            feature_index=feature_index,
                            start_index=start_index,
                            end_index=end_index,
                            start_logit=result.start_logits[start_index],
                            end_logit=result.end_logits[end_index]))

        prelim_predictions = sorted(
            prelim_predictions,
            key=lambda x: (x.start_logit + x.end_logit),
            reverse=True)

        _NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
            "NbestPrediction", ["text", "start_logit", "end_logit", "start_index", "end_index"])

        seen_predictions = {}
        nbest = []
        for pred in prelim_predictions:
            if len(nbest) >= n_best_size:
                break
            feature = features[pred.feature_index]
            if pred.start_index > 0:  # this is a non-null prediction
                tok_tokens = feature.tokens[pred.start_index:(
                    pred.end_index + 1)]
                orig_doc_start = feature.token_to_orig_map[pred.start_index]
                orig_doc_end = feature.token_to_orig_map[pred.end_index]
                orig_tokens = example.doc_tokens[orig_doc_start:(
                    orig_doc_end + 1)]
                tok_text = " ".join(tok_tokens)

                # De-tokenize WordPieces that have been split off.
                tok_text = tok_text.replace(" ##", "")
                tok_text = tok_text.replace("##", "")

                # Clean whitespace
                tok_text = tok_text.strip()
                tok_text = " ".join(tok_text.split())
                orig_text = " ".join(orig_tokens)

                final_text = get_final_text(tok_text, orig_text, do_lower_case)
                final_text = final_text.replace(' ', '')
                if final_text in seen_predictions:
                    continue

                seen_predictions[final_text] = True
            else:
                final_text = ""
                seen_predictions[final_text] = True

            nbest.append(
                _NbestPrediction(
                    text=final_text,
                    start_logit=pred.start_logit,
                    end_logit=pred.end_logit,
                    start_index=pred.start_index,
                    end_index=pred.end_index))

        # In very rare edge cases we could have no valid predictions. So we
        # just create a nonce prediction in this case to avoid failure.
        if not nbest:
            nbest.append(
                _NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0))

        assert len(nbest) >= 1

        total_scores = []
        best_non_null_entry = None
        for entry in nbest:
            total_scores.append(entry.start_logit + entry.end_logit)
            if not best_non_null_entry:
                if entry.text:
                    best_non_null_entry = entry

        probs = _compute_softmax(total_scores)

        nbest_json = []
        for (i, entry) in enumerate(nbest):
            output = collections.OrderedDict()
            output["text"] = entry.text
            output["probability"] = probs[i]
            output["start_logit"] = entry.start_logit
            output["end_logit"] = entry.end_logit
            output["start_index"] = entry.start_index
            output["end_index"] = entry.end_index
            nbest_json.append(output)

        assert len(nbest_json) >= 1

        all_predictions[example.qas_id] = best_non_null_entry.text
        all_nbest_json[example.qas_id] = nbest_json

    with tf.gfile.GFile(output_prediction_file, "w") as writer:
        writer.write(json.dumps(all_predictions,
                                indent=4, ensure_ascii=False) + "\n")

    with tf.gfile.GFile(output_nbest_file, "w") as writer:
        writer.write(json.dumps(all_nbest_json, indent=4) + "\n")


def get_final_text(pred_text, orig_text, do_lower_case):
    """Project the tokenized prediction back to the original text."""

    # When we created the data, we kept track of the alignment between original
    # (whitespace tokenized) tokens and our WordPiece tokenized tokens. So
    # now `orig_text` contains the span of our original text corresponding to the
    # span that we predicted.
    #
    # However, `orig_text` may contain extra characters that we don't want in
    # our prediction.
    #
    # For example, let's say:
    #   pred_text = steve smith
    #   orig_text = Steve Smith's
    #
    # We don't want to return `orig_text` because it contains the extra "'s".
    #
    # We don't want to return `pred_text` because it's already been normalized
    # (the SQuAD eval script also does punctuation stripping/lower casing but
    # our tokenizer does additional normalization like stripping accent
    # characters).
    #
    # What we really want to return is "Steve Smith".
    #
    # Therefore, we have to apply a semi-complicated alignment heruistic between
    # `pred_text` and `orig_text` to get a character-to-charcter alignment. This
    # can fail in certain cases in which case we just return `orig_text`.

    def _strip_spaces(text):
        ns_chars = []
        ns_to_s_map = collections.OrderedDict()
        for (i, c) in enumerate(text):
            if c == " ":
                continue
            ns_to_s_map[len(ns_chars)] = i
            ns_chars.append(c)
        ns_text = "".join(ns_chars)
        return (ns_text, ns_to_s_map)

    # We first tokenize `orig_text`, strip whitespace from the result
    # and `pred_text`, and check if they are the same length. If they are
    # NOT the same length, the heuristic has failed. If they are the same
    # length, we assume the characters are one-to-one aligned.
    tokenizer = tokenization.BasicTokenizer(do_lower_case=do_lower_case)

    tok_text = " ".join(tokenizer.tokenize(orig_text))

    start_position = tok_text.find(pred_text)
    if start_position == -1:
        if FLAGS.verbose_logging:
            tf.logging.info(
                "Unable to find text: '%s' in '%s'" % (pred_text, orig_text))
        return orig_text
    end_position = start_position + len(pred_text) - 1

    (orig_ns_text, orig_ns_to_s_map) = _strip_spaces(orig_text)
    (tok_ns_text, tok_ns_to_s_map) = _strip_spaces(tok_text)

    if len(orig_ns_text) != len(tok_ns_text):
        if FLAGS.verbose_logging:
            tf.logging.info("Length not equal after stripping spaces: '%s' vs '%s'",
                            orig_ns_text, tok_ns_text)
        return orig_text

    # We then project the characters in `pred_text` back to `orig_text` using
    # the character-to-character alignment.
    tok_s_to_ns_map = {}
    for (i, tok_index) in six.iteritems(tok_ns_to_s_map):
        tok_s_to_ns_map[tok_index] = i

    orig_start_position = None
    if start_position in tok_s_to_ns_map:
        ns_start_position = tok_s_to_ns_map[start_position]
        if ns_start_position in orig_ns_to_s_map:
            orig_start_position = orig_ns_to_s_map[ns_start_position]

    if orig_start_position is None:
        if FLAGS.verbose_logging:
            tf.logging.info("Couldn't map start position")
        return orig_text

    orig_end_position = None
    if end_position in tok_s_to_ns_map:
        ns_end_position = tok_s_to_ns_map[end_position]
        if ns_end_position in orig_ns_to_s_map:
            orig_end_position = orig_ns_to_s_map[ns_end_position]

    if orig_end_position is None:
        if FLAGS.verbose_logging:
            tf.logging.info("Couldn't map end position")
        return orig_text

    output_text = orig_text[orig_start_position:(orig_end_position + 1)]
    return output_text


def _get_best_indexes(logits, n_best_size):
    """Get the n-best logits from a list."""
    index_and_score = sorted(
        enumerate(logits), key=lambda x: x[1], reverse=True)

    best_indexes = []
    for i in range(len(index_and_score)):
        if i >= n_best_size:
            break
        best_indexes.append(index_and_score[i][0])
    return best_indexes


def _compute_softmax(scores):
    """Compute softmax probability over raw logits."""
    if not scores:
        return []

    max_score = None
    for score in scores:
        if max_score is None or score > max_score:
            max_score = score

    exp_scores = []
    total_sum = 0.0
    for score in scores:
        x = math.exp(score - max_score)
        exp_scores.append(x)
        total_sum += x

    probs = []
    for score in exp_scores:
        probs.append(score / total_sum)
    return probs


class FeatureWriter(object):
    """Writes InputFeature to TF example file."""

    def __init__(self, filename, is_training):
        self.filename = filename
        self.is_training = is_training
        self.num_features = 0
        self._writer = tf.python_io.TFRecordWriter(filename)

    def process_feature(self, feature):
        """Write a InputFeature to the TFRecordWriter as a tf.train.Example."""
        self.num_features += 1

        def create_int_feature(values):
            feature = tf.train.Feature(
                int64_list=tf.train.Int64List(value=list(values)))
            return feature

        features = collections.OrderedDict()
        features["unique_ids"] = create_int_feature([feature.unique_id])
        features["input_ids"] = create_int_feature(feature.input_ids)
        features["input_mask"] = create_int_feature(feature.input_mask)
        features["segment_ids"] = create_int_feature(feature.segment_ids)
        features["input_span_mask"] = create_int_feature(
            feature.input_span_mask)

        if self.is_training:
            features["start_positions"] = create_int_feature(
                [feature.start_position])
            features["end_positions"] = create_int_feature(
                [feature.end_position])

        tf_example = tf.train.Example(
            features=tf.train.Features(feature=features))
        self._writer.write(tf_example.SerializeToString())

    def close(self):
        self._writer.close()


def validate_flags_or_throw(bert_config):
    """Validate the input FLAGS or throw an exception."""
    tokenization.validate_case_matches_checkpoint(FLAGS.do_lower_case,
                                                  FLAGS.init_checkpoint)

    if not FLAGS.do_train and not FLAGS.do_predict:
        raise ValueError(
            "At least one of `do_train` or `do_predict` must be True.")

    if FLAGS.do_train:
        if not FLAGS.train_file:
            raise ValueError(
                "If `do_train` is True, then `train_file` must be specified.")
    if FLAGS.do_predict:
        if not FLAGS.predict_file:
            raise ValueError(
                "If `do_predict` is True, then `predict_file` must be specified.")
    if FLAGS.do_eval:
        if not FLAGS.eval_file:
            raise ValueError(
                "If `do_eval` is True, then `eval_file` must be specified.")

    if FLAGS.max_seq_length > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length %d because the BERT model "
            "was only trained up to sequence length %d" %
            (FLAGS.max_seq_length, bert_config.max_position_embeddings))

    if FLAGS.max_seq_length <= FLAGS.max_query_length + 3:
        raise ValueError(
            "The max_seq_length (%d) must be greater than max_query_length "
            "(%d) + 3" % (FLAGS.max_seq_length, FLAGS.max_query_length))


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)

    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

    validate_flags_or_throw(bert_config)

    tf.gfile.MakeDirs(FLAGS.output_dir)

    tokenizer = tokenization.FullTokenizer(
        vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

    tpu_cluster_resolver = None
    if FLAGS.use_tpu and FLAGS.tpu_name:
        tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
            FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

    is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
    run_config = tf.contrib.tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        master=FLAGS.master,
        model_dir=FLAGS.output_dir,
        save_checkpoints_steps=FLAGS.save_checkpoints_steps,
        keep_checkpoint_max=2,
        tpu_config=tf.contrib.tpu.TPUConfig(
            iterations_per_loop=FLAGS.iterations_per_loop,
            num_shards=FLAGS.num_tpu_cores,
            per_host_input_for_training=is_per_host))

    train_examples = None
    num_train_steps = None
    num_warmup_steps = None
    if FLAGS.do_train:
        train_examples = read_squad_examples(
            input_file=FLAGS.train_file, is_training=True)
        # Pre-shuffle the input to avoid having to make a very large shuffle
        # buffer in in the `input_fn`.
        rng = random.Random(int(FLAGS.rand_seed))
        rng.shuffle(train_examples)

        # We write to a temporary file to avoid storing very large constant tensors
        # in memory.
        train_writer = FeatureWriter(
            filename=os.path.join(FLAGS.output_dir, "train.tf_record"),
            is_training=True)
        '''
        examples是一个list，其中每一个元素为一个SquadExample，list长度为问题的个数。
        举例example[0]为：
            qas_id: TRAIN_2022_QUERY_3, question_text: 为什么鄂图曼军队纷纷丢弃阵地向内陆退却？,
            doc_tokens: [英 法 两 国 投 入 战 役 共 计 62 艘 战 舰 以 及 大 量 辅 助 船 只 ， 并 指 定 英 国 皇 家 海 军 地 中 海 舰 队 司 令 萨 克 维 尔 · 卡 登 上 将 负 责 指 挥 这 次 战 役 。 舰 队 从 1915 年 2 月 19 日 起 开 始 炮 轰 达 达 尼 尔 海 峡 。 3 月 18 日 ， 16 艘 军 舰 企 图 强 行 闯 入 狭 窄 的 海 峡 通 道 ， 4 艘 军 舰 触 发 水 雷 ， 舰 只 慌 忙 撤 退 。 在 陆 地 上 ， 鄂 图 曼 军 队 因 遭 受 突 然 袭 击 ， 纷 纷 丢 弃 阵 地 向 内 陆 退 却 ， 英 国 突 击 部 队 在 没 有 遇 到 抵 抗 的 情 况 下 冲 上 海 岸 。 德 国 军 事 顾 问 奥 托 · 冯 · 桑 德 斯 洞 悉 对 方 计 划 登 陆 加 里 波 利 ， 火 速 调 动 军 队 至 战 区 。 鄂 图 曼 军 队 掘 壕 坚 守 ， 依 据 半 岛 复 杂 的 地 形 建 立 了 强 大 的 防 御 体 系 ， 又 在 该 地 集 结 炮 兵 部 队 。 在 英 法 军 队 准 备 扩 大 战 果 时 ， 隐 蔽 在 阵 地 中 的 鄂 图 曼 士 兵 一 起 开 火 ， 把 正 在 攀 登 悬 崖 的 英 军 打 的 措 手 不 及 。 3 月 3 日 ， 联 军 的 首 轮 登 陆 行 动 宣 告 失 败 ， 卡 登 上 将 因 伤 送 回 英 国 。],
            start_position: 128, 
            end_position: 133
        '''
        convert_examples_to_features(
            examples=train_examples,
            tokenizer=tokenizer,
            max_seq_length=FLAGS.max_seq_length,
            doc_stride=FLAGS.doc_stride,
            max_query_length=FLAGS.max_query_length,
            is_training=True,
            output_fn=train_writer.process_feature)
        train_writer.close()
        num_features = train_writer.num_features
        train_examples_len = len(train_examples)
        del train_examples

        num_train_steps = int(
            num_features / FLAGS.train_batch_size * FLAGS.num_train_epochs)
        num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

        tf.logging.info("***** Running training *****")
        tf.logging.info("  Num orig examples = %d", train_examples_len)
        tf.logging.info("  Num split examples = %d", num_features)
        tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
        tf.logging.info("  Num steps = %d", num_train_steps)

    model_fn = model_fn_builder(
        bert_config=bert_config,
        init_checkpoint=FLAGS.init_checkpoint,
        learning_rate=FLAGS.learning_rate,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        use_tpu=FLAGS.use_tpu,
        use_one_hot_embeddings=FLAGS.use_tpu)

    # If TPU is not available, this will fall back to normal Estimator on CPU
    # or GPU.
    estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu=FLAGS.use_tpu,
        model_fn=model_fn,
        config=run_config,
        train_batch_size=FLAGS.train_batch_size,
        predict_batch_size=FLAGS.predict_batch_size)

    # do training
    if FLAGS.do_train:
        train_writer_filename = train_writer.filename

        train_input_fn = input_fn_builder(
            input_file=train_writer_filename,
            seq_length=FLAGS.max_seq_length,
            is_training=True,
            drop_remainder=True)
        estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)

    # do predictions
    if FLAGS.do_predict:
        eval_examples = read_squad_examples(
            input_file=FLAGS.predict_file, is_training=False)

        eval_writer = FeatureWriter(
            filename=os.path.join(FLAGS.output_dir, "predict.tf_record"),
            is_training=False)
        eval_features = []

        def append_feature(feature):
            eval_features.append(feature)
            eval_writer.process_feature(feature)

        convert_examples_to_features(
            examples=eval_examples,
            tokenizer=tokenizer,
            max_seq_length=FLAGS.max_seq_length,
            doc_stride=FLAGS.doc_stride,
            max_query_length=FLAGS.max_query_length,
            is_training=False,
            output_fn=append_feature)
        eval_writer.close()

        tf.logging.info("***** Running predictions *****")
        tf.logging.info("  Num orig examples = %d", len(eval_examples))
        tf.logging.info("  Num split examples = %d", len(eval_features))
        tf.logging.info("  Batch size = %d", FLAGS.predict_batch_size)

        all_results = []

        predict_input_fn = input_fn_builder(
            input_file=eval_writer.filename,
            seq_length=FLAGS.max_seq_length,
            is_training=False,
            drop_remainder=False)

        # If running eval on the TPU, you will need to specify the number of
        # steps.
        all_results = []
        for result in estimator.predict(
                predict_input_fn, yield_single_examples=True):
            if len(all_results) % 1000 == 0:
                tf.logging.info("Processing example: %d" % (len(all_results)))
            unique_id = int(result["unique_ids"])
            start_logits = [float(x) for x in result["start_logits"].flat]
            end_logits = [float(x) for x in result["end_logits"].flat]
            all_results.append(
                RawResult(
                    unique_id=unique_id,
                    start_logits=start_logits,
                    end_logits=end_logits))

        output_json_name = "dev_predictions.json"
        output_nbest_name = "dev_nbest_predictions.json"

        output_prediction_file = os.path.join(
            FLAGS.output_dir, output_json_name)
        output_nbest_file = os.path.join(FLAGS.output_dir, output_nbest_name)

        write_predictions(eval_examples, eval_features, all_results,
                          FLAGS.n_best_size, FLAGS.max_answer_length,
                          FLAGS.do_lower_case, output_prediction_file,
                          output_nbest_file)

    # do predictions
    if FLAGS.do_eval:
        eval_examples = read_squad_examples(
            input_file=FLAGS.eval_file, is_training=False)

        eval_writer = FeatureWriter(
            filename=os.path.join(FLAGS.output_dir, "eval.tf_record"),
            is_training=False)
        eval_features = []

        def append_feature(feature):
            eval_features.append(feature)
            eval_writer.process_feature(feature)

        convert_examples_to_features(
            examples=eval_examples,
            tokenizer=tokenizer,
            max_seq_length=FLAGS.max_seq_length,
            doc_stride=FLAGS.doc_stride,
            max_query_length=FLAGS.max_query_length,
            is_training=False,
            output_fn=append_feature)
        eval_writer.close()

        tf.logging.info("***** Running evals *****")
        tf.logging.info("  Num orig examples = %d", len(eval_examples))
        tf.logging.info("  Num split examples = %d", len(eval_features))
        tf.logging.info("  Batch size = %d", FLAGS.predict_batch_size)

        all_results = []

        predict_input_fn = input_fn_builder(
            input_file=eval_writer.filename,
            seq_length=FLAGS.max_seq_length,
            is_training=False,
            drop_remainder=False)

        # If running eval on the TPU, you will need to specify the number of
        # steps.
        all_results = []
        for result in estimator.predict(
                predict_input_fn, yield_single_examples=True):
            if len(all_results) % 1000 == 0:
                tf.logging.info("Processing example: %d" % (len(all_results)))
            unique_id = int(result["unique_ids"])
            start_logits = [float(x) for x in result["start_logits"].flat]
            end_logits = [float(x) for x in result["end_logits"].flat]
            all_results.append(
                RawResult(
                    unique_id=unique_id,
                    start_logits=start_logits,
                    end_logits=end_logits))

        output_json_name = "test_predictions.json"
        output_nbest_name = "test_nbest_predictions.json"

        output_prediction_file = os.path.join(
            FLAGS.output_dir, output_json_name)
        output_nbest_file = os.path.join(FLAGS.output_dir, output_nbest_name)

        write_predictions(eval_examples, eval_features, all_results,
                          FLAGS.n_best_size, FLAGS.max_answer_length,
                          FLAGS.do_lower_case, output_prediction_file,
                          output_nbest_file)


if __name__ == "__main__":
    flags.mark_flag_as_required("vocab_file")
    flags.mark_flag_as_required("bert_config_file")
    flags.mark_flag_as_required("output_dir")
    tf.app.run()
