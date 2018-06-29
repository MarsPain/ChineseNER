import os
# encoding=utf8
import codecs
import pickle
import itertools
from collections import OrderedDict
import tensorflow as tf
import numpy as np
from model import Model
from loader import load_sentences, update_tag_scheme
from loader import char_mapping, tag_mapping
from loader import augment_with_pretrained, prepare_dataset
from utils import get_logger, make_path, clean, create_model, save_model
from utils import print_config, save_config, load_config, test_ner
from data_utils import load_word2vec, create_input, input_from_line, BatchManager

flags = tf.app.flags
#若要训练则将clean和train设置为True
flags.DEFINE_boolean("clean",       True,      "clean train folder")
flags.DEFINE_boolean("train",       True,      "Wither train the model")
#若要通过自己的输入进行验证则将clean和train均设置为False
# flags.DEFINE_boolean("clean",       False,      "clean train folder")
# flags.DEFINE_boolean("train",       False,      "Wither train the model")
# configurations for the model
#seg_dim为分割特征的维度，分割特征即为词向量，对应的char_dim为自向量的维度，分别对应于
# 英语文本中的词向量和字符向量
flags.DEFINE_integer("seg_dim",     0,         "Embedding size for segmentation, 0 if not used")
flags.DEFINE_integer("char_dim",    100,        "Embedding size for characters")
flags.DEFINE_integer("lstm_dim",    100,        "Num of hidden units in LSTM")
flags.DEFINE_string("tag_schema",   "iobes",    "tagging schema iobes or iob")

# configurations for training
flags.DEFINE_float("clip",          5,          "Gradient clip")
flags.DEFINE_float("dropout",       0.7,        "Dropout rate")
flags.DEFINE_float("batch_size",    30,         "batch size")
flags.DEFINE_float("lr",            0.001,      "Initial learning rate")
flags.DEFINE_string("optimizer",    "adam",     "Optimizer for training")
flags.DEFINE_boolean("pre_emb",     True,       "Wither use pre-trained embedding")
flags.DEFINE_boolean("zeros",       False,      "Wither replace digits with zero")
flags.DEFINE_boolean("lower",       True,       "Wither lower case")

flags.DEFINE_integer("max_epoch",   10,        "maximum training epochs")
flags.DEFINE_integer("steps_check", 100,        "steps per checkpoint")
flags.DEFINE_string("ckpt_path",    "ckpt",      "Path to save model")
flags.DEFINE_string("summary_path", "summary",      "Path to store summaries")
flags.DEFINE_string("log_file",     "train.log",    "File for log")
flags.DEFINE_string("map_file",     "maps.pkl",     "file for maps")
flags.DEFINE_string("vocab_file",   "vocab.json",   "File for vocab")
flags.DEFINE_string("config_file",  "config_file",  "File for config")
flags.DEFINE_string("script",       "conlleval",    "evaluation script")
flags.DEFINE_string("result_path",  "result",       "Path for results")
flags.DEFINE_string("emb_file",     "wiki_100.utf8", "Path for pre_trained embedding")
#用原数据集进行训练和测试
# flags.DEFINE_string("train_file",   os.path.join("data", "example.train"),  "Path for train data")
# flags.DEFINE_string("dev_file",     os.path.join("data", "example.dev"),    "Path for dev data")
# flags.DEFINE_string("test_file",    os.path.join("data", "example.test"),   "Path for test data")
#用中医证候数据集example_medicine_three进行训练和测试
# flags.DEFINE_string("train_file",   os.path.join("data", "example_medicine_three.train"),  "Path for train data")
# flags.DEFINE_string("dev_file",     os.path.join("data", "example_medicine_three.dev"),    "Path for dev data")
# flags.DEFINE_string("test_file",    os.path.join("data", "example_medicine_three.test"),   "Path for test data")
#用中医证候数据集example_medicine_all进行训练和测试
flags.DEFINE_string("train_file",   os.path.join("data", "example_medicine_all.train"),  "Path for train data")
flags.DEFINE_string("dev_file",     os.path.join("data", "example_medicine_all.dev"),    "Path for dev data")
flags.DEFINE_string("test_file",    os.path.join("data", "example_medicine_all.test"),   "Path for test data")


FLAGS = tf.app.flags.FLAGS
assert FLAGS.clip < 5.1, "gradient clip should't be too much"
assert 0 <= FLAGS.dropout < 1, "dropout rate between 0 and 1"
assert FLAGS.lr > 0, "learning rate must larger than zero"
assert FLAGS.optimizer in ["adam", "sgd", "adagrad"]


# config for the model
def config_model(char_to_id, tag_to_id):
    config = OrderedDict()
    config["num_chars"] = len(char_to_id)
    config["char_dim"] = FLAGS.char_dim
    config["num_tags"] = len(tag_to_id)
    config["seg_dim"] = FLAGS.seg_dim
    config["lstm_dim"] = FLAGS.lstm_dim
    config["batch_size"] = FLAGS.batch_size

    config["emb_file"] = FLAGS.emb_file
    config["clip"] = FLAGS.clip
    config["dropout_keep"] = 1.0 - FLAGS.dropout
    config["optimizer"] = FLAGS.optimizer
    config["lr"] = FLAGS.lr
    config["tag_schema"] = FLAGS.tag_schema
    config["pre_emb"] = FLAGS.pre_emb
    config["zeros"] = FLAGS.zeros
    config["lower"] = FLAGS.lower
    return config


def evaluate(sess, model, name, data, id_to_tag, logger):
    logger.info("evaluate:{}".format(name))
    ner_results = model.evaluate(sess, data, id_to_tag)
    eval_lines = test_ner(ner_results, FLAGS.result_path)
    for line in eval_lines:
        logger.info(line)
    f1 = float(eval_lines[1].strip().split()[-1])

    if name == "dev":
        best_test_f1 = model.best_dev_f1.eval()
        if f1 > best_test_f1:
            tf.assign(model.best_dev_f1, f1).eval()
            logger.info("new best dev f1 score:{:>.3f}".format(f1))
        return f1 > best_test_f1
    elif name == "test":
        best_test_f1 = model.best_test_f1.eval()
        if f1 > best_test_f1:
            tf.assign(model.best_test_f1, f1).eval()
            logger.info("new best test f1 score:{:>.3f}".format(f1))
        return f1 > best_test_f1


def train():
    # load data sets
    #加载数据集的sentence，并处理成列表，每个sentence中的词及相应的标签也处理成列表
    train_sentences = load_sentences(FLAGS.train_file, FLAGS.lower, FLAGS.zeros)
    dev_sentences = load_sentences(FLAGS.dev_file, FLAGS.lower, FLAGS.zeros)
    test_sentences = load_sentences(FLAGS.test_file, FLAGS.lower, FLAGS.zeros)
    # print("dev_sentences:", dev_sentences)

    #原数据的标注模式与需要的标注模式不同时用update_tag_scheme对标注模式进行转换
    # Use selected tagging scheme (IOB / IOBES)
    # update_tag_scheme(train_sentences, FLAGS.tag_schema)
    # update_tag_scheme(test_sentences, FLAGS.tag_schema)

    # create maps if not exist
    #os.path.isfile查找是否存在该文件
    if not os.path.isfile(FLAGS.map_file):
        # create dictionary for word
        #使用预训练的词向量
        if FLAGS.pre_emb:
            #取返回列表的第一个值——字典
            dico_chars_train = char_mapping(train_sentences, FLAGS.lower)[0]
            #用预训练字典填充字典：将在预训练词向量文件中存在但是不存在于字典中的词加入字典
            dico_chars, char_to_id, id_to_char = augment_with_pretrained(
                dico_chars_train.copy(),
                FLAGS.emb_file,
                list(itertools.chain.from_iterable(
                    [[w[0] for w in s] for s in test_sentences])
                )
            )
        #若不使用预训练的词向量
        else:
            #直接返回字典与两个映射
            _c, char_to_id, id_to_char = char_mapping(train_sentences, FLAGS.lower)

        # Create a dictionary and a mapping for tags
        _t, tag_to_id, id_to_tag = tag_mapping(train_sentences)
        print("tag_to_id", tag_to_id, len(tag_to_id))
        with open(FLAGS.map_file, "wb") as f:
            pickle.dump([char_to_id, id_to_char, tag_to_id, id_to_tag], f)
    else:
        with open(FLAGS.map_file, "rb") as f:
            char_to_id, id_to_char, tag_to_id, id_to_tag = pickle.load(f)

    # prepare data, get a collection of list containing index
    #将sentence中的word和tag进行拆分和处理，得到字序列、字到ID的映射的序列（作为x_train）、
    #Segment_feature（还没理解作用和意义，原理是用jieba对整个句子进行分词，然后处理得到的某种标签）
    #   （应该是作为辅助判断的标签、计算损失函数的一部分）（根据源代码注释，是分割特征）、
    #标签到ID的映射（对于IOBES的编码格式而言，有13种，比如E-ORG和E-PER）（作为y-train）
    train_data = prepare_dataset(
        train_sentences, char_to_id, tag_to_id, FLAGS.lower
    )
    dev_data = prepare_dataset(
        dev_sentences, char_to_id, tag_to_id, FLAGS.lower
    )
    test_data = prepare_dataset(
        test_sentences, char_to_id, tag_to_id, FLAGS.lower
    )
    # print("dev_data:", dev_data, len(dev_data))
    print("%i / %i / %i sentences in train / dev / test." % (
        len(train_data), 0, len(test_data)))

    train_manager = BatchManager(train_data, int(FLAGS.batch_size))
    dev_manager = BatchManager(dev_data, 100)
    test_manager = BatchManager(test_data, 100)
    # make path for store log and model if not exist
    make_path(FLAGS)
    if os.path.isfile(FLAGS.config_file):
        config = load_config(FLAGS.config_file)
    else:
        config = config_model(char_to_id, tag_to_id)
        save_config(config, FLAGS.config_file)
    make_path(FLAGS)

    log_path = os.path.join("log", FLAGS.log_file)
    logger = get_logger(log_path)
    print_config(config, logger)

    # limit GPU memory
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    steps_per_epoch = train_manager.len_data
    with tf.Session(config=tf_config) as sess:
        model = create_model(sess, Model, FLAGS.ckpt_path, load_word2vec, config, id_to_char, logger)
        logger.info("start training")
        loss = []
        for i in range(FLAGS.max_epoch):
            for batch in train_manager.iter_batch(shuffle=True):
                step, batch_loss = model.run_step(sess, True, batch)
                loss.append(batch_loss)
                if step % FLAGS.steps_check == 0:
                    iteration = step // steps_per_epoch + 1
                    logger.info("iteration:{} step:{}/{}, "
                                "NER loss:{:>9.6f}".format(
                        iteration, step%steps_per_epoch, steps_per_epoch, np.mean(loss)))
                    loss = []

            best = evaluate(sess, model, "dev", dev_manager, id_to_tag, logger)
            if best:
                save_model(sess, model, FLAGS.ckpt_path, logger)
            evaluate(sess, model, "test", test_manager, id_to_tag, logger)


def evaluate_line():
    config = load_config(FLAGS.config_file)
    logger = get_logger(FLAGS.log_file)
    # limit GPU memory
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    with open(FLAGS.map_file, "rb") as f:
        char_to_id, id_to_char, tag_to_id, id_to_tag = pickle.load(f)
    with tf.Session(config=tf_config) as sess:
        model = create_model(sess, Model, FLAGS.ckpt_path, load_word2vec, config, id_to_char, logger)
        while True:
            # try:
            #     line = input("请输入测试句子:")
            #     result = model.evaluate_line(sess, input_from_line(line, char_to_id), id_to_tag)
            #     print(result)
            # except Exception as e:
            #     logger.info(e)

                line = input("请输入测试句子:")
                result = model.evaluate_line(sess, input_from_line(line, char_to_id), id_to_tag)
                print(result)


def main(_):

    if FLAGS.train:
        if FLAGS.clean:
            clean(FLAGS)
        train()
    else:
        evaluate_line()


if __name__ == "__main__":
    tf.app.run(main)



