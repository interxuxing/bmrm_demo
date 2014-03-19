# -*- coding: utf-8 -*-
"""
__author__ = 'xuxing'
"""

# import packages
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
from gensim import corpora, models, similarities

try:
    import cPickle as pickle
except ImportError:
    import pickle


from operator import itemgetter
def sort_dict_by_value(d, type):
    ''' proposed in PEP 265, using  the itemgetter
        返回的是list类型！
    '''
    if type == 'descend':
        return sorted(d.iteritems(), key=itemgetter(1), reverse=True)
    elif type == 'ascend':
        return sorted(d.iteritems(), key=itemgetter(1), reverse=False)
    else:
        print('type error! ascend or descend need!')
        return None

# initial configuration
src_filename = 'photosCLEFtags.txt'
src_file = open(src_filename, 'r')
dictionary_file = 'dictionaryCLEF.dict'
bow_filename = 'bowCLEF.txt'

if 0:
    # read file and get documents
    documents = src_file.read()
    texts = [[word for word in documents.lower().split()]]
    src_file.close()
    # remove words that appears only once
    all_tokens = sum(texts, [])
    tokens_once = set(word for word in set(all_tokens) if all_tokens.count(word) == 1)
    texts = [[word for word in text if word not in tokens_once] for text in texts]

    # now create dictionary for these documents
    dictionary = corpora.Dictionary(texts)
    dictionary.save(dictionary_file)
    print dictionary


if 1:
    pickleCLEF = pickle.load(open('pickleCLEF.pick', 'rb'))
    photo_id = pickleCLEF[0]
    photo_usrid = pickleCLEF[1]
    # genereate bow for each photo iteratively
    dictionary = corpora.Dictionary.load(dictionary_file)

    # 打开目标文件， 按照bmrm的textPASCAL.txt文件格式写入内容
    bow_file = open(bow_filename, 'w')

    # 先写入dictionary中word的个数
    total_words = len(dictionary)
    bow_file.writelines([str(total_words) + '\n'])

    # 然后按照 序号 word 的格式逐行写入每个word信息
    # first sort the dictionary by value ascending
    sorted_dict_list = sort_dict_by_value(dictionary, 'ascend')

    for item in sorted_dict_list:
        bow_file.writelines(['%d %s\n' % (item[0], item[1])])

    # 最后逐行写入每个图片的bow, 格式为'photo_id (long), usrid (string), word出现个数(int)， 每个word出现统计'
    src_file = open(src_filename, 'r')
    for index in range(len(photo_id)):
        line_word_count = 0
        line_word = src_file.readline().strip('\n')
        # photo_id usrid
        syntax_header = '%ld %s' % (long(photo_id[index]), photo_usrid[index])
        # line = line.strip('\n')
        if line_word == '':
            bow_file.writelines([syntax_header + ' '+ str(line_word_count) + '\n'])
        else:
            bow_line = []
            vect = dictionary.doc2bow(line_word.lower().split())
            for item in range(len(vect)):
                temp = '%d:%d' % (vect[item][0], vect[item][1])
                bow_line.append(temp)
            line_word_count = len(vect)
            bow_file.writelines([syntax_header + ' ' + str(line_word_count) + ' ' + \
                                 ' '.join(bow_line) + '\n'])

    bow_file.close()
    src_file.close()

# generate the bow file for bmrm file format

