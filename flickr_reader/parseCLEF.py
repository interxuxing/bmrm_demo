# -*- coding: utf-8 -*-
"""
    __author__ = 'xuxing'
"""

import re
import sys
reload(sys)
sys.setdefaultencoding("utf-8")

from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer as RT

try:
    import xml.etree.cElementTree as etree
except ImportError:
    import xml.etree.ElementTree as etree

try:
    import cPickle as pickle
except ImportError:
    import pickle

import enchant




"""
def fast_iter(context, func, path):
    # http://www.ibm.com/developerworks/xml/library/x-hiperfparse/
    # Author: Liza Daly
    for event, elem in context:
        func(elem, path)
        elem.clear()
        while elem.getprevious() is not None:
            del elem.getparent()[0]
    del context


def process_element(elem, path):
    print elem.xpath(path)

context = etree.iterparse(in_file, tag='item')
fast_iter(context, process_element, 'title')


"""


def remove_stopwords(src_str, stop_words):
    new_str = ' '.join([word for word in src_str.lower().split() \
        if word not in  stop_words])
    return new_str

def remove_punctutations(src_str):
    pattern = re.compile(r'[}|{@#%^&*_+-.?\'!,":;()/•|0-9]')
    new_str = ' '.join([pattern.sub('', word) for word in src_str.lower().split()])
    return new_str

def remove_symbol(src_str):
    # remove html symbol
    html_pattern = re.compile(r'<a.*?>')
    # new_str = ' '.join([html_pattern.sub('', word) for word in src_str.lower().split()])
    new_str = html_pattern.sub('', src_str)
    return new_str

def remove_non_english(src_str, dict):
    new_list = []
    for word in src_str.split():
        try:
            if dict.check(word) == True:
                new_list.append(word)
        except Exception, ex:
            continue
    return ' '.join(new_list)


def divide2word(src_str):
    tokenizer = RT(r'\w+')
    new_str = ' '.join(tokenizer.tokenize(src_str))
    return new_str

# some initial parameters
stopwords_en = stopwords.words('english')
enchant_en = enchant.Dict('en_US')

# 需要选取的类型
# branch_list = ['title', 'description', 'comment']
branch_list = ['title', 'description', 'tag']
# 源文件
in_file = open('photosCLEF.xml', 'r')
# 以iterparse的方式每次读取一个photo的实例，并将接受start, end事件
context = etree.iterparse(in_file, events=('start','end'))

# 设置此bool变量， 保证每次读取字标签信息属于一个phto单元的
is_photo = False

out_file = open('photosCLEFtags.txt', 'w')
photo_count = 0

photo_usrid = []
photo_id = []
pickleCLEF = {}
for event, elem in context:
    tag = elem.tag
    value = elem.text
    if value:
        value = value.encode('utf-8').strip()

    if event == 'start':
        if tag == 'photo':
            is_photo = True
            # print 'get a new photo, id=' + elem.attrib.get('id')
            instance_content = []
    if event == 'end':
        if tag == 'photo' and is_photo == True:
            is_photo = False
            photo_count += 1
            current_id = elem.attrib.get('id')
            photo_id.append(current_id)
            if photo_count % 100 == 0:
                print 'parse ' + str(photo_count) + '-th photo, id=' + \
                      current_id + ' finished!'

            # 如果得到读photo标签的end事件， 则清空当前elem的缓存
            elem.clear()

            photo_content = ' '.join(instance_content) # 把所有子标签中提取的信息合并到一个string中

            #接下来， 剔除photo_content中的stopwords, 标点符号
            newstr = remove_symbol(photo_content)
            newstr = remove_punctutations(newstr)
            newstr = remove_stopwords(newstr, stopwords_en)

            #写入文件
            # finalstr = ' '.join([word for word in newstr.split() if enchant_en.check(word) == True])
            newstr = remove_non_english(newstr, enchant_en)
            if newstr is None:
                newstr = ' '
            out_file.writelines([newstr+'\n'])

        if tag == 'owner':
            current_usrid = elem.attrib.get('nsid')
            photo_usrid.append(current_usrid)
            if current_usrid is None:
                print '...get usrid error!'

        if tag in branch_list:
            try:
                if value is None:
                    value = ''
                # print '......get a content, tag=' + tag + ' ,text='+ value
                instance_content.append(value)
            except Exception, ex:
                print Exception, ':', ex

if len(photo_id) == len(photo_usrid):
    pickleCLEF[0] = photo_id
    pickleCLEF[1] = photo_usrid
    pickle.dump(pickleCLEF, open('pickleCLEF.pick', "wb"), True)

    in_file.close()
    out_file.close()
    print 'finish parse total %d photos ' % photo_count
else:
    print '... the number of photo_id and photo_usrid is not identical!'

