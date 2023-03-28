import math
import os
import glob
import jieba
from collections import Counter

folder_path = "data"  # 文件夹路径
file_extension = "*.txt"  # 文件扩展名


# 基于字的一元模型
def calEntropySingleWord(text):
    # 建立一个字典，将每个字符与其出现次数相关联
    char_count = {}
    total_chars = 0
    for words in text:
        for char in words:
            total_chars += 1
            if char in char_count:
                char_count[char] += 1
            else:
                char_count[char] = 1
    # 计算信息熵
    entropy = 0
    for count in char_count.values():
        frequency = count / total_chars
        entropy += frequency * math.log2(frequency)

    print("对中文基于单个字的一元模型的信息熵为:", -entropy)


# 基于词的一元模型
def calEntropySingleTerm(text):
    split_words = []
    for term in text:
        split_words += list(jieba.cut(term))
    words_count = len(split_words)
    unary_words_dict = dict(Counter(split_words))
    entropy = []
    for _, value in unary_words_dict.items():
        entropy.append(-(value / words_count) * math.log(value / words_count, 2))
    print("对中文基于单个词的一元模型的信息熵为:", sum(entropy))


def calEntropySingleTerm_2(text):
    all_words = []
    binary_words_dict = {}
    for word in text:
        split_words = list(jieba.cut(word))
        for i in range(len(split_words) - 1):
            binary_words_dict[(split_words[i], split_words[i + 1])] = binary_words_dict.get(
                (split_words[i], split_words[i + 1]), 0) + 1
        all_words += split_words
    words_count = len(all_words)
    unary_words_dict = dict(Counter(all_words))
    binary_words_count = sum([value for _, value in binary_words_dict.items()])
    entropy = []
    for key, value in binary_words_dict.items():
        joint_probability_xy = value / binary_words_count  # 计算联合概率p(x,y)
        conditional_probability_x_y = joint_probability_xy / (unary_words_dict[key[0]]/words_count)  # 计算条件概率p(x|y)
        entropy.append(-joint_probability_xy * math.log(conditional_probability_x_y, 2))  # 计算二元模型的信息熵
    print("对中文基于单个词的二元模型的信息熵为:", sum(entropy))


def calEntropySingleTerm_3(text):
    all_words = []
    binary_words_dict = {}
    ternary_words_dict = {}
    for term in text:
        split_words = list(jieba.cut(term))
        for i in range(len(split_words) - 1):
            binary_words_dict[(split_words[i], split_words[i + 1])] = binary_words_dict.get(
                (split_words[i], split_words[i + 1]), 0) + 1
        for i in range(len(split_words) - 2):
            ternary_words_dict[((split_words[i], split_words[i + 1]), split_words[i + 2])] = ternary_words_dict.get(
                ((split_words[i], split_words[i + 1]), split_words[i + 2]), 0) + 1
        all_words += split_words
    binary_words_count = sum([value for _, value in binary_words_dict.items()])
    ternary_words_count = sum([value for _, value in ternary_words_dict.items()])

    entropy = []
    for key, value in ternary_words_dict.items():
        joint_probability_xyz = value / ternary_words_count
        conditional_probability_x_yz = joint_probability_xyz / (binary_words_dict[key[0]] / binary_words_count)
        entropy.append(-joint_probability_xyz * math.log(conditional_probability_x_yz, 2))
    print("对中文基于单个词的三元模型的信息熵为:", sum(entropy))


if __name__ == '__main__':
    # 从文件中读取文本
    # 数据处理，清除无用数据
    with open('delete.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()
        delete = [line.strip() for line in lines]
    remove_set = set(delete)
    # 使用 glob.glob() 方法获取符合条件的所有文件路径
    file_paths = glob.glob(os.path.join(folder_path, file_extension))
    words_list = []
    count = 0
    # 循环遍历所有文件路径，并读取文件内容
    for path in file_paths:
        with open(path, "r", encoding='ansi') as file:
            temp = file.read()
            result = filter(lambda x: x not in remove_set, temp)
            new_temp = ''.join(result)
            new_temp = new_temp.replace("\n", '')
            new_temp = new_temp.replace("\u3000", '')
            new_temp = new_temp.replace(" ", '')
            words_list.append(new_temp)
            count += len(new_temp)

    # 文字分割
    with open('cn_stopwords.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()
        stop_words = [line.strip() for line in lines]

    contents_after_stop = []
    count = 0
    for text in words_list:
        new_words = []
        split_words = list(jieba.cut(text))
        for word in split_words:
            if word not in stop_words:
                new_words.append(word)
        count += len(''.join(map(str, new_words)))
        contents_after_stop.append(''.join(map(str, new_words)))

    # 计算中文信息熵
    calEntropySingleWord(contents_after_stop)
    calEntropySingleTerm(contents_after_stop)
    calEntropySingleTerm_2(contents_after_stop)
    calEntropySingleTerm_3(contents_after_stop)


