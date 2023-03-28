import math
import os
import glob

folder_path = "data"  # 文件夹路径
file_extension = "*.txt"  # 文件扩展名


def calculate_entropy_single_word(text):
    # 建立一个字典，将每个字符与其出现次数相关联
    char_count = {}
    for char in text:
        if char in char_count:
            char_count[char] += 1
        else:
            char_count[char] = 1

    # 计算信息熵
    entropy = 0
    total_chars = len(text)
    for count in char_count.values():
        frequency = count / total_chars
        entropy += frequency * math.log2(frequency)

    return -entropy


if __name__ == '__main__':
    # 从文件中读取文本
    # 使用 glob.glob() 方法获取符合条件的所有文件路径
    file_paths = glob.glob(os.path.join(folder_path, file_extension))
    file_contents = {}
    # 循环遍历所有文件路径，并读取文件内容
    for path in file_paths:
        with open(path, "r", encoding='ansi') as file:
            file_contents = file.read()
    # 数据处理，清除无用数据
    with open('cn_stopwords.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()
        stop_words = [line.strip() for line in lines]
    remove_set = set(stop_words)
    result = filter(lambda x: x not in remove_set, file_contents)
    new_contents = ''.join(result)
    new_contents = new_contents.replace("\n", '')
    new_contents = new_contents.replace("\u3000", '')
    # 计算中文信息熵
    entropy = calculate_entropy_single_word(new_contents)
    print(f"该文本的中文信息熵为: {entropy}")
