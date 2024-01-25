#coding=utf-8
import os
import json,re
from collections import Counter
import tiktoken

import chardet

def detect_file_encoding(file_path):
    # 打开文件，读取一部分内容进行编码检测
    with open(file_path, 'rb') as file:
        # 读取一定数量的字节用于检测
        raw_data = file.read(10000)
        result = chardet.detect(raw_data)
        encoding = result['encoding']
        return encoding


def get_first_subdirname(directory):
    subdirectories = []
    for name in os.listdir(directory):
        if os.path.isdir(os.path.join(directory, name)):
            subdirectories.append(name)
    subdirectories.sort()
    return subdirectories

def most_pre_ele(lst,num=1):
    counter = Counter(lst)
    pre_lis = counter.most_common(num)
    pre_ele = counter.most_common(num)[0][0]
    return pre_lis,pre_ele

def most_common_element(lst):
    counter = Counter(lst)
    return counter.most_common(1)[0][0]

def write_to_json(file_path,data):
    with open(file_path, 'a+',encoding='utf-8') as f_jsonl:
        json.dump(data, f_jsonl,ensure_ascii=False)
        f_jsonl.write('\n')
    # print("数据已写入 JSON 文件.")

def append_to_json(file_path,data):

    with open(file_path, 'a') as file:
        json.dump(data, file)
    # print("数据已追加到 JSON 文件.")


def get_filelist(directory):
    file_list = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            if not file.startswith('.') and os.path.isfile(file_path):
                file_list.append(file_path)
    file_list.sort()
    return file_list


class Split2blocks():

    @staticmethod
    def find_def_class_indices(file_content_list, start_patterns):
        """
        Find indices of lines in a file content list where the line starts with specified patterns.
        """
        indices = [0]
        for i, line in enumerate(file_content_list):
            # Check if the line starts with any of the specified patterns
            if any(line.startswith(pattern) for pattern in start_patterns):
                indices.append(i)
        return indices

    @staticmethod
    def split_into_blocks(file_content_list, indices):
        """
        Split the file content into blocks based on the provided indices.
        """
        blocks = []
        for i in range(len(indices)):
            # Start of the current block
            start = indices[i]
            # End of the current block (start of the next block or end of file)
            end = indices[i + 1] if i + 1 < len(indices) else len(file_content_list)
            # Extract the block
            block = file_content_list[start:end]
            blocks.append(block)
        return blocks

    @staticmethod
    def combine_blocks(blocks, max_length):
        """
        Combine smaller blocks into larger blocks, ensuring the combined block length does not exceed max_length.
        """
        combined_blocks = []
        current_block = []

        for block in blocks:
            # Calculate the length of the current block and the next block
            current_block_length = sum(get_tokens(line) for line in current_block)
            next_block_length = sum(get_tokens(line) for line in block)

            # Check if adding the next block exceeds the max_length
            if current_block_length + next_block_length > max_length:
                # If it does, start a new block
                if current_block:
                    combined_blocks.append(current_block)
                    current_block = block
            else:
                # If it doesn't, add the block to the current block
                current_block.extend(block)

        # Add the last block if it's not empty
        if current_block:
            combined_blocks.append(current_block)

        return combined_blocks

    @staticmethod
    def spilit_bloks_py(blocks,token_lth):

        sum_tokens = sum(get_tokens(''.join(line)) for line in blocks)
        if sum_tokens < token_lth:
            flattened_list = [element for sublist in blocks for element in sublist]
            return [flattened_list]
        else:
            sum_tokens_pre1 = sum(get_tokens(''.join(line)) for line in blocks[:1])
            max_tokens = token_lth - sum_tokens_pre1
            pre_blocks = blocks[:1]
            flattened_list = [element for sublist in pre_blocks for element in sublist]
            merge_blocks = Split2blocks.combine_blocks(blocks[1:], max_tokens)
            merge_blocks.insert(0, flattened_list)
            return merge_blocks

    @staticmethod
    def spilit_bloks_fun(blocks,token_lth):
        sum_tokens_pre2 = sum(get_tokens(''.join(line)) for line in blocks[:2])
        sum_tokens_pre3 = sum(get_tokens(''.join(line)) for line in blocks[:3])
        if sum_tokens_pre3 < token_lth:
            max_tokens = token_lth - sum_tokens_pre3
            pre_blocks3 = blocks[:3]
            flattened_list = [element for sublist in pre_blocks3 for element in sublist]
            merge_blocks = Split2blocks.combine_blocks(blocks[3:], max_tokens)
            merge_blocks.insert(0, flattened_list)
        else:
            max_tokens = token_lth - sum_tokens_pre2
            pre_blocks2 = blocks[:2]
            flattened_list = [element for sublist in pre_blocks2 for element in sublist]
            merge_blocks = Split2blocks.combine_blocks(blocks[2:], max_tokens)
            merge_blocks.insert(0, flattened_list)
        return merge_blocks

    @staticmethod
    def get_py_blocks(file_path,token_lth):
        with open(file_path, 'r', encoding='utf-8') as f:
            file_content_list = f.readlines()
        # print(file_content_list)

        # Find indices
        indices = Split2blocks.find_def_class_indices(file_content_list, ['def ', 'class '])
        blocks = Split2blocks.split_into_blocks(file_content_list, indices)
        merge_blocks = Split2blocks.spilit_bloks_py(blocks,token_lth)
        """
        第一层循环 是类和函数
        第二层忙 是类内部和函数内部
        """
        all_blocks = []
        for block in merge_blocks[:]:
            content = ''.join(block)
            tokens = get_tokens(content)
            print(f"---block {tokens}  {block[:1]} ---")
            # print(content) 这里理论上应该去掉第一层内容
            if tokens > token_lth:
                sub_indices = Split2blocks.find_def_class_indices(block, ['    assert', '    def ', '    @', '    """'])
                sub_blocks = Split2blocks.split_into_blocks(block, sub_indices)
                sub_merge_blocks = Split2blocks.spilit_bloks_fun(sub_blocks,token_lth)
                sum_tokens = 0
                for sub_block in sub_merge_blocks[:]:
                    sub_content = ''.join(sub_block)
                    tokens = get_tokens(sub_content)
                    sum_tokens += tokens
                    print(f"---   sub_block {tokens} {sub_block[:1]} ---")
                # print(f'---   sum_tokens {sum_tokens}')
                all_blocks.extend(sub_merge_blocks)
            else:
                all_blocks.append(block)

        return all_blocks

def get_filenames(directory,format=None):
    file_list = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            if not file.startswith('.') and os.path.isfile(file_path):
                if format:
                    if file.endswith(format):
                        file_list.append(file)
                else:
                    file_list.append(file)
    file_list.sort()
    return file_list

def is_file_path(string):
    # 检查字符串是否是一个存在的文件路径
    return os.path.isfile(string)
def get_tokens(content, model="gpt-3.5-turbo"):
    if is_file_path(content):
        try:
            with open(content, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            print(f'content: {content}')
            print(f'error: {e}')
            return None
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(content))

def get_file_size(file_path):
    try:
        # 获取文件大小
        size = os.path.getsize(file_path)
        return size
    except OSError as e:
        # 错误处理，例如文件不存在或路径错误
        print(f"错误: {e}")
        return None

def get_filelisform(directory,format=None):
    file_list = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            # file_endoing = detect_file_encoding(file_path)
            if not file.startswith('.') and os.path.isfile(file_path):
                if format:
                    if file.endswith(format):
                        file_list.append(file_path)
                else:
                    file_list.append(file_path)
    file_list.sort()
    return file_list


def get_filename(directory,format=None):
    file_list = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            if not file.startswith('.') and os.path.isfile(file_path):
                if format:
                    if file.endswith(format):
                        file_list.append([file,file_path])
                else:
                    file_list.append([file, file_path])
    file_list.sort()
    return file_list

#创建类
def create_class(class_name, **kwargs):
    class_dict = {}

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    class_dict['__init__'] = __init__

    new_class = type(class_name, (object,), class_dict)

    # 构建类的定义字符串
    class_definition = ""
    if kwargs:
        class_definition += f"class {class_name}(object):\n"
        class_definition += f"    def __init__(self, {', '.join(kwargs.keys())}):\n"
        for key in kwargs.keys():
            class_definition += f"        self.{key} = {key}\n"
    else:
        class_definition += f"class {class_name}(object):\n"
        class_definition += f"    def __init__(self):\n"
        class_definition += f"        pass\n"

    # 构建实例化代码
    instantiation_code = ""
    if kwargs:
        # instantiation_code += f"{class_name} = create_class('{class_name}', {', '.join([f'{key}={value!r}' for key, value in kwargs.items()])})\n"
        instantiation_code += f"obj = {class_name}({', '.join([f'{key}={value!r}' for key, value in kwargs.items()])})\n"
    else:
        # instantiation_code += f"{class_name} = create_class('{class_name}')\n"
        instantiation_code += f"obj = {class_name}()\n"

    # 构建输出内容
    output = class_definition + "\nif __name__ == '__main__':\n    " + instantiation_code.replace("\n", "\n    ")

    return output

#预训练大模型
def move_to_subdir(directory):
    
    file_list = get_filelist(directory)


    lis = ['pr.merged','pr.open','pr.closed','issue.open','issue.closed']
    for item in lis:
        new_dir = os.path.join(directory, item)
        if not os.path.exists(new_dir):
            os.makedirs(new_dir)
    
    for old_file_pth in file_list:
        file_name = old_file_pth.split('/')[-1]
        name = '.'.join(file_name.split('.')[2:-1])
        new_file_pth = os.path.join(directory, f'{name}/{file_name}')
        os.rename(old_file_pth,new_file_pth)
    


    print('移动完成')



def read_tolist(file,encoding='utf-8'):
    with open(file,'r',encoding=encoding) as f:
        lines = f.readlines()
        lines = [item.strip() for item in lines if item.strip()]
    return lines

#获取当前时间
def get_time():
    import time
    time_now = time.strftime('%m.%d_%H.%M',time.localtime(time.time()))
    return time_now


#把一个列表写入到文本文件
def write_to_filekuai(file,lines,num=0):
    if num==1:
        file = 'txt_'+get_time()+'___'+file
    with open(file,'w',encoding='utf-8') as f:
        for line in lines:
            f.write(str(line)+'\n')

#把一个字典写入到文本文件
def write_file2dic(file,lines):
    with open(file,'w',encoding='utf-8') as f:
        f.write(str(lines)+'\n')


#获取一级子目录
def get_first_subdir(directory):
    subdirectories = []
    for name in os.listdir(directory):
        if os.path.isdir(os.path.join(directory, name)):
            subdirectories.append(os.path.join(directory, name))
    subdirectories.sort()
    return subdirectories

#获取子目录
def get_subdir(directory):
    subdirectories = []
    for dirpath, dirnames, files in os.walk(directory):
        for dirname in dirnames:
            subdirectories.append(os.path.join(dirpath, dirname))
    subdirectories.sort()
    return subdirectories

#把一个列表写入到文本文件
def write_to_filekuaiw(file,lines):
    with open(file,'w',encoding='utf-8') as f:
        for line in lines:
            if isinstance(line, str):
                f.write(line + '\n')
            elif isinstance(line, list):
                f.write('###'.join(line) + '\n')
            else:
                f.write(str(line) + '\n')

#把一个字符串写入到文本文件
def write_to_filea(file,line,num=1):
    if num==0:
        file = 'txt_'+get_time()+'___'+file
    with open(file,'a+',encoding='utf-8') as f:
        f.write(line+'\n')


#把一个字符串写入到文本文件
def write_to_filekong(file,line,num=1):
    if num==0:
        file = 'txt_'+get_time()+'___'+file
    with open(file,'w',encoding='utf-8') as f:
        f.write(line)


#把一个字符串写入到文本文件
def write_to_file(file,line,num=1):
    if num==0:
        file = 'txt_'+get_time()+'___'+file
    with open(file,'w',encoding='utf-8') as f:
        f.write(line+'\n')


#把一个字典当作文本文件写入
def read_bigone(file):
    with open(file,'r',encoding='utf-8') as f:
        line = f.readline()
        line = eval(line.strip())

    return line

#两个列表的交集
def jiaoji(list1,list2):
    list3 = list(set(list1).intersection(set(list2)))
    return list3

#给出文件名，自动创建文件并返回文件对象
def make_file(file,num=0):
    if num==0:
        file = 'py_'+get_time()+'___'+file
    with open(file,'w',encoding='utf-8') as f:
        
        f.write('#coding: utf-8'+'\n')
        f.write('from read import write_to_filekuai, read_tolist, write_to_filekuaiw, write_to_file, read_bigone'+'\n')
        
def make_pyfile(file, num=0):
    if num == 0:
        file = 'py_' + get_time() + '_' + file
    with open(file, 'w', encoding='utf-8') as f:
        f.write('#coding: utf-8' + '\n')
        f.write(
            'from read import write_to_filekuai, read_tolist, write_to_filekuaiw, write_to_file, read_bigone' + '\n')

def make_pyfile_xulie(file,root):
    file_list = get_filelist(root)
    config_py = os.path.join(root, 'config.py')
    lth = len(file_list)
    if config_py not in file_list:
        with open(config_py, 'w', encoding='utf-8') as f:
            f.write("""cfg = {"srt_dir_origin": "./data/ASS",}""")

        file = 'py' + str(lth).zfill(2)+'_' + file
        with open(file, 'w', encoding='utf-8') as f:
            f.write('#coding: utf-8' + '\n')
            f.write(
                'from read import write_to_filekuai, write_to_filekuaiw, write_to_file' + '\n'
                'from read import read_tolist, read_bigone' + '\n'
                'from read import get_filename, get_filelist, get_filelisform' + '\n'
                'from config import cfg' + '\n')
    else:
        file = 'py' + str(lth-1).zfill(2) + '_' + file
        with open(file, 'w', encoding='utf-8') as f:
            f.write('#coding: utf-8' + '\n')
            f.write(
                'from read import write_to_filekuai, write_to_filekuaiw, write_to_file' + '\n'
                                                                                          'from read import read_tolist, read_bigone' + '\n'
                                                                                                                                        'from read import get_filename, get_filelist, get_filelisform' + '\n')


def make_pyfile(file, num=0):
    if num == 0:
        file = 'py_' + get_time() + '_' + file
    with open(file, 'w', encoding='utf-8') as f:
        f.write('#coding: utf-8' + '\n')
        f.write(
            'from read import write_to_filekuai, write_to_filekuaiw, write_to_file' + '\n'
            'from read import read_tolist, read_bigone' + '\n'
            'from read import get_filename, get_filelist, get_filelisform' + '\n')

def make_class_file(file,classname,num=1):
    if num == 0:
        file = 'py_' + get_time() + '_' + file
    content = create_class(classname)
    if '.' not in file:
        file += '.py'
    with open(file, 'w', encoding='utf-8') as f:
        f.write('#coding: utf-8' + '\n')
        f.write('from read import write_to_filekuai, read_tolist, write_to_filekuaiw, write_to_file, read_bigone' + '\n')
        f.write('from read import get_filelist' +'\n')
        f.write('import openai' +'\n')
        f.write('import json,os' +'\n')
        f.write('from dotenv import load_dotenv' +'\n')
        f.write('load_dotenv()'+'\n')
        f.write("openai.api_key = os.environ.get('OPENAI_API_KEY')"+'\n')
        f.write('os.environ["http_proxy"] = os.environ.get("http_proxy")'+'\n')
        f.write('os.environ["https_proxy"] = os.environ.get("https_proxy")'+'\n'*4)
        f.write(content)
    with open('.env', 'w', encoding='utf-8') as f:
        f.write('OPENAI_API_KEY=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' + '\n')
        f.write('http_proxy=http://127.0.0.1:7890' + '\n')
        f.write('https_proxy=https://127.0.0.1:7890' + '\n')



#给出文件夹名，自动创建文件夹
def make_dir(root,real_dir):
    import os
    path = root +'\\' +real_dir
    os.mkdir(path)

