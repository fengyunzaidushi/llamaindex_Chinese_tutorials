import json
import os

from read import get_filelist

root_dir = './index-pdf-zilv'
files = get_filelist(root_dir)
print(files)

for file in files[1:2]:
    print(file)
    # 读取json文件
    with open(file, 'r') as f:
        data = json.load(f)
        # print(data['docstore/data'])
    # 数据写入json文件
    new_file = file.split('/')[-1][:-5]
    with open(f'{root_dir}/{new_file}_1.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)