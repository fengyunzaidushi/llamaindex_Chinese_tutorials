import json

# 读取json文件
with open('./index-zh/docstore.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

print(data.keys())
for k in data.keys():
    res = data[k]
    # 写入json文件
    k_name = k.split('/')[-1]
    with open(f'./json_analysis/{k_name}.json', 'w', encoding='utf-8') as f:
        json.dump(res, f, ensure_ascii=False, indent=4)
    print(f'first key is {k}')
    second_keys = data[k].keys()
    for second_key in second_keys:
        print(f'{k} | {second_key}')
        third_keys = data[k][second_key].keys()
        for third_key in third_keys:
            print(f'{k} | {second_key}|{third_key}')
            res = data[k][second_key][third_key]
            # print(type(res))
            if isinstance(res, dict):
                forth_keys = res.keys()
                for forth_key in forth_keys:
                    print(f'{k} | {second_key}  |  {third_key}  | {forth_key}')
                    print(f'text:{data[k][second_key][third_key][forth_key]}')
            # forth_keys = data[k][second_key][third_key].keys()
            # for forth_key in forth_keys:
            #     print(f'the k first is {k} second key is {second_key} third_key is {third_key} forth_key is {forth_key}')
            #     print(data[k][second_key][third_key][forth_key])

    print('------------------')
