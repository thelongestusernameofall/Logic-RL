import os
import pandas as pd
import json
import argparse
import math

def excel_to_json(directory, output_file):
    # 获取目录下所有的Excel文件
    excel_files = [f for f in os.listdir(directory) if f.endswith(('.xls', '.xlsx'))]
    print(f"processing files: {excel_files}")

    # 用于存储所有Excel文件转换后的JSON数据
    all_data = {}

    all_samples = []
    for excel_file in excel_files:
        file_path = os.path.join(directory, excel_file)
        # 读取Excel文件
        xls = pd.ExcelFile(file_path)
        
        # 用于存储每个Excel文件的所有sheet的数据
        # excel_data = {}
        
        for sheet_name in xls.sheet_names:
            # 读取每个sheet的数据
            df = pd.read_excel(file_path, sheet_name=sheet_name)
            # 将DataFrame转换为JSON格式
            # excel_data[sheet_name] = df.to_dict(orient='records')
            all_samples += df.to_dict(orient='records')
        
        # 将当前Excel文件的数据添加到总数据中
        # all_data[excel_file] = excel_data
    print(f"all_samples length: {len(all_samples)}")
    filtered_samples = [ele for ele in map(filter_and_convert_sample, all_samples) if ele is not None]
    
    # 将所有数据转换为JSON字符串
    json_data = json.dumps(filtered_samples, ensure_ascii=False, indent=4)
    
    # 将结果写入到一个JSON文件中
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(json_data)

    print(f"Excel文件已成功转换为JSON格式，并保存到 {output_file}。")

def filter_and_convert_sample(sample):
    ndr_input = sample['ndr输入']
    human_tag = sample['人工研判']
    same_tag = sample['结果对比']

    human_tag = "非攻击" if human_tag == '误报' else human_tag

    if not is_float_equal(same_tag,1.0):
        return None
    
    if human_tag not in ['攻击成功','攻击失败','非攻击']:
        print(f"unexpected human tag: {human_tag}")
        return None

    res = {
        "type": "chatml",
        "messages": [
            {
                "role": "user",
                "content": ndr_input
            },
            {
                "role": "assistant",
                "content": human_tag
            }
        ],
        "source": "simon-sxz-r1-match"
    }
    return res




def is_float_equal(va, vb, tol=1e-9):
    return math.isclose(vb, va, rel_tol=tol, abs_tol=tol)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="将目录中的所有Excel文件转换为JSON格式")
    parser.add_argument('-i', '--input', default='data/ndr/xlsx', required=False, help="输入Excel文件的目录")
    parser.add_argument('-o', '--output', default='data/ndr/json/ndr-simon-sxz-r1-human.json', required=False, help="输出JSON文件的路径")

    args = parser.parse_args()

    excel_to_json(args.input, args.output)
