#!env python3
# coding utf-8


# 从json文件读入样本，并且修改样本的query，然后保存到新的jsonl文件中

#!env python3
# coding utf-8

import json
import re
import matplotlib.pyplot as plt

def process_content(content):
    # 使用正则表达式分割request和response
    match = re.search(r'(HTTP请求内容：)(.*?)(HTTP响应内容：)(.*)', content, re.DOTALL)
    if match:
        request = match.group(2).strip()
        response = match.group(4).strip()
        return {"request": request, "response": response}
    return None

def convert_json_to_jsonl(input_file, output_file):
    lengths = []  # 用于存储每个message["content"]的长度

    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    with open(output_file, 'w', encoding='utf-8') as f:
        for item in data:
            if item.get("type") == "chatml" and "messages" in item:
                messages = item["messages"]
                if messages[1]['content'] not in ['攻击', '非攻击','攻击成功', '攻击失败']:
                    print(f"unexpected tag = {messages[1]['content']}")
                    continue
                for message in messages:
                    if message["role"] == "user":
                        processed_content = process_content(message["content"])
                        if processed_content:
                            message["content"] = json.dumps(processed_content, ensure_ascii=False)
                            lengths.append(len(message["content"]))  # 记录长度
                if len(messages[0]['content']) > 6000:
                    print(f"Exceeds Length Limits: len = { len( messages[0]['content'] ) }")
                    continue
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    return lengths

def plot_length_distribution(lengths, output_image_file):
    plt.figure(figsize=(10, 6))
    plt.hist(lengths, bins=50, edgecolor='k')
    plt.title('Message Content Length Distribution')
    plt.xlabel('Length')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.savefig(output_image_file)  # 保存图像到本地文件
    plt.close()

if __name__ == "__main__":
    input_file = 'data/ndr/json/ndr-simon-sxz-r1-human.json'  # 输入文件路径
    output_file = 'data/ndr/json/ndr-simon-sxz-r1-human.jsonl'  # 输出文件路径
    output_image_file = 'data/ndr/json/ndr-simon-sxz-r1-human-length-distribution.png'  # 输出图像文件路径
    lengths = convert_json_to_jsonl(input_file, output_file)
    plot_length_distribution(lengths, output_image_file)
