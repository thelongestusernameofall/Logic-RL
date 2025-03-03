""" Preprocess dataset for ndr analysis task """

import os
from datasets import Dataset, load_dataset
from tqdm import tqdm
from verl.utils.hdfs_io import copy, makedirs
import argparse
import json

def get_system_prompt():
    sys_prompt = """你是一个乐于助人的助手。你首先在脑海中思考推理过程，然后为用户提供答案。推理过程和答案分别用<think> </think>和<answer> </answer>标签括起来。"""
    return sys_prompt

def get_task_prompt():
    task_prompt = """请从网络安全的角度分析下面的网络流量，回答结果必须为(攻击失败|攻击成功|非攻击)三者之一。"""
    return task_prompt
def make_prefix(query, template_type):
    quiz = query
    if template_type == 'base':
        prefix = f"""System:{get_system_prompt()}\n\nUser:{get_task_prompt()}\n{quiz}\n\nAssistant: <think>"""
    elif template_type == 'qwen-instruct':
        prefix = f"""<|im_start|>system\n{get_system_prompt()}\n<|im_end|>\n<|im_start|>user\n{get_task_prompt()}\n{quiz}\n<|im_end|>\n<|im_start|>assistant\n<think>"""
    else:
        raise Exception("Unexpected template_type: {template_type}")
    
    return prefix

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='data/ndr/parquet')
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument('--data_path', default='data/ndr/json/ndr_data-0221-attack_noattack.jsonl')
    parser.add_argument('--train_size', type=int, default=4690)
    parser.add_argument('--test_size', type=int, default=530)
    parser.add_argument('--val_size', type=int, default=10)
    parser.add_argument('--template_type', type=str, default='qwen-instruct')
    
    args = parser.parse_args()
    
    data_source = 'simon-ndr'
    TRAIN_SIZE = args.train_size
    TEST_SIZE = args.test_size
    VAL_SIZE = args.val_size

    # Load custom JSONL dataset
    def gen_from_jsonl(path):
        with open(path) as f:
            for line in f:
                yield json.loads(line)
    
    def gen_from_json(path):
        with open(path) as f:
            all_samples = json.loads(f.read())
            for sample in all_samples:
                yield sample
            

    """
    {"type": "chatml", "messages": [{"role": "user", "content": "{\"request\": \"POST /plus/weixin.php?signature=da39a3ee5e6b4b0d3255bfef95601890afd80709\\\\xc3...\", \"response\": \"HTTP/1.1 404 Not Found\\n...</html>\"}"}, {"role": "assistant", "content": "攻击"}], "source": "self-made"}
    """

    raw_dataset = Dataset.from_generator(gen_from_jsonl, gen_kwargs={'path': args.data_path})
    print(len(raw_dataset))

    assert len(raw_dataset) >= TRAIN_SIZE + TEST_SIZE + VAL_SIZE
    train_dataset = raw_dataset.select(range(TRAIN_SIZE))
    test_dataset = raw_dataset.select(range(TRAIN_SIZE, TRAIN_SIZE + TEST_SIZE))
    val_dataset = raw_dataset.select(range(TRAIN_SIZE + TEST_SIZE, TRAIN_SIZE + TEST_SIZE + VAL_SIZE))

    def make_map_fn(split):
        def process_fn(example, idx):
            assert example.get('messages')[0].get('role') == 'user'
            user_ctn = example.get('messages')[0].get('content')
            assert example.get('messages')[1].get('role') == 'assistant'
            reply_ctn = example.get('messages')[1].get('content')

            question = make_prefix(user_ctn, template_type=args.template_type)
            solution = reply_ctn
            data = {
                "data_source": data_source,
                "prompt": [{
                    "role": "user",
                    "content": question,
                }],
                "ability": "ndr",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": solution
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                }
            }
            return data
        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)
    val_dataset = val_dataset.map(function=make_map_fn('val'), with_indices=True)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    # Create local directory if not exists
    os.makedirs(os.path.expanduser(local_dir), exist_ok=True)

    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))
    val_dataset.to_parquet(os.path.join(local_dir, 'val.parquet'))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)