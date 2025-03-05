#!env python3
# coding: utf-8

import pandas as pd
import argparse

def merge_parquet_files(input_files, output_file):
    # 读取所有输入的parquet文件并合并
    dataframes = [pd.read_parquet(file) for file in input_files]
    merged_dataframe = pd.concat(dataframes)
    
    # 将合并后的数据写入输出文件
    merged_dataframe.to_parquet(output_file)

def main():
    parser = argparse.ArgumentParser(description='Merge multiple parquet files into one.')
    parser.add_argument('-i', '--inputs', type=str, required=True, help='Comma separated list of input parquet file paths.')
    parser.add_argument('-o', '--output', type=str, required=True, help='Output file path for the merged parquet file.')
    
    args = parser.parse_args()
    
    input_files = args.inputs.split(',')
    output_file = args.output
    
    merge_parquet_files(input_files, output_file)

if __name__ == '__main__':
    main()

## python data/merge-parquet.py -i data/ndr/parquet/train.parquet,data/kk/instruct/3ppl/train.parquet -o data/kkndr/ndrppl3/train.parquet
