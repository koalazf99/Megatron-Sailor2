import os
import json
import gzip
import pandas as pd
from tqdm import tqdm

# 输入输出路径和其他配置
input_path = "/mbz/users/fan.zhou/llmosaic/megamath-translated-code/"
output_path = "/mbz/users/fan.zhou/tron/Megatron-Sailor2/jsonl_dataset/translated_code"
output_prefix = "translated_code"
text_key = "text"
per_file_num = 100000

# 如果输出文件夹不存在则创建
os.makedirs(output_path, exist_ok=True)

output_index = 0      # 输出文件编号
tmp_data = []         # 临时存储每次读取的记录（以字典形式存储）

# 遍历输入目录下的所有 parquet 文件
# for file in tqdm(os.listdir(input_path)):
for root, dirs, files in tqdm(os.walk(input_path)):
    for file in tqdm(files):
        if file.endswith(".parquet"):
            file_path = os.path.join(root, file)
            # 读取 parquet 文件
            df = pd.read_parquet(file_path)
            # 将 DataFrame 转换为字典列表
            records = df.to_dict(orient="records")
            
            # # 使用 gzip 读取 jsonl.gz 文件
            # records = []
            # with gzip.open(file_path, 'rt') as f:
            #     for line in f:
            #         data = json.loads(line)
            #         records.append({text_key: data[text_key]})
            
            
            # 追加到临时数据列表中
            for record in records:
                tmp_data.append({text_key: record[text_key]})
                # 当达到每个文件的行数限制时，写出到 jsonl 文件
                if len(tmp_data) >= per_file_num:
                    output_file = os.path.join(output_path, f"{output_prefix}_{output_index}.jsonl")
                    with open(output_file, "w", encoding="utf-8") as f_out:
                        for rec in tmp_data:
                            f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    output_index += 1
                    tmp_data = []  # 清空临时数据列表

# 将剩余的数据写出到最后一个文件中（如果有）
if tmp_data:
    output_file = os.path.join(output_path, f"{output_prefix}_{output_index}.jsonl")
    with open(output_file, "w", encoding="utf-8") as f_out:
        for rec in tmp_data:
            f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
