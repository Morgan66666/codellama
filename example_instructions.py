# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from typing import Optional
import fire
from llama import Llama
import os
from bs4 import BeautifulSoup

def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.2,
    top_p: float = 0.95,
    max_seq_len: int = 512,
    max_batch_size: int = 1,
    max_gen_len: Optional[int] = None,
):
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    folder_path = 'problem_descriptions'
    all_instructions = []
    index = []

    for filename in os.listdir(folder_path):
        if filename.endswith('.html'):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                html_content = f.read()
            soup = BeautifulSoup(html_content, 'html.parser')
            single_file_instructions = []
            single_file_instructions.append({"role": "metadata", "filename": filename.removesuffix('.html')})  # 记录文件名
            single_file_instructions.append({"role": "system", "content": "Provide answers in Python"})
            question_text = ''
            # for p in soup.find_all('p'):
            #     question_text += p.text
            # single_file_instructions.append({"role": "user", "content": question_text})
            # all_instructions.append(single_file_instructions)
            entire_text = soup.get_text()
            position = entire_text.find("Output")

            if position != -1:
                question_text = entire_text[:position]

            single_file_instructions.append({"role": "user", "content": question_text})
            all_instructions.append(single_file_instructions)


    batch_size = 4
    batched_instructions = [all_instructions[i:i + batch_size] for i in range(0, len(all_instructions), batch_size)]
    output_folder = "output_folder"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)


    # for idx, instruction in enumerate(batched_instructions):
        # results = generator.chat_completion(
        #     [instruction],
        #     max_gen_len=max_gen_len,
        #     temperature=temperature,
        #     top_p=top_p,
        # )
        #
        # result = results[0]
        # output_filename = f"{output_folder}/{idx}.txt"
        #
        # with open(output_filename, 'w', encoding='utf-8') as f:
        #     for msg in instruction:
        #         f.write(f"{msg['role'].capitalize()}: {msg['content']}\n")
        #     f.write(f"> {result['generation']['role'].capitalize()}: {result['generation']['content']}\n")

    for batch_idx, batch in enumerate(batched_instructions):
        results = generator.chat_completion(
            [inst[1:] for inst in batch],  # 注意这里跳过了每个指令列表的第一个元素（包含文件名的字典）
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
        )
        for idx, (instruction, result) in enumerate(zip(batch, results)):
            output_filename = f"{output_folder}/{instruction[0]['filename']}.txt"  #从第一个字典中获取文件名
            with open(output_filename, 'w', encoding='utf-8') as f:
                for msg in instruction[1:]:  # 注意这里从第二个元素开始
                    f.write(f"{msg['role'].capitalize()}: {msg['content']}\n")
                f.write(f"> {result['generation']['role'].capitalize()}: {result['generation']['content']}\n")


if __name__ == "__main__":
    fire.Fire(main)
