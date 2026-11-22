#!/usr/bin/env python3
"""
将评估生成的多个 result_{split}_{dataset}_{epoch}.txt 文件合并为一个 CSV，
便于横向比较不同 epoch 的指标。

用法:
    python merge_results.py --result_dir <dir>
    python merge_results.py --result_dir <dir> --split test --dataset PVCP2
"""

import os
import sys
import json
import glob
import re
import csv
import argparse


def parse_result_filename(filename, pattern_prefix):
    """从文件名中提取测试文本条件和 checkpoint，'last' 排在最后"""
    match = re.search(
        rf'{re.escape(pattern_prefix)}_(?:(?P<text_condition>text-[^_]+)_)?(?P<checkpoint>checkpoint_(?P<epoch>last|\d+))\.txt$',
        filename,
    )
    if not match:
        legacy_match = re.search(rf'{re.escape(pattern_prefix)}_(?P<epoch>last|\d+)\.txt$', filename)
        if legacy_match:
            epoch_str = legacy_match.group("epoch")
            epoch_label = "last" if epoch_str == "last" else epoch_str
            sort_key = float('inf') if epoch_str == "last" else int(epoch_str)
            return sort_key, epoch_label, "legacy"
        return float('inf'), 'unknown', 'unknown'

    epoch_str = match.group("epoch")
    epoch_label = match.group("checkpoint")
    text_condition = match.group("text_condition") or "legacy"
    if epoch_str == 'last':
        return float('inf'), epoch_label, text_condition
    return int(epoch_str), epoch_label, text_condition


def parse_epoch(filename, pattern_prefix):
    sort_key, epoch_label, _ = parse_result_filename(filename, pattern_prefix)
    return sort_key, epoch_label


def merge_results_to_csv(result_dir, split='test', dataset='PVCP2',
                         output_csv_name=None, text_condition_prefix=None):
    """将目录下所有 result_{split}_{dataset}_{epoch}.txt 合并为一个 CSV"""
    pattern_prefix = f'result_{split}_{dataset}'
    glob_pattern = os.path.join(result_dir, f'{pattern_prefix}_*.txt')
    txt_files = glob.glob(glob_pattern)

    # 排除 avg 文件（如果存在）
    txt_files = [f for f in txt_files if '_avg_' not in os.path.basename(f)]

    if text_condition_prefix:
        txt_files = [
            f for f in txt_files
            if parse_result_filename(os.path.basename(f), pattern_prefix)[2].startswith(text_condition_prefix)
        ]

    if not txt_files:
        condition_msg = f" with text condition prefix '{text_condition_prefix}'" if text_condition_prefix else ""
        print(f"⚠️  未在 {result_dir} 中找到 {pattern_prefix}_*.txt 文件{condition_msg}")
        return None

    # 按 epoch 排序
    txt_files.sort(key=lambda f: parse_epoch(os.path.basename(f), pattern_prefix)[0])

    rows = []
    all_keys = []
    seen_keys = set()

    for fpath in txt_files:
        fname = os.path.basename(fpath)
        epoch_num, epoch_label, text_condition = parse_result_filename(fname, pattern_prefix)

        with open(fpath, 'r') as f:
            data = json.load(f)

        # 记录所有 key（保持顺序）
        for k in data:
            if k not in seen_keys:
                all_keys.append(k)
                seen_keys.add(k)

        rows.append({'epoch': epoch_label, 'text_condition': text_condition, **data})

    # 写入 CSV
    csv_filename = output_csv_name or f'{pattern_prefix}_all.csv'
    if not csv_filename.endswith('.csv'):
        csv_filename += '.csv'
    csv_path = os.path.join(result_dir, csv_filename)
    fieldnames = ['epoch', 'text_condition'] + all_keys

    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"✅ 已合并 {len(rows)} 个结果文件 -> {csv_path}")

    # 删除原始 txt 文件
    for fpath in txt_files:
        os.remove(fpath)
    print(f"🗑️  已删除 {len(txt_files)} 个原始 txt 文件")

    # 删除非 last 的 checkpoint pth 文件，只保留 checkpoint_last.pth
    pth_files = glob.glob(os.path.join(result_dir, 'checkpoint_*.pth'))
    deleted_pth = 0
    for pth in pth_files:
        if os.path.basename(pth) != 'checkpoint_last.pth':
            os.remove(pth)
            deleted_pth += 1
    if deleted_pth > 0:
        print(f"🗑️  已删除 {deleted_pth} 个 checkpoint pth 文件（保留 checkpoint_last.pth）")

    return csv_path


def main():
    parser = argparse.ArgumentParser(description='合并评估结果txt文件为CSV')
    parser.add_argument('--result_dir', type=str, required=True,
                        help='结果文件所在目录')
    parser.add_argument('--split', type=str, default='test',
                        help='数据集划分 (默认: test)')
    parser.add_argument('--dataset', type=str, default='PVCP2',
                        help='数据集版本 (默认: PVCP2)')
    parser.add_argument('--output_csv_name', type=str, default=None,
                        help='输出CSV文件名')
    parser.add_argument('--text_condition_prefix', type=str, default=None,
                        help='只合并指定测试文本条件前缀的结果，例如 text-full、text-mask 或 text-noisy')
    args = parser.parse_args()

    if not os.path.isdir(args.result_dir):
        print(f"❌ 目录不存在: {args.result_dir}")
        sys.exit(1)

    csv_path = merge_results_to_csv(
        args.result_dir,
        args.split,
        args.dataset,
        output_csv_name=args.output_csv_name,
        text_condition_prefix=args.text_condition_prefix,
    )
    if csv_path is None:
        sys.exit(1)


if __name__ == '__main__':
    main()
