#!/usr/bin/env python3
"""
从 StreamVGGT 模型文件中提取 DINOv2 权重

使用方法:
    python extract_dinov2_weights.py --input ckpts/model.pt --output ckpts/dinov2_stream_vggt.pt
"""

import torch
import argparse
from collections import OrderedDict


def extract_dinov2_weights(input_path, output_path):
    """
    从模型检查点中提取 DINOv2 权重
    
    Args:
        input_path: 输入的模型文件路径
        output_path: 输出的 DINOv2 权重文件路径
    """
    print(f"正在加载模型文件: {input_path}")
    
    # 加载模型检查点
    checkpoint = torch.load(input_path, map_location='cpu', weights_only=False)
    
    # 如果 checkpoint 是字典且包含 'state_dict' 键，则使用 state_dict
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif isinstance(checkpoint, (dict, OrderedDict)):
        state_dict = checkpoint
    else:
        raise ValueError(f"不支持的检查点格式: {type(checkpoint)}")
    
    print(f"检查点中共有 {len(state_dict)} 个权重键")
    
    # 提取所有以 'aggregator.patch_embed.' 开头的权重
    dinov2_weights = OrderedDict()
    prefix = 'aggregator.patch_embed.'
    
    for key, value in state_dict.items():
        if key.startswith(prefix):
            # 移除 'aggregator.patch_embed.' 前缀，得到标准的 DINOv2 权重名称
            new_key = key[len(prefix):]
            dinov2_weights[new_key] = value
            print(f"提取: {key} -> {new_key}")
    
    print(f"\n成功提取 {len(dinov2_weights)} 个 DINOv2 权重")
    
    # 显示提取的权重键（前10个）
    print("\n提取的权重键示例（前10个）:")
    for i, key in enumerate(list(dinov2_weights.keys())[:10]):
        print(f"  {i+1}. {key}")
    
    # 保存提取的权重
    print(f"\n正在保存到: {output_path}")
    torch.save(dinov2_weights, output_path)
    print("✓ 提取完成！")
    
    return dinov2_weights


def main():
    parser = argparse.ArgumentParser(description='从 StreamVGGT 模型文件中提取 DINOv2 权重')
    parser.add_argument(
        '--input',
        type=str,
        default='ckpts/model.pt',
        help='输入的模型文件路径 (默认: ckpts/model.pt)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='ckpts/dinov2_weights.pt',
        help='输出的 DINOv2 权重文件路径 (默认: ckpts/dinov2_weights.pt)'
    )
    
    args = parser.parse_args()
    
    extract_dinov2_weights(args.input, args.output)


if __name__ == '__main__':
    main()

