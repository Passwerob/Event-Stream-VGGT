import os
from huggingface_hub import hf_hub_download

print("正在查找或下载 StreamVGGT 权重...")
try:
    # 尝试从 HuggingFace Hub 下载（如果已存在缓存中则直接返回路径）
    checkpoint_path = hf_hub_download(
        repo_id="lch01/StreamVGGT",
        filename="checkpoints.pth",
        revision="main"
    )
    print(f"权重文件位于: {checkpoint_path}")

    # 在当前目录下创建 ckpt 文件夹并建立软链接
    target_dir = "ckpt"
    os.makedirs(target_dir, exist_ok=True)
    target_path = os.path.join(target_dir, "checkpoints.pth")
    
    if os.path.exists(target_path):
        print(f"目标路径已存在: {target_path}")
    else:
        os.symlink(checkpoint_path, target_path)
        print(f"已创建软链接: {target_path} -> {checkpoint_path}")
        
    print("\n现在您可以在命令中使用: --checkpoint ckpt/checkpoints.pth")

except Exception as e:
    print(f"\n发生错误: {e}")
    print("请确保您有网络连接以访问 Hugging Face。")