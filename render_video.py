import os
import cv2
import argparse

def sort_img_by_number(img_name):
    """
    核心排序函数：提取文件名中的纯数字，用于按数字大小升序排序
    支持的文件名格式：1.png / 001.png / frame_5.png / img_0020.png 等任意含数字的png命名
    """
    # 提取文件名中的所有数字字符并拼接成数字
    num_str = ''.join(filter(str.isdigit, img_name))
    return int(num_str) if num_str else 0

def png_sequence_to_video(img_dir, output_video_path, fps=24):
    """
    将指定文件夹的PNG序列帧合成视频
    :param img_dir: 存放PNG图片的文件夹路径
    :param output_video_path: 输出视频的完整路径（如 ./output/result.mp4）
    :param fps: 视频帧率，默认24帧/秒，可自定义
    """
    # 1. 校验输入文件夹是否存在
    if not os.path.isdir(img_dir):
        print(f"❌ 错误：指定的文件夹 {img_dir} 不存在！")
        return

    # 2. 读取文件夹内所有PNG文件，过滤非png后缀的文件（忽略大小写）
    img_files = [f for f in os.listdir(img_dir) 
                 if f.lower().endswith('.png') and os.path.isfile(os.path.join(img_dir, f))]
    
    # 3. 判断是否有PNG文件
    if not img_files:
        print(f"⚠️ 警告：文件夹 {img_dir} 内未找到任何PNG图片！")
        return

    # 4. 核心排序：按文件名中的数字从小到大排序（重点！）
    img_files_sorted = sorted(img_files, key=sort_img_by_number)
    print(f"✅ 共找到 {len(img_files_sorted)} 张PNG图片，已按数字升序排列")
    print(f"📄 排序后的前5张图片：{img_files_sorted[:5]}")

    # 5. 拼接每张图片的完整路径
    img_paths = [os.path.join(img_dir, f) for f in img_files_sorted]

    # 6. 读取第一张图片，获取视频的宽高尺寸（所有PNG建议尺寸一致）
    first_img = cv2.imread(img_paths[0])
    if first_img is None:
        print(f"❌ 错误：无法读取第一张图片 {img_paths[0]}，请检查文件是否损坏")
        return
    height, width = first_img.shape[:2]
    print(f"📐 视频尺寸：宽={width}px，高={height}px，帧率={fps}fps")

    # 7. 设置视频编码器和写入器（兼容Windows/Mac/Linux，无损编码，mp4格式通用）
    # mp4编码：mp4v 兼容性最强，输出文件体积适中，画质无损
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # 8. 逐帧写入图片合成视频
    for idx, img_path in enumerate(img_paths, start=1):
        img = cv2.imread(img_path)
        if img is not None:
            video_writer.write(img)
            # 打印进度（每10帧打印一次，避免刷屏）
            if idx % 10 == 0 or idx == len(img_paths):
                print(f"📥 正在合成：{idx}/{len(img_paths)} 帧")
        else:
            print(f"⚠️ 警告：跳过损坏的图片 -> {img_path}")

    # 9. 释放资源，完成合成
    video_writer.release()
    cv2.destroyAllWindows()
    print(f"\n🎉 视频合成完成！输出路径：{output_video_path}")

if __name__ == "__main__":
    # 命令行参数解析，方便直接运行+传参，新手友好
    parser = argparse.ArgumentParser(description='将指定文件夹的PNG序列帧按数字排序合成视频')
    parser.add_argument('--img_dir', type=str, required=True, help='存放PNG图片的文件夹绝对路径/相对路径')
    parser.add_argument('--output', type=str, default='./output_video.mp4', help='输出视频的路径+文件名，默认当前目录output_video.mp4')
    parser.add_argument('--fps', type=int, default=24, help='视频帧率，默认24，常用值：24/30/60')
    
    args = parser.parse_args()
    # 执行合成函数
    png_sequence_to_video(args.img_dir, args.output, args.fps)