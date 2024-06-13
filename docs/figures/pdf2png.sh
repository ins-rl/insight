#!/bin/bash

# 设定包含 PDF 文件的文件夹路径
PDF_FOLDER="/home/l/Downloads/Pages/figures"

# 设定输出 PNG 图像的文件夹路径
OUTPUT_FOLDER="/home/l/Downloads/Pages/figures"

# 检查输出文件夹是否存在，如果不存在则创建
if [ ! -d "$OUTPUT_FOLDER" ]; then
  mkdir -p "$OUTPUT_FOLDER"
fi

# 遍历文件夹中的所有 PDF 文件
for pdf in "$PDF_FOLDER"/*.pdf; do
  # 获取不带路径的文件名
  filename=$(basename -- "$pdf")
  # 删除文件扩展名，准备作为输出文件的前缀
  base="${filename%.pdf}"
  # 使用 pdftoppm 将 PDF 转换为 PNG，输出到指定的文件夹
  pdftoppm -png "$pdf" "$OUTPUT_FOLDER/$base"
done

echo "转换完成。"

