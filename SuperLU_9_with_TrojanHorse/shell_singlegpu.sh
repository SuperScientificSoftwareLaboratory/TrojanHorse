#!/bin/bash

if [ -z "$MTX_A100" ]; then
    echo "erro: MTX_A100 not set."
    exit 1
fi

LIST_FILE="$HOME/16/200_mtx.csv"

if [ ! -f "$LIST_FILE" ]; then
    echo "erro: no file  $LIST_FILE"
    exit 1
fi

while IFS= read -r mtx; do
    # 忽略空行
    [ -z "$mtx" ] && continue

    # 查找 .mtx 文件
    path=$(find "$MTX_A100" -type f -name "${mtx}.mtx" | head -n 1)

    if [ -z "$path" ]; then
        echo "# 未找到矩阵: $mtx"
    else
        mpirun -n 1 ./build_gpu/EXAMPLE/pddrive -r 1 -c 1 "$path"
    fi
done < "$LIST_FILE"
