#!/bin/bash

# 数字列表，替换 hy_837171 中的数字
#numbers=("837171" "837170" "837169" )  # 在此处填入需要下载的数字列表

# 定义基础路径
base_path='/alice/cern.ch/user/a/alihyperloop/jobs/0083/hy_'

start=837056
end=837168

# 遍历范围内的每个数字

## 遍历数字列表
# for number in "${numbers[@]}"; do
for number in $(seq "$start" "$end"); do
    # 生成当前数字对应的远程路径
    remote_path="${base_path}${number}/AOD/[0-9]*/"

    # 定义本地下载目录
    local_target_dir="/Users/zhengqingwang/alice/data/derived/Hypertriton_2body/LHC23_PbPb_fullTPC/pass4/ep3/${number}"

    # 创建本地下载目录
    mkdir -p "$local_target_dir"

    # 下载符合条件的 .root 文件
    echo "Downloading .root files from $remote_path to $local_target_dir ..."
    alien_cp "${remote_path}" -name ends_.root "file:${local_target_dir}/"
done

echo "All files downloaded successfully."

