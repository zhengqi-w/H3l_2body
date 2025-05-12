import ROOT

# 从 input.txt 中读取输入文件路径
with open("/Users/zhengqingwang/alice/data/derived/Hypertriton_2body/LHC23_PbPb_fullTPC/pass4/tmp.txt", "r") as f:
    root_files = [line.strip() for line in f if line.strip()]

# 定义输出文件
output_file = ROOT.TFile("/Users/zhengqingwang/alice/data/derived/Hypertriton_2body/LHC23_PbPb_fullTPC/pass4/merged_AO2D.root", "RECREATE")

# 使用 TChain 来合并所有文件中的树
chain = ROOT.TChain()  # 树名会在遍历文件时设置

# 初始化变量以存储第一个目录和树的名称
first_directory_name = None
tree_name = None

# 遍历所有文件，动态获取目录名和树名
for file_path in root_files:
    
    input_file = ROOT.TFile.Open(file_path)
    if not input_file:
        print(f"Failed to open {file_path}")
        continue
    # else:
        # print(f"Opened {file_path}")

    # 获取第一个 TDirectory 名称和树名称
    for key in input_file.GetListOfKeys():
        obj = key.ReadObj()
        if obj.InheritsFrom("TDirectory"):
            if obj.GetName() == "parentFiles":
                continue
            print(f"found directory: {obj.GetName()}")
            if not first_directory_name:
                first_directory_name = obj.GetName()  # 记录第一个目录名
            input_file.cd(obj.GetName())
            
            # 获取第一个树的名称
            for tree_key in input_file.GetDirectory(obj.GetName()).GetListOfKeys():
                tree = tree_key.ReadObj()
                if tree.InheritsFrom("TTree"):
                    if not tree_name:
                        tree_name = tree.GetName()  # 记录第一个树名
                        chain.SetName(tree_name)  # 设置 TChain 的树名
                    chain.Add(f"{file_path}/{obj.GetName()}/{tree_name}")
                    print(f"Added tree: {tree.GetName()} from file: {file_path}")
    input_file.Close()

# 检查是否成功读取到目录名和树名
if not first_directory_name or not tree_name:
    print("No valid TDirectory or TTree found in the provided files.")
else:
    # 在输出文件中创建第一个读取到的目录
    output_dir = output_file.mkdir(first_directory_name)
    output_dir.cd()

    # 将合并的 TChain 写入为一个新的 TTree
    merged_tree = chain.CopyTree("")  # 复制 TChain 中的所有条目
    merged_tree.Write(tree_name)

    print("All trees have been successfully merged into a single TTree in merged_AO2D.root")

output_file.Close()

