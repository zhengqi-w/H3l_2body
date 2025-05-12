# H3l_2body_spectrum

### some information
- [colors for root in Hexadecimal](https://www.colorhexa.com)
- [hipe4ml index](https://hipe4ml.github.io/hipe4ml/index.html)
- [pandas API reference](https://pandas.pydata.org/pandas-docs/stable/reference/index.html#api)
- [matplotlib API reference](https://matplotlib.org/stable/users/explain/figure/api_interfaces.html#api-interfaces)

### How to update O2Physics
1. update alidist 
```bash
cd ~/alice/alidist
git reset --hard HEAD(abandon the uncommited version)
git pull
```
2. update O2Physics source code
```bash
cd ~/alice/O2Physics
git checkout master
git pull --rebase upstream master
```
3. rebuild O2Physics
```bash
cd ~/alice
aliBuild build O2Physics --defaults --debug(option) o2
```


### Dependencies
- ROOT > 6.26 (if conda please forged with GSL)
- hipe4ml, possibly in dev mode. To install the in-development package:
```bash
conda config --add channels conda-forge
conda config --set channel_priority strict
conda install root=6.30 gsl -y
```

```bash
git clone https://github.com/hipe4ml/hipe4ml.git
```

Then, from the repository base directory
```bash
git checkout -b dev
git pull upstream/dev
pip install -e .[dev]
```


### How to run the analyses
- Inspect the output tree and produce basic histograms:
```bash
python3 analyse_tree.py --config-file config/analyse_tree/your_config.yaml
```
- Extract the raw and corrected pt spectrum:
```bash
python3 pt_analysis.py --config-file config/pt_analysis/your_config.yaml
```

- Extract the raw and corrected ct spectrum:
```bash
python3 ct_analysis.py --config-file config/ct_analysis/your_config.yaml
```

- Do the BDT Training and extract the spection:
```bash
python3 train_test_data.py --config-file config/train_test_data/your_config.yaml
python3 spectrum_training.py --config-file config/spectrum_training/your_config.yaml
```

- lifetime Extraction with sWeightes and COWs
```bash
python3 lifetime_extraction.py --config-file config/lifetime_extraction/your_config.yaml
```

# Vim 常用快捷键指南

Vim 是一款功能强大的文本编辑器，熟练使用快捷键能大大提高编辑效率。本文列出了一些常用且实用的 Vim 快捷键。

---

## 目录
- [基本操作](#基本操作)
- [移动光标](#移动光标)
- [文本操作](#文本操作)
- [查找和替换](#查找和替换)
- [可视模式](#可视模式)
- [代码缩进](#代码缩进)
- [保存和退出](#保存和退出)
- [窗口分割](#窗口分割)
- [标签页](#标签页)

---

### 基本操作
| 快捷键 | 说明                     |
|--------|--------------------------|
| `i`    | 进入插入模式（在光标前插入） |
| `a`    | 进入插入模式（在光标后插入） |
| `I`    | 进入插入模式，并跳到行首 |
| `A`    | 进入插入模式，并跳到行尾 |
| `o`    | 在当前行下方新建一行并进入插入模式 |
| `O`    | 在当前行上方新建一行并进入插入模式 |
| `Esc`  | 退出插入模式，回到正常模式 |

### 移动光标
| 快捷键      | 说明                |
|-------------|---------------------|
| `h`         | 向左移动            |
| `j`         | 向下移动            |
| `k`         | 向上移动            |
| `l`         | 向右移动            |
| `w`         | 移动到下一个单词的开头 |
| `b`         | 移动到上一个单词的开头 |
| `e`         | 移动到当前单词的末尾 |
| `0`         | 移动到行首          |
| `$`         | 移动到行尾          |
| `gg`        | 移动到文件开头       |
| `G`         | 移动到文件末尾       |
| `Ctrl + u`  | 向上滚动半屏        |
| `Ctrl + d`  | 向下滚动半屏        |

### 文本操作
| 快捷键      | 说明                  |
|-------------|-----------------------|
| `x`         | 删除光标所在字符      |
| `dd`        | 删除当前行            |
| `yy`        | 复制当前行            |
| `p`         | 粘贴复制/剪切的内容   |
| `u`         | 撤销操作              |
| `Ctrl + r`  | 重做操作              |
| `r`         | 替换光标所在的字符    |
| `c`         | 更改当前光标处的单词  |

### 查找和替换
| 快捷键               | 说明                            |
|----------------------|---------------------------------|
| `/`                  | 输入后跟关键词，查找关键词       |
| `?`                  | 反向查找                        |
| `n`                  | 跳到下一个匹配项                |
| `N`                  | 跳到上一个匹配项                |
| `:%s/旧文本/新文本/g` | 替换文件中所有匹配项            |

### 可视模式
| 快捷键      | 说明                      |
|-------------|---------------------------|
| `v`         | 进入可视模式              |
| `V`         | 进入行可视模式            |
| `Ctrl + v`  | 进入块可视模式（适合列操作） |
| `y`         | 在可视模式下复制选中文本  |
| `d`         | 在可视模式下删除选中文本  |

### 代码缩进
| 快捷键    | 说明                  |
|-----------|-----------------------|
| `>>`      | 向右缩进              |
| `<<`      | 向左缩进              |
| `=`       | 自动对齐              |
| `gg=G`    | 自动格式化整个文件     |

### 保存和退出
| 快捷键       | 说明                |
|--------------|---------------------|
| `:w`         | 保存文件            |
| `:q`         | 退出文件            |
| `:wq` 或 `:x` | 保存并退出         |
| `:q!`        | 不保存强制退出      |
| `ZZ`         | 保存并退出          |

### 窗口分割
| 快捷键             | 说明                      |
|--------------------|---------------------------|
| `:split` 或 `:sp`  | 水平分割窗口               |
| `:vsplit` 或 `:vsp` | 垂直分割窗口              |
| `Ctrl + w + w`     | 在分割窗口间切换           |
| `Ctrl + w + h/j/k/l` | 在分割窗口间移动         |

### 标签页
| 快捷键               | 说明                    |
|----------------------|-------------------------|
| `:tabnew 文件名`     | 新建标签页              |
| `gt`                 | 切换到下一个标签页      |
| `gT`                 | 切换到上一个标签页      |
| `:tabclose`          | 关闭当前标签页          |
| `:tabs`              | 查看所有标签页          |

---

掌握这些快捷键能够显著提高 Vim 的使用效率，可以从基础命令开始，逐步熟悉更复杂的操作。


### Some Python modole guide in this analysis
- pdg for debugging
```bash 
pdb.set_trace() # set a breakpoint
n # continue to next line
c  # continue to next breakpoint
l  # list the code
p <value> # print the value
q  # quit
```
- enumerate use with random array
```bash
random_arr = np.random.rand(len(df)) # gengerate a random array 
for ind, (val, rand) in enumerate(zip(df[var],random_arr)): # zip: pair the two different numpy array// zip(df[var], random_arr) 会返回一个由元组组成的迭代器，每个元组包含 df[var] 和 random_arr 中对应位置的元素。 enumerate 为每个 zip 生成的元组提供一个索引值 ind，这个索引值从 0 开始递增。
```
- isinstance 
```bash
if isinstance(df, pd.DataFrame): # check if df is a pandas DataFrame
```

