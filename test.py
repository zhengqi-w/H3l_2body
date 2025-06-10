import matplotlib.pyplot as plt
import numpy as np

# 创建上下两个子图 (2行1列)
fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(8, 6))

# 设置子图间距和高度比例
fig.subplots_adjust(hspace=0.3)  # 调整水平间距
ax_top.set_position([0.1, 0.7, 0.8, 0.25])  # [左, 下, 宽, 高]
ax_bottom.set_position([0.1, 0.1, 0.8, 0.6])  # 下轴高度更大

# 在上子图绘制数据
x = np.linspace(0, 10, 100)
ax_top.plot(x, np.sin(x), 'r-', label="Top Pad")
ax_top.legend()

# 在下子图绘制数据
ax_bottom.plot(x, np.cos(x), 'b-', label="Bottom Pad")
ax_bottom.legend()

plt.show()