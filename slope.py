import torch
import matplotlib.pyplot as plt

# 生成示例数据
x = torch.linspace(0, 10, 100)
y = torch.sin(x)

# 计算相邻斜率差异
slopes = torch.diff(y) / torch.diff(x)

# 计算斜率差异的均值和方差
mean_slope = torch.mean(slopes)
var_slope = torch.var(slopes)

# 绘制折线图
plt.plot(x, y)
# plt.
plt.xlabel('x')
plt.ylabel('y')
plt.title('Line Plot')

# 打印平滑度信息
print('Mean slope:', mean_slope.item())
print('Variance of slopes:', var_slope.item())

# 显示图形
plt.show()
