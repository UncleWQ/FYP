import numpy as np
import matplotlib.pyplot as plt

# 加载.npy文件
data = np.load('./12_DTweighted/train/4101010001210420_1_001_DTweighted.npy')  # 使用实际路径替换
out_file_path = 'data.txt'
with open(out_file_path,'w') as f:
    f.write(np.array2string(data, threshold=np.inf))

print("Printed!")
print("Array Shape:",data.shape)
print("Data Shape:",data.dtype)

# 设置时间轴，假设每十个数据点代表1秒，总共有64个数据点
time_ticks = np.linspace(0, 6.4, 64)  # 64个数据点，最大值为6.4秒

# 设置频率轴，从-640 Hz到+640 Hz，总共128个数据点
frequency_ticks = np.linspace(-640, 640, 128)

# 创建图像并设置大小
plt.figure(figsize=(12, 6))

# 绘制Doppler-time map
plt.imshow(data, aspect='auto', cmap='viridis', interpolation='none',
           extent=[time_ticks.min(), time_ticks.max(), frequency_ticks.min(), frequency_ticks.max()])

# 设置颜色条和标题
plt.colorbar(label='Intensity')
plt.title('Doppler-Time Map')
plt.xlabel('Time (seconds)')
plt.ylabel('Frequency (Hz)')

# 调整刻度密度
plt.xticks(np.arange(0, time_ticks.max(), 0.5))
plt.yticks(np.arange(frequency_ticks.min(), frequency_ticks.max()+1, 128))

# 保存图像，使用较高的dpi值
plt.savefig('doppler_time_map_high_res.png', dpi=300)  # 使用高分辨率保存

plt.show()

