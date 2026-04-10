import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# ========================================================
# 全局设置：使用 seaborn 提供的科研级别美观白底网格样式
# ========================================================
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.4)

def plot_line_chart():
    # --------------------------------------------------------
    # 2. 折线图：基于 Table IV (Homography Estimation on HPatches)
    # --------------------------------------------------------
    methods_line = ['PointCN', 'OANet++', 'CLNet', 'MS2DGNet', 'NCMNet', 'BCLNet', 'MatchMamba', 'SFMambaNet(Ours)']
    
    # 提取论文中 3PX, 5PX, 10PX 的准确率数据
    acc_3px = [67.93, 69.66, 69.83, 65.00, 70.69, 70.92, 71.55, 73.00]
    acc_5px = [82.59, 82.93, 81.55, 78.97, 81.90, 82.87, 84.66, 85.47]
    acc_10px = [92.76, 91.90, 90.69, 88.45, 91.03, 91.57, 92.23, 92.81]

    thresholds = ['3PX', '5PX', '10PX']

    fig, ax = plt.subplots(figsize=(10, 6.5))
    
    # 使用 husl 均匀色板分配对比色的基线模型
    colors = sns.color_palette("husl", len(methods_line))
    markers = ['o', 's', '^', 'v', 'p', 'h', 'D', '*']

    # 循环画线
    for i, method in enumerate(methods_line):
        scores = [acc_3px[i], acc_5px[i], acc_10px[i]]
        
        # 重点：加粗并用醒目的红色突出我们自己的方法 (SFMambaNet)
        if 'SFMambaNet' in method:
            linewidth = 3.5
            markersize = 14
            color = '#E74C3C'  # 鲜明的红色
            zorder = 10        # 置于最上层，不被其他线遮挡
        else:
            linewidth = 1.8
            markersize = 8
            color = colors[i]
            zorder = 1
            
        ax.plot(thresholds, scores, marker=markers[i], markersize=markersize, linewidth=linewidth, 
                label=method, color=color, zorder=zorder)

    # 图表细节修饰
    ax.set_xlabel('Error Threshold', fontweight='bold', fontsize=14)
    ax.set_ylabel('Accuracy (%)', fontweight='bold', fontsize=14)
    ax.set_title('Homography Estimation Accuracy on HPatches', fontweight='bold', pad=15, fontsize=16)

    # 将图例放在图表外部右侧，避免遮挡数据线
    ax.legend(title='Methods', title_fontsize='13', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.7)

    fig.tight_layout()
    plt.savefig('homography_line.png', dpi=300, bbox_inches='tight')
    plt.show()

# 执行绘图
if __name__ == '__main__':
    plot_line_chart()