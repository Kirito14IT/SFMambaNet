import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os

# 获取当前脚本所在绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))

# 全局设置：使用 seaborn 提供的科研级别美观样式
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.4)

def plot_efficiency_dual_axis_updated():
    # 1. 补上缺失的 GCTNet，并保留所有的 Benchmark 模型
    methods = ['CLNet', 'MS2DGNet', 'NCMNet', 'GCTNet', 'BCLNet', 'SFMambaNet']
    
    # 提取参数量 (Params, M) 和 准确率 (mAP5°)
    params = [0.95, 2.53, 4.49, 4.09, 6.65, 2.01]
    map5 = [54.05, 49.13, 63.52, 63.80, 66.08, 71.83]

    x = np.arange(len(methods))
    
    fig, ax1 = plt.subplots(figsize=(11, 6.5))

    # ------------------ 左侧 Y 轴：画柱状图表示参数量 ------------------
    color_bar = '#A0CBE8'
    bars = ax1.bar(x, params, width=0.45, color=color_bar, edgecolor='black', linewidth=1, label='Params (M)')
    ax1.set_ylabel('Parameters (M)', fontweight='bold', fontsize=14, color='#2C3E50')
    
    # 【修复重叠】设置轴的范围，将柱状图整体压低
    ax1.set_ylim(0, 9.5) 
    ax1.tick_params(axis='y', labelcolor='#2C3E50')
    
    # 【文字倾斜】x 轴模型名字倾斜：设置 rotation=35 和 ha='right' (自左下向右上斜)
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, rotation=35, ha='right', fontweight='bold', fontsize=13)
    
    # 为柱状图添加具体的数字标签
    for bar in bars:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, yval + 0.15, f'{yval}M', ha='center', va='bottom', fontsize=11, color='#2C3E50')

    # ------------------ 右侧 Y 轴：画折线图表示准确率 ------------------
    ax2 = ax1.twinx()
    color_line = '#E74C3C'
    line = ax2.plot(x, map5, color=color_line, marker='o', markersize=10, linewidth=3, label='mAP5° (%)')
    ax2.set_ylabel('mAP5° (%)', fontweight='bold', fontsize=14, color=color_line)
    
    # 【修复重叠】设置右侧 Y 轴的范围，使得曲线位于图表中上方
    ax2.set_ylim(40, 80)
    ax2.tick_params(axis='y', labelcolor=color_line)
    ax2.grid(False) 
    
    # 为折线图添加具体的数字标签，针对低谷点进行微调
    for i, txt in enumerate(map5):
        if i == 1: # MS2DGNet 处于谷底，标签放在点下方避免压线
            y_pos = map5[i] - 1.5
            va = 'top'
        else:      # 其他点标签放在上方
            y_pos = map5[i] + 1.2
            va = 'bottom'
        ax2.text(x[i], y_pos, f'{txt}%', ha='center', va=va, fontsize=12, fontweight='bold', color=color_line)

    ax1.set_title('Trade-off between Performance and Model Size', fontweight='bold', pad=15, fontsize=16)
    
    # 突出我们的方法 (SFMambaNet 变红)
    ax1.get_xticklabels()[-1].set_color('#E74C3C')
    
    # 将左右两轴的图例合并放在左上角
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper left', fontsize=12)

    fig.tight_layout()
    save_path = os.path.join(current_dir, 'efficiency_dual_axis_updated.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"更新后的双轴图已保存至: {save_path}")
    plt.show()

if __name__ == '__main__':
    plot_efficiency_dual_axis_updated()