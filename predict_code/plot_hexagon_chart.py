import numpy as np
import matplotlib.pyplot as plt
import re
import os

# 创建保存图表的目录
if not os.path.exists('figure'):
    os.makedirs('figure')

# 从输入字符串中提取指标
def extract_metrics(input_str):
    # 提取模型名称和指标
    match = re.match(r'([^,]+),\s*F1:\s*([\d.]+)%,\s*Accuracy:\s*([\d.]+)%,\s*Precision:\s*([\d.]+)%,\s*Recall:\s*([\d.]+)%,\s*Specificity:\s*([\d.]+)%,\s*Macro AUC:\s*([\d.]+)', input_str)
    
    if match:
        model_name = match.group(1).strip()
        metrics = {
            'Model': model_name,
            'F1': float(match.group(2)),
            'Accuracy': float(match.group(3)),
            'Precision': float(match.group(4)),
            'Recall': float(match.group(5)),
            'Specificity': float(match.group(6)),
            'AUC': float(match.group(7)) * 100  # 转换为百分比以便统一比例
        }
        return metrics
    else:
        print("输入格式不正确，请使用以下格式：")
        print("模型名称, F1: 值%, Accuracy: 值%, Precision: 值%, Recall: 值%, Specificity: 值%, Macro AUC: 值")
        return None

# 绘制六边形图
def plot_hexagon_chart(metrics_list, title='模型性能比较六边形图', save_path='figure/hexagon_chart.png'):
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    
    # 设置图表大小
    fig = plt.figure(figsize=(12, 10), facecolor='white')
    ax = fig.add_subplot(111, polar=True)
    
    # 设置六边形的角度，用于平分圆周
    categories = ['F1', 'Accuracy', 'Precision', 'Recall', 'Specificity', 'AUC']
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # 闭合六边形
    
    # 设置坐标刻度和网格样式
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=14, fontweight='bold')
    
    # 设置y轴刻度范围和网格
    ax.set_ylim(0, 100)
    ax.set_yticks(np.arange(20, 101, 20))  # 设置刻度为20,40,60,80,100
    ax.set_yticklabels(['20', '40', '60', '80', '100'], fontsize=10)
    
    # 设置网格样式
    ax.grid(True, linestyle='-', alpha=0.3)
    
    # 美化坐标轴
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    # 绘制每个模型的六边形图
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    markers = ['o', 's', '^', 'D', 'v', 'p']
    
    for i, metrics in enumerate(metrics_list):
        values = [metrics[cat] for cat in categories]
        values += values[:1]  # 闭合六边形
        
        # 绘制折线和标记点
        ax.plot(angles, values, '-', linewidth=3, label=metrics['Model'], 
                color=colors[i % len(colors)], alpha=0.8)
        ax.plot(angles, values, markers[i % len(markers)], markersize=8, 
                color=colors[i % len(colors)], alpha=1.0)
        
        # 填充颜色
        ax.fill(angles, values, alpha=0.2, color=colors[i % len(colors)])
    
    # 添加图例和标题
    plt.legend(loc='upper right', bbox_to_anchor=(0.2, 0.1), fontsize=12, 
               frameon=True, facecolor='white', edgecolor='gray')
    plt.title(title, size=18, y=1.1, fontweight='bold')
    
    # 保存图表
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"六边形性能图已保存至 {save_path}")

# 主函数
def main():
    print("请输入模型指标数据，格式如下：")
    print("模型名称, F1: 值%, Accuracy: 值%, Precision: 值%, Recall: 值%, Specificity: 值%, Macro AUC: 值")
    print("输入'done'完成输入")
    
    all_metrics = []
    
    while True:
        input_str = input("请输入模型指标 (或输入'done'结束): ")
        if input_str.lower() == 'done':
            break
        
        metrics = extract_metrics(input_str)
        if metrics:
            all_metrics.append(metrics)
            print(f"已添加 {metrics['Model']} 的指标数据")
    
    # 绘制六边形图
    if all_metrics:
        title = input("请输入六边形图标题 (默认为'心血管疾病检测模型性能比较'): ")
        if not title:
            title = '心血管疾病检测模型性能比较'
        
        save_path = input("请输入保存路径 (默认为'figure/models_hexagon_chart.png'): ")
        if not save_path:
            save_path = 'figure/models_hexagon_chart.png'
        
        plot_hexagon_chart(all_metrics, title=title, save_path=save_path)
    else:
        print("没有输入有效的模型指标，无法绘制六边形图")

if __name__ == "__main__":
    main()