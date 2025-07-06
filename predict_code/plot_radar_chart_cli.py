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

# 绘制雷达图
def plot_radar_chart(metrics_list, title='模型性能比较雷达图', save_path='figure/radar_chart.png'):
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    
    # 设置图表大小
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, polar=True)
    
    # 设置雷达图的角度，用于平分圆周
    categories = ['F1', 'Accuracy', 'Precision', 'Recall', 'Specificity', 'AUC']
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # 闭合雷达图
    
    # 设置雷达图的坐标刻度
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=12)
    
    # 设置y轴刻度范围
    ax.set_ylim(0, 100)
    
    # 绘制每个模型的雷达图
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    for i, metrics in enumerate(metrics_list):
        values = [metrics[cat] for cat in categories]
        values += values[:1]  # 闭合雷达图
        
        # 绘制折线
        ax.plot(angles, values, 'o-', linewidth=2, label=metrics['Model'], color=colors[i % len(colors)])
        # 填充颜色
        ax.fill(angles, values, alpha=0.1, color=colors[i % len(colors)])
    
    # 添加图例和标题
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1), fontsize=12)
    plt.title(title, size=15, y=1.1)
    
    # 保存图表
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"雷达图已保存至 {save_path}")

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
    
    # 绘制雷达图
    if all_metrics:
        title = input("请输入雷达图标题 (默认为'心血管疾病检测模型性能比较'): ")
        if not title:
            title = '心血管疾病检测模型性能比较'
        
        save_path = input("请输入保存路径 (默认为'figure/models_radar_chart.png'): ")
        if not save_path:
            save_path = 'figure/models_radar_chart.png'
        
        plot_radar_chart(all_metrics, title=title, save_path=save_path)
    else:
        print("没有输入有效的模型指标，无法绘制雷达图")

if __name__ == "__main__":
    main()