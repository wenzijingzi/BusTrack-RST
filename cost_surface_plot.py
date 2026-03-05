import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

def plot_cost_surfaces():
    # 1. 数据准备
    # X轴: IoU Overlap (从 0 到 1)
    iou = np.linspace(0, 1, 100)
    # Y轴: Time Interval (从 0 到 30 帧)
    time_interval = np.linspace(0, 30, 100)
    X_iou, Y_time = np.meshgrid(iou, time_interval)

    # 2. 参数设置 (基于 matching_soft+oip+tcr_v1.py 的默认值)
    lambda_occ = 0.25  # 对应源码中的 alpha (Soft-OIP 权重)
    alpha_tcr = 0.20  # 对应源码中的 beta (Soft-TCR 权重)
    iou_threshold = 0.35  # 对应源码中的 trig_iou (作为硬截断对比的阈值)

    # 3. Soft-Constraint Cost (软约束代价计算)
    # 基础代价: 假设为 IoU Distance (1 - IoU)
    C_base = 1 - X_iou
    # Soft-OIP: 随 IoU 减小呈指数增长
    P_occ = np.exp(lambda_occ * (1 - X_iou))
    # Soft-TCR: 随时间增加呈对数增长
    W_time = 1 + alpha_tcr * np.log(1 + Y_time)
    # 最终软代价
    Z_soft = C_base * P_occ * W_time

    # 4. Hard Gating Cost (硬阈值代价计算)
    # 初始化为基础代价
    Z_hard = 1 - X_iou
    # 模拟“硬截断”: IoU < 阈值时，代价设为一个极大值 (模拟无穷大)
    high_cost_value = np.max(Z_soft) * 1.1 # 设为比软代价最大值稍高，方便视觉对比
    Z_hard[X_iou < iou_threshold] = high_cost_value

    # 5. 绘图
    fig = plt.figure(figsize=(16, 7))

    # --- 子图 1: Soft Constraints ---
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    surf1 = ax1.plot_surface(X_iou, Y_time, Z_soft, cmap='viridis',
                           linewidth=0, antialiased=False, alpha=0.9)
    # 标签与美化
    ax1.set_title('(a) Proposed Soft-Constraint Mechanism\n(Continuous Decision Boundary)', fontsize=14, pad=10, weight='bold')
    ax1.set_xlabel('IoU Overlap', fontsize=11, labelpad=10)
    ax1.set_ylabel('Time Interval (Frames)', fontsize=11, labelpad=10)
    ax1.set_zlabel('Association Cost', fontsize=11, labelpad=5)
    ax1.set_zlim(0, high_cost_value)
    ax1.view_init(elev=30, azim=220) # 调整视角以获得最佳观察效果
    ax1.invert_xaxis() # 反转 X 轴，让 IoU 从 1 到 0 变化更符合直觉（Cost 升高）

    # --- 子图 2: Hard Gating ---
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    surf2 = ax2.plot_surface(X_iou, Y_time, Z_hard, cmap='coolwarm',
                           linewidth=0, antialiased=False, alpha=0.9)
    # 标签与美化
    ax2.set_title('(b) Traditional Hard Gating\n(Discrete/Cutoff Boundary)', fontsize=14, pad=10, weight='bold')
    ax2.set_xlabel('IoU Overlap', fontsize=11, labelpad=10)
    ax2.set_ylabel('Time Interval (Frames)', fontsize=11, labelpad=10)
    ax2.set_zlabel('Association Cost', fontsize=11, labelpad=5)
    ax2.set_zlim(0, high_cost_value)
    ax2.view_init(elev=30, azim=220)
    ax2.invert_xaxis()

    plt.tight_layout()
    plt.savefig('Cost_Surface_Comparison.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    plot_cost_surfaces()