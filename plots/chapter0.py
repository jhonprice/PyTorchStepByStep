import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
from sklearn.linear_model import LinearRegression
plt.style.use('fivethirtyeight')


def fit_model(x_train, y_train):
    # Fits a linear regression to find the actual b and w that minimize the loss
    # 使用线性回归拟合训练数据，找到使损失最小的截距 b 和斜率 w
    # 该函数使用 sklearn 的 LinearRegression 解析求解最优参数，作为梯度下降的参考目标
    regression = LinearRegression()
    regression.fit(x_train, y_train)
    # regression.intercept_ 返回截距，regression.coef_ 返回系数（斜率）
    b_minimum, w_minimum = regression.intercept_[0], regression.coef_[0][0]
    return b_minimum, w_minimum


def find_index(b, w, bs, ws):
    # Looks for the closer indexes for the updated b and w inside their respective ranges
    # 在参数网格 bs 和 ws 中查找最接近给定 b 和 w 的索引位置
    # 由于损失曲面是离散采样的，需要在离散网格中找到与连续参数值最接近的网格点
    b_idx = np.argmin(np.abs(bs[0, :] - b))
    w_idx = np.argmin(np.abs(ws[:, 0] - w))

    # Closest values for b and w
    # 获取网格中与给定 b、w 最接近的实际参数值
    fixedb, fixedw = bs[0, b_idx], ws[w_idx, 0]

    return b_idx, w_idx, fixedb, fixedw


def figure1(x_train, y_train, x_val, y_val):
    # 绘制训练集和验证集数据的散点图，用于可视化数据分布
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    # 左子图：训练集散点图（蓝色默认）
    ax[0].scatter(x_train, y_train)
    ax[0].set_xlabel('x')
    ax[0].set_ylabel('y')
    ax[0].set_ylim([0, 3.1])
    ax[0].set_title('Generated Data - Train')

    # 右子图：验证集散点图（红色）
    ax[1].scatter(x_val, y_val, c='r')
    ax[1].set_xlabel('x')
    ax[1].set_ylabel('y')
    ax[1].set_ylim([0, 3.1])
    ax[1].set_title('Generated Data - Validation')
    fig.tight_layout()

    return fig, ax


def figure2(x_train, y_train, b, w, color='k'):
    # Generates evenly spaced x feature
    # 生成 [0, 1] 区间内均匀分布的 101 个 x 值，用于绘制连续的预测线
    x_range = np.linspace(0, 1, 101)
    # Computes yhat
    # 根据线性模型 y = b + w*x 计算预测值 yhat
    yhat_range = b + w * x_range

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_ylim([0, 3])

    # Dataset
    # 绘制训练数据的散点图
    ax.scatter(x_train, y_train)
    # Predictions
    # 绘制模型的预测线（虚线），表示拟合结果
    ax.plot(x_range, yhat_range, label='Model\'s predictions', c=color, linestyle='--')

    # Annotations
    # 在图中标注当前的参数值 b 和 w
    ax.annotate('b = {:.4f} w = {:.4f}'.format(b[0], w[0]), xy=(.2, .55), c=color)
    ax.legend(loc=0)
    fig.tight_layout()
    return fig, ax


def figure3(x_train, y_train, b, w):
    # 在 figure2 的基础上，高亮第一个数据点的预测误差
    fig, ax = figure2(x_train, y_train, b, w)

    # First data point
    # 获取第一个训练数据点的坐标
    x0, y0 = x_train[0][0], y_train[0][0]
    # First data point
    # 将第一个数据点标为红色，突出显示
    ax.scatter([x0], [y0], c='r')
    # Vertical line showing error between point and prediction
    # 绘制从真实值到预测值的垂直误差线（红色虚线）
    ax.plot([x0, x0], [b[0] + w[0] * x0, y0 - .03], c='r', linewidth=2, linestyle='--')
    # 在真实值端绘制向下的箭头
    ax.arrow(x0, y0 - .03, 0, .03, color='r', shape='full', lw=0, length_includes_head=True, head_width=.03)
    # 在预测值端绘制向上的箭头
    ax.arrow(x0, b[0] + w[0] * x0 + .05, 0, -.03, color='r', shape='full', lw=0, length_includes_head=True, head_width=.03)
    # Annotations
    # 标注误差项 error_0，表示第一个样本的预测误差
    ax.annotate(r'$error_0$', xy=(.8, 1.5))

    fig.tight_layout()
    return fig, ax


def figure4(x_train, y_train, b, w, bs, ws, all_losses):
    # 使用 sklearn 线性回归求出解析最优解，作为全局最小值的参考点
    b_minimum, w_minimum = fit_model(x_train, y_train)

    figure = plt.figure(figsize=(12, 6))

    # 1st plot
    # 左子图：3D 损失曲面图，直观展示参数 (b, w) 与损失值 MSE 的关系
    ax1 = figure.add_subplot(1, 2, 1, projection='3d')
    ax1.set_xlabel('b')
    ax1.set_ylabel('w')
    ax1.set_title('Loss Surface')

    # 绘制 3D 曲面，alpha 控制透明度，cmap 控制颜色映射
    surf = ax1.plot_surface(bs, ws, all_losses, rstride=1, cstride=1, alpha=.5, cmap=plt.cm.jet, linewidth=0, antialiased=True)
    # 在 z=-1 的平面（曲面下方）绘制等高线投影，方便从顶部观察损失地貌
    ax1.contour(bs[0, :], ws[:, 0], all_losses, 10, offset=-1, cmap=plt.cm.jet)

    # 在 3D 曲面上标记全局最优点（黑色点），并标注 "Minimum"
    bidx, widx, _, _ = find_index(b_minimum, w_minimum, bs, ws)
    ax1.scatter(b_minimum, w_minimum, all_losses[bidx, widx], c='k')
    ax1.text(-.3, 2.5, all_losses[bidx, widx], 'Minimum', zdir=(1, 0, 0))
    # Random start
    # 在 3D 曲面上标记随机初始点（黑色点），并标注 "Random Start"
    bidx, widx, _, _ = find_index(b, w, bs, ws)
    ax1.scatter(b, w, all_losses[bidx, widx], c='k')
    # Annotations
    ax1.text(-.2, -1.5, all_losses[bidx, widx], 'Random\n Start', zdir=(1, 0, 0))

    # 设置 3D 视角（仰角 40°，方位角 260°）
    ax1.view_init(40, 260)

    # 2nd plot
    # 右子图：2D 等高线图（损失曲面的俯视图），便于在二维平面上观察优化路径
    ax2 = figure.add_subplot(1, 2, 2)
    ax2.set_xlabel('b')
    ax2.set_ylabel('w')
    ax2.set_title('Loss Surface')

    # Loss surface
    # 绘制损失曲面的等高线，每条等高线代表相同的损失值
    CS = ax2.contour(bs[0, :], ws[:, 0], all_losses, cmap=plt.cm.jet)
    # 在等高线上标注对应的损失值
    ax2.clabel(CS, inline=1, fontsize=10)
    # Minimum
    # 标记全局最优点
    ax2.scatter(b_minimum, w_minimum, c='k')
    # Random start
    # 标记随机梯度下降的起始点
    ax2.scatter(b, w, c='k')
    # Annotations
    ax2.annotate('Random Start', xy=(-.2, 0.05), c='k')
    ax2.annotate('Minimum', xy=(.5, 2.2), c='k')

    figure.tight_layout()
    return figure, (ax1, ax2)


def figure5(x_train, y_train, b, w, bs, ws, all_losses):
    # 展示固定 b 时的损失曲线（垂直截面），用于理解沿 w 方向的损失变化
    b_minimum, w_minimum = fit_model(x_train, y_train)

    # 在网格中找到当前参数 (b, w) 对应的索引和实际网格值
    b_idx, w_idx, fixedb, fixedw = find_index(b, w, bs, ws)

    b_range = bs[0, :]
    w_range = ws[:, 0]

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # 左子图：损失曲面等高线图 + 红色虚线标注的垂直截面（固定 b 值）
    axs[0].set_title('Loss Surface')
    axs[0].set_xlabel('b')
    axs[0].set_ylabel('w')
    # Loss surface - 等高线
    CS = axs[0].contour(bs[0, :], ws[:, 0], all_losses, cmap=plt.cm.jet)
    axs[0].clabel(CS, inline=1, fontsize=10)
    # Minimum - 全局最优点
    axs[0].scatter(b_minimum, w_minimum, c='k')
    # Starting point - 当前参数位置
    axs[0].scatter(fixedb, fixedw, c='k')
    # Vertical section
    # 垂直截面线（红色虚线）：固定 b 不变，展示沿 w 方向切一刀的截面位置
    axs[0].plot([fixedb, fixedb], w_range[[0, -1]], linestyle='--', c='r', linewidth=2)
    # Annotations
    axs[0].annotate('Minimum', xy=(.5, 2.2), c='k')
    axs[0].annotate('Random Start', xy=(fixedb + .1, fixedw + .1), c='k')

    # 右子图：固定 b 时损失随 w 变化的曲线（垂直截面的展开视图）
    axs[1].set_ylim([-.1, 15.1])
    axs[1].set_xlabel('w')
    axs[1].set_ylabel('Loss')
    axs[1].set_title('Fixed: b = {:.2f}'.format(fixedb))
    # Loss
    # 绘制沿 w 方向的损失曲线（红色虚线），all_losses[:, b_idx] 取固定 b 索引下所有 w 的损失
    axs[1].plot(w_range, all_losses[:, b_idx], c='r', linestyle='--', linewidth=2)
    # Starting point
    # 标记当前参数在损失曲线上的位置（红色圆点）
    axs[1].plot([fixedw], [all_losses[w_idx, b_idx]], 'or')

    fig.tight_layout()
    return fig, axs


def figure6(x_train, y_train, b, w, bs, ws, all_losses):
    # 展示固定 w 时的损失曲线（水平截面），用于理解沿 b 方向的损失变化
    # 与 figure5 互补：figure5 固定 b 看 w，figure6 固定 w 看 b
    b_minimum, w_minimum = fit_model(x_train, y_train)

    b_idx, w_idx, fixedb, fixedw = find_index(b, w, bs, ws)

    b_range = bs[0, :]
    w_range = ws[:, 0]

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # 左子图：损失曲面等高线图 + 黑色虚线标注的水平截面（固定 w 值）
    axs[0].set_title('Loss Surface')
    axs[0].set_xlabel('b')
    axs[0].set_ylabel('w')
    # Loss surface - 等高线
    CS = axs[0].contour(bs[0, :], ws[:, 0], all_losses, cmap=plt.cm.jet)
    axs[0].clabel(CS, inline=1, fontsize=10)
    # Minimum - 全局最优点
    axs[0].scatter(b_minimum, w_minimum, c='k')
    # Starting point - 当前参数位置
    axs[0].scatter(fixedb, fixedw, c='k')
    # Horizontal section
    # 水平截面线（黑色虚线）：固定 w 不变，展示沿 b 方向切一刀的截面位置
    axs[0].plot(b_range[[0, -1]], [fixedw, fixedw], linestyle='--', c='k', linewidth=2)
    # Annotations
    axs[0].annotate('Minimum', xy=(.5, 2.2), c='k')
    axs[0].annotate('Random Start', xy=(fixedb + .1, fixedw + .1), c='k')

    # 右子图：固定 w 时损失随 b 变化的曲线（水平截面的展开视图）
    axs[1].set_ylim([-.1, 15.1])
    axs[1].set_xlabel('b')
    axs[1].set_ylabel('Loss')
    axs[1].set_title('Fixed: w = {:.2f}'.format(fixedw))
    # Loss
    # 绘制沿 b 方向的损失曲线（黑色虚线），all_losses[w_idx, :] 取固定 w 索引下所有 b 的损失
    axs[1].plot(b_range, all_losses[w_idx, :], c='k', linestyle='--', linewidth=2)
    # Starting point
    # 标记当前参数在损失曲线上的位置（黑色圆点）
    axs[1].plot([fixedb], [all_losses[w_idx, b_idx]], 'ok')

    fig.tight_layout()
    return fig, axs


def figure7(b, w, bs, ws, all_losses):
    # 在 figure5/figure6 的基础上添加矩形区域，放大展示当前参数附近的损失曲线细节
    b_range = bs[0, :]
    w_range = ws[:, 0]

    b_idx, w_idx, fixedb, fixedw = find_index(b, w, bs, ws)

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # 左子图：固定 b，损失随 w 变化（红色），红色半透明矩形标注放大的局部区域
    axs[0].set_ylim([-.1, 6.1])
    axs[0].set_xlabel('w')
    axs[0].set_ylabel('MSE (loss)')
    axs[0].set_title('Fixed: b = {:.2f}'.format(fixedb))
    # Red rectangle
    # 红色半透明矩形：标识 figure8 中将放大展示的区域范围
    rect = Rectangle((-.3, 2.3), .5, .5)
    pc = PatchCollection([rect], facecolor='r', alpha=.3, edgecolor='r')
    axs[0].add_collection(pc)
    # Loss - fixed b
    # 固定 b 时的损失曲线
    axs[0].plot(w_range, all_losses[:, b_idx], c='r', linestyle='--', linewidth=2)
    # Starting point
    axs[0].plot([fixedw], [all_losses[w_idx, b_idx]], 'or')

    # 右子图：固定 w，损失随 b 变化（黑色），黑色半透明矩形标注放大的局部区域
    axs[1].set_ylim([-.1, 6.1])
    axs[1].set_xlabel('b')
    axs[1].set_ylabel('MSE (loss)')
    axs[1].set_title('Fixed: w = {:.2f}'.format(fixedw))
    axs[1].label_outer()
    # Black rectangle
    # 黑色半透明矩形：标识 figure8 中将放大展示的区域范围
    rect = Rectangle((.3, 2.3), .5, .5)
    pc = PatchCollection([rect], facecolor='k', alpha=.3, edgecolor='k')
    axs[1].add_collection(pc)
    # Loss - fixed w
    # 固定 w 时的损失曲线
    axs[1].plot(b_range, all_losses[w_idx, :], c='k', linestyle='--', linewidth=2)
    # Starting point
    axs[1].plot([fixedb], [all_losses[w_idx, b_idx]], 'ok')

    fig.tight_layout()
    return fig, axs


def loss_curves(b_idx, w_idx, b_idx_after, w_idx_after, all_losses):
    # 从预计算的损失矩阵中提取更新前后的各条损失曲线和损失值
    # 用于后续计算数值梯度和绘制参数更新前后的对比图

    # BEFORE - 更新前的状态
    # Loss curve for b, given w is fixed
    # 固定 w 时，损失随 b 变化的整条曲线（所有 b 值对应的损失）
    loss_fixedw = all_losses[w_idx, :]
    # Loss curve for w, given b is fixed
    # 固定 b 时，损失随 w 变化的整条曲线（所有 w 值对应的损失）
    loss_fixedb = all_losses[:, b_idx]
    # Loss before
    # 更新前当前参数位置的损失值（单个标量）
    loss_before = all_losses[w_idx, b_idx]

    # AFTER - 更新后的状态（参数分别沿 b 和 w 方向移动后的损失值）
    # Loss after w is updated
    # 仅更新 w（b 保持原值）后的损失值
    loss_after_w = all_losses[w_idx_after, b_idx]
    # Loss after b is updated
    # 仅更新 b（w 保持原值）后的损失值
    loss_after_b = all_losses[w_idx, b_idx_after]
    return loss_fixedb, loss_fixedw, loss_before, loss_after_b, loss_after_w


def calc_gradient(parm_before, parm_after, loss_before, loss_after):
    # 使用有限差分法（数值微分）手动计算参数的梯度
    # 梯度 ≈ Δloss / Δparam，即损失变化量与参数变化量的比值
    # Computes changes in parm and loss
    # 计算参数的变化量 Δparam
    delta_parm = parm_after - parm_before
    # 计算损失的变化量 Δloss
    delta_loss = loss_after - loss_before
    # Computes gradient for parm
    # 数值梯度 = Δloss / Δparam，这是对解析梯度的一种近似
    manual_grad = delta_loss / delta_parm
    return manual_grad, delta_parm, delta_loss


def figure8(b, w, bs, ws, all_losses):
    # 放大展示 figure7 中矩形标注的区域，可视化数值梯度的计算过程
    # 核心思想：通过参数微扰（+0.12）观察损失变化，用 Δloss/Δparam 近似梯度
    b_range = bs[0, :]
    w_range = ws[:, 0]

    # BEFORE
    # 获取更新前参数在网格中的索引和实际值
    b_idx, w_idx, bs_before, ws_before = find_index(b, w, bs, ws)
    # AFTER
    # 获取参数微扰后（b+0.12, w+0.12）在网格中的索引和实际值
    b_idx_after, w_idx_after, bs_after, ws_after = find_index(b + .12, w + .12, bs, ws)

    # 提取更新前后的各条损失曲线和损失值
    loss_fixedb, loss_fixedw, loss_before, loss_after_b, loss_after_w = loss_curves(b_idx, w_idx, b_idx_after, w_idx_after, all_losses)

    # Computes gradient for b
    # 用有限差分法计算 b 方向的数值梯度（固定 w，看 b 变化对损失的影响）
    manual_grad_b, delta_b, delta_mse_b = calc_gradient(bs_before, bs_after, loss_before, loss_after_b)
    # Computes gradient for w
    # 用有限差分法计算 w 方向的数值梯度（固定 b，看 w 变化对损失的影响）
    manual_grad_w, delta_w, delta_mse_w = calc_gradient(ws_before, ws_after, loss_before, loss_after_w)

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # 左子图：放大展示固定 b 时 w 方向的损失曲线局部，用箭头标注 δw 和 δMSE
    axs[0].set_ylim([2.3, 2.8])
    axs[0].set_xlim([-.3, .2])
    axs[0].set_xlabel('w')
    axs[0].set_ylabel('MSE (loss)')
    axs[0].set_title('Fixed: b = {:.2f}'.format(bs_before))
    # Loss curve - 损失曲线（红色虚线）
    axs[0].plot(w_range, loss_fixedb, c='r', linestyle='--', linewidth=2)
    # Point - before - 更新前的点
    axs[0].plot([ws_before], [loss_before], 'or')
    # Point - after - 更新后的点
    axs[0].plot([ws_after], [loss_after_w], 'or')

    # Arrows - 用箭头和线段标注参数变化量 δw 和损失变化量 δMSE
    # 水平箭头：标注 w 的变化方向
    axs[0].arrow(ws_after, loss_before, .01, 0, color='r', shape='full', lw=0, length_includes_head=True, head_width=.01)
    # 垂直箭头：标注 MSE 的变化方向
    axs[0].arrow(ws_before, loss_after_w, 0, -0.01, color='r', shape='full', lw=0, length_includes_head=True, head_width=.01)
    # 水平线段：连接更新前后的 w 值
    axs[0].plot([ws_before, ws_after], [loss_before, loss_before], 'r-', linewidth=1.5)
    # 垂直线段：连接更新前后的损失值
    axs[0].plot([ws_before, ws_before], [loss_after_w, loss_before], 'r-', linewidth=1.5)

    # Annotations - 用 LaTeX 公式标注 δw、δMSE 和梯度近似值
    axs[0].annotate(r'$\delta w = {:.2f}$'.format(delta_w), xy=(.0, 2.7), c='k', fontsize=15)
    axs[0].annotate(r'$\delta MSE = {:.2f}$'.format(delta_mse_w), xy=(-.23, 2.45), c='k', fontsize=15)
    axs[0].annotate(r'$\frac{\delta MSE}{\delta w} \approx' + '{:.2f}$'.format(manual_grad_w), xy=(-.05, 2.6), c='k', fontsize=17)

    # 右子图：放大展示固定 w 时 b 方向的损失曲线局部，用箭头标注 δb 和 δMSE
    axs[1].set_ylim([2.3, 2.8])
    axs[1].set_xlim([.3, .8])
    axs[1].set_xlabel('b')
    axs[1].set_ylabel('MSE (loss)')
    axs[1].set_title('Fixed: w = {:.2f}'.format(ws_before))
    # Loss Curve - 损失曲线（黑色虚线）
    axs[1].plot(b_range, loss_fixedw, c='k', linestyle='--', linewidth=2)
    # Point - before - 更新前的点
    axs[1].plot([bs_before], [loss_before], 'ok')
    # Point - after - 更新后的点
    axs[1].plot([bs_after], [loss_after_b], 'ok')

    # Arrows - 用箭头和线段标注参数变化量 δb 和损失变化量 δMSE
    # 水平箭头：标注 b 的变化方向
    axs[1].arrow(bs_after, loss_before, .01, 0, color='k', shape='full', lw=0, length_includes_head=True, head_width=.01)
    # 垂直箭头：标注 MSE 的变化方向
    axs[1].arrow(bs_before, loss_after_b, 0, -0.01, color='k', shape='full', lw=0, length_includes_head=True, head_width=.01)
    # 水平线段：连接更新前后的 b 值
    axs[1].plot([bs_before, bs_after], [loss_before, loss_before], 'k-', linewidth=1.5)
    # 垂直线段：连接更新前后的损失值
    axs[1].plot([bs_before, bs_before], [loss_after_b, loss_before], 'k-', linewidth=1.5)

    # Annotations - 用 LaTeX 公式标注 δb、δMSE 和梯度近似值
    axs[1].annotate(r'$\delta b = {:.2f}$'.format(delta_b), xy=(.67, 2.7), c='k', fontsize=15)
    axs[1].annotate(r'$\delta MSE = {:.2f}$'.format(delta_mse_b), xy=(.45, 2.32), c='k', fontsize=15)
    axs[1].annotate(r'$\frac{\delta MSE}{\delta b} \approx' + '{:.2f}$'.format(manual_grad_b), xy=(.62, 2.6), c='k', fontsize=17)

    axs[1].label_outer()

    fig.tight_layout()
    return fig, axs


def figure9(x_train, y_train, b, w):
    # 对比初始模型（随机参数）和更新一次后的模型在数据上的拟合效果
    # 展示梯度下降的一步更新如何改善模型预测
    # Since we updated b and w, let's regenerate the initial ones
    # That's how using a random seed is useful, for instance
    # 固定随机种子重新生成初始参数（与训练开始时一致），用于对比
    np.random.seed(42)
    b_initial = np.random.randn(1)
    w_initial = np.random.randn(1)

    # 绘制初始随机参数的拟合线（黑色虚线）
    fig, ax = figure2(x_train, y_train, b_initial, w_initial)

    # Generates evenly spaced x feature
    # 生成均匀分布的 x 值用于绘制预测线
    x_range = np.linspace(0, 1, 101)
    # Model's predictions for updated paramaters
    # 用更新后的参数计算预测值
    yhat_range = b + w * x_range
    # Updated predictions
    # 绘制更新一次后的预测线（绿色虚线），与初始模型形成对比
    ax.plot(x_range, yhat_range, label='Using parameters\nafter one update', c='g', linestyle='--')
    # Annotations
    # 标注更新后的参数值
    ax.annotate('b = {:.4f} w = {:.4f}'.format(b[0], w[0]), xy=(.2, .95), c='g')

    fig.tight_layout()
    return fig, ax


def figure10(b, w, bs, ws, all_losses, manual_grad_b, manual_grad_w, lr):
    # 展示使用梯度下降更新规则（θ_new = θ_old - η * grad）后参数的变化
    # 学习率 η 控制每次更新的步长，是梯度下降的关键超参数
    b_range = bs[0, :]
    w_range = ws[:, 0]

    # BEFORE
    # 获取更新前参数在网格中的索引和实际值
    b_idx, w_idx, bs_before, ws_before = find_index(b, w, bs, ws)
    # AFTER
    # 根据梯度下降更新规则计算新参数位置：θ_new = θ_old - η * ∇θ
    new_b_idx, new_w_idx, bs_after, ws_after = find_index(bs_before - lr * manual_grad_b,
                                                          ws_before - lr * manual_grad_w,
                                                          bs,
                                                          ws)
    # Loss before - 更新前的损失值
    loss_before = all_losses[w_idx, b_idx]
    # 固定 b 时损失随 w 变化的曲线
    loss_fixedb = all_losses[:, b_idx]
    # 固定 w 时损失随 b 变化的曲线
    loss_fixedw = all_losses[w_idx, :]
    # 更新 b 后的损失值
    loss_after_b = all_losses[w_idx, new_b_idx]
    # 更新 w 后的损失值
    loss_after_w = all_losses[new_w_idx, b_idx]

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # 左子图：w 方向的损失曲线，展示梯度下降沿 w 方向的更新
    axs[0].set_ylim([-.1, 6.1])
    axs[0].set_xlabel('w')
    axs[0].set_ylabel('MSE (loss)')
    axs[0].set_title('Fixed: b = {:.2f}'.format(bs_before))

    # Loss curve for w, given fixed b
    axs[0].plot(w_range, loss_fixedb, c='r', linestyle='--', linewidth=2)
    # w before update - 更新前的参数位置（红色圆点）
    axs[0].plot([ws_before], [loss_before], 'or')

    # Arrows - 用箭头和线段展示参数从旧位置移动到新位置
    axs[0].arrow(ws_after, loss_before, .1, 0, color='r', shape='full', lw=0, length_includes_head=True, head_width=.1)
    axs[0].plot([ws_before, ws_after], [loss_before, loss_before], 'r-', linewidth=1.5)
    axs[0].plot([ws_after], [loss_after_w], 'or')

    # Annotations - 标注学习率 η 和更新量 -η * grad
    axs[0].annotate(r'$\eta = {:.2f}$'.format(lr), xy=(1.6, 5.5), c='k', fontsize=17)
    axs[0].annotate(r'$-\eta \frac{\delta MSE}{\delta b} \approx' + '{:.2f}$'.format(-lr * manual_grad_w), xy=(1, 2), c='k', fontsize=17)

    # 右子图：b 方向的损失曲线，展示梯度下降沿 b 方向的更新
    axs[1].set_ylim([-.1, 6.1])
    axs[1].set_xlabel('b')
    axs[1].set_ylabel('MSE (loss)')
    axs[1].set_title('Fixed: w = {:.2f}'.format(ws_before))
    axs[1].label_outer()

    # Loss curve for b, given fixed w
    axs[1].plot(b_range, loss_fixedw, c='k', linestyle='--', linewidth=2)
    # b before update - 更新前的参数位置（黑色圆点）
    axs[1].plot([bs_before], [loss_before], 'ok')

    # Arrows - 用箭头和线段展示参数从旧位置移动到新位置
    axs[1].arrow(bs_after, loss_before, .1, 0, color='k', shape='full', lw=0, length_includes_head=True, head_width=.1)
    axs[1].plot([bs_before, bs_after], [loss_before, loss_before], 'k-', linewidth=1.5)
    axs[1].plot([bs_after], [loss_after_b], 'ok')

    # Annotations - 标注学习率 η 和更新量 -η * grad
    axs[1].annotate(r'$\eta = {:.2f}$'.format(lr), xy=(0.6, 5.5), c='k', fontsize=17)
    axs[1].annotate(r'$-\eta \frac{\delta MSE}{\delta w} \approx' + '{:.2f}$'.format(-lr * manual_grad_b), xy=(1, 2), c='k', fontsize=17)

    fig.tight_layout()
    return fig, axs


def figure14(x_train, y_train, b, w, bad_bs, bad_ws, bad_x_train):
    # 对比特征尺度对损失曲面形状的影响：原始数据 vs "坏"尺度数据
    # 当特征 x 的尺度差异很大时，损失曲面的等高线会变得狭长（扁平），影响梯度下降的效率
    bad_b_range = bad_bs[0, :]
    bad_w_range = bad_ws[:, 0]

    # So we recompute the surface for X_TRAIN using the new ranges
    # 使用新的参数网格重新计算原始训练数据的损失曲面
    # np.apply_along_axis 沿第1轴（每个样本）应用线性模型计算预测值
    all_predictions = np.apply_along_axis(func1d=lambda x: bad_bs + bad_ws * x, axis=1, arr=x_train)
    # 计算每个样本在所有参数组合下的预测误差
    all_errors = (all_predictions - y_train.reshape(-1, 1, 1))
    # 计算 MSE 损失：所有样本误差平方的均值
    all_losses = (all_errors ** 2).mean(axis=0)

    # Then we compute the surface for BAD_X_TRAIN using the new ranges
    # 使用相同参数网格计算"坏"尺度数据的损失曲面
    bad_all_predictions = np.apply_along_axis(func1d=lambda x: bad_bs + bad_ws * x, axis=1, arr=bad_x_train)
    bad_all_errors = (bad_all_predictions - y_train.reshape(-1, 1, 1))
    bad_all_losses = (bad_all_errors ** 2).mean(axis=0)

    # 当前随机初始参数在网格中的位置
    b_idx, w_idx, fixedb, fixedw = find_index(b, w, bad_bs, bad_ws)

    # 原始数据对应的最优参数（解析解）
    b_minimum, w_minimum = fit_model(x_train, y_train)

    # "坏"尺度数据对应的最优参数（解析解）
    bad_b_minimum, bad_w_minimum = fit_model(bad_x_train, y_train)

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # 左子图：原始数据的损失曲面等高线（BEFORE）
    axs[0].set_xlabel('b')
    axs[0].set_ylabel('w')
    axs[0].set_title('Loss Surface - Before')

    # Loss surface - BEFORE - 原始数据的损失等高线
    CS = axs[0].contour(bad_bs[0, :], bad_ws[:, 0], all_losses, cmap=plt.cm.jet)
    axs[0].clabel(CS, inline=1, fontsize=10)
    # Minimum point - BEFORE - 原始数据的最优点
    axs[0].scatter(b_minimum, w_minimum, c='k')
    # Initial random point - 随机初始点
    axs[0].scatter(fixedb, fixedw, c='k')

    # Vertical cross section - 垂直截面线（固定 b）
    axs[0].plot([fixedb, fixedb], bad_w_range[[0, -1]], linestyle='--', c='r', linewidth=2)
    # Horizontal cross section - 水平截面线（固定 w）
    axs[0].plot(bad_b_range[[0, -1]], [fixedw, fixedw], linestyle='--', c='k', linewidth=2)

    # Annotations
    axs[0].annotate('Minimum', xy=(.5, .35), c='k')
    axs[0].annotate('Random Start', xy=(fixedb - .6, fixedw - .3), c='k')

    # 右子图："坏"尺度数据的损失曲面等高线（AFTER），等高线通常更狭长
    axs[1].set_xlabel('b')
    axs[1].set_ylabel('w')
    axs[1].set_title('Loss Surface - After')

    # Loss surface - AFTER - "坏"尺度数据的损失等高线
    CS = axs[1].contour(bad_bs[0, :], bad_ws[:, 0], bad_all_losses, cmap=plt.cm.jet)
    axs[1].clabel(CS, inline=1, fontsize=10)
    # Minimum point - AFTER - "坏"尺度数据的最优点
    axs[1].scatter(bad_b_minimum, bad_w_minimum, c='k')
    # Initial random point - 随机初始点（位置不变）
    axs[1].scatter(fixedb, fixedw, c='k')

    # Vertical cross section
    axs[1].plot([fixedb, fixedb], bad_w_range[[0, -1]], linestyle='--', c='r', linewidth=2)
    # Horizontal cross section
    axs[1].plot(bad_b_range[[0, -1]], [fixedw, fixedw], linestyle='--', c='k', linewidth=2)

    # Annotations
    axs[1].annotate('Minimum', xy=(.5, .35), c='k')
    axs[1].annotate('Random Start', xy=(fixedb - .6, fixedw - .3), c='k')

    fig.tight_layout()
    return fig, axs


def figure15(x_train, y_train, b, w, bad_bs, bad_ws, bad_x_train):
    # 展示特征尺度变化对损失曲线形状的影响（截面视图）
    # 对比原始数据和"坏"尺度数据在固定 b 或固定 w 时损失曲线的差异
    bad_b_range = bad_bs[0, :]
    bad_w_range = bad_ws[:, 0]

    # So we recompute the surface for X_TRAIN using the new ranges
    # 重新计算原始训练数据的损失曲面
    all_predictions = np.apply_along_axis(func1d=lambda x: bad_bs + bad_ws * x, axis=1, arr=x_train)
    all_errors = (all_predictions - y_train.reshape(-1, 1, 1))
    all_losses = (all_errors ** 2).mean(axis=0)

    # Then we compute the surface for BAD_X_TRAIN using the new ranges
    # 重新计算"坏"尺度训练数据的损失曲面
    bad_all_predictions = np.apply_along_axis(func1d=lambda x: bad_bs + bad_ws * x, axis=1, arr=bad_x_train)
    bad_all_errors = (bad_all_predictions - y_train.reshape(-1, 1, 1))
    bad_all_losses = (bad_all_errors ** 2).mean(axis=0)

    # 当前参数在网格中的位置
    bad_b_idx, bad_w_idx, fixedb, fixedw = find_index(b, w, bad_bs, bad_ws)

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # 左子图：固定 b，对比原始数据和"坏"数据沿 w 方向的损失曲线
    axs[0].set_ylim([-.1, 15.1])
    axs[0].set_xlim([-1, 3.2])
    axs[0].set_xlabel('w')
    axs[0].set_ylabel('Loss')
    axs[0].set_title('Fixed: b = {:.2f}'.format(fixedb))

    # Loss curve for b, given fixed w - BEFORE - 原始数据的 w 方向损失曲线（细虚线）
    axs[0].plot(bad_w_range, all_losses[:, bad_b_idx], c='r', linestyle='--', linewidth=1, label='Before')
    axs[0].plot([fixedw], [all_losses[bad_w_idx, bad_b_idx]], 'or')
    # Loss curve for b, given fixed w - AFTER - "坏"数据的 w 方向损失曲线（粗虚线）
    axs[0].plot(bad_w_range, bad_all_losses[:, bad_b_idx], c='r', linestyle='--', linewidth=2, label='After')
    axs[0].plot([fixedw], [bad_all_losses[bad_w_idx, bad_b_idx]], 'or')

    axs[0].legend()

    # 右子图：固定 w，对比原始数据和"坏"数据沿 b 方向的损失曲线
    axs[1].set_ylim([-.1, 15.1])
    axs[1].set_xlabel('b')
    axs[1].set_ylabel('Loss')
    axs[1].set_title('Fixed: w = {:.2f}'.format(fixedw))

    # Loss curve for w, given fixed b - BEFORE - 原始数据的 b 方向损失曲线（细虚线）
    axs[1].plot(bad_b_range, all_losses[bad_w_idx, :], c='k', linestyle='--', linewidth=1, label='Before')
    axs[1].plot([fixedb], [all_losses[bad_w_idx, bad_b_idx]], 'ok')
    # Loss curve for w, given fixed b - AFTER - "坏"数据的 b 方向损失曲线（粗虚线）
    axs[1].plot(bad_b_range, bad_all_losses[bad_w_idx, :], c='k', linestyle='--', linewidth=2, label='After')
    axs[1].plot([fixedb], [bad_all_losses[bad_w_idx, bad_b_idx]], 'ok')

    axs[1].legend()

    fig.tight_layout()
    return fig, axs


def figure17(x_train, y_train, scaled_bs, scaled_ws, bad_x_train, scaled_x_train):
    # 三路对比：原始数据 vs "坏"尺度数据 vs 标准化后数据的损失曲面
    # 展示特征标准化（Z-score normalization）如何使损失曲面等高线更接近圆形，从而加速梯度下降收敛
    # So we recompute the surface for X_TRAIN using the new ranges
    # 计算原始数据的损失曲面
    all_predictions = np.apply_along_axis(func1d=lambda x: scaled_bs + scaled_ws * x, axis=1, arr=x_train)
    all_errors = (all_predictions - y_train.reshape(-1, 1, 1))
    all_losses = (all_errors ** 2).mean(axis=0)

    # So we recompute the surface for BAD_X_TRAIN using the new ranges
    # 计算"坏"尺度数据的损失曲面
    bad_all_predictions = np.apply_along_axis(func1d=lambda x: scaled_bs + scaled_ws * x, axis=1, arr=bad_x_train)
    bad_all_errors = (bad_all_predictions - y_train.reshape(-1, 1, 1))
    bad_all_losses = (bad_all_errors ** 2).mean(axis=0)

    # Then we compute the surface for SCALED_X_TRAIN using the new ranges
    # 计算标准化后数据的损失曲面
    scaled_all_predictions = np.apply_along_axis(func1d=lambda x: scaled_bs + scaled_ws * x, axis=1, arr=scaled_x_train)
    scaled_all_errors = (scaled_all_predictions - y_train.reshape(-1, 1, 1))
    scaled_all_losses = (scaled_all_errors ** 2).mean(axis=0)

    # 各数据集对应的解析最优解
    b_minimum, w_minimum = fit_model(x_train, y_train)

    bad_b_minimum, bad_w_minimum = fit_model(bad_x_train, y_train)

    scaled_b_minimum, scaled_w_minimum = fit_model(scaled_x_train, y_train)

    fig, axs = plt.subplots(1, 3, figsize=(15, 6))

    # 左子图：原始数据的损失曲面（等高线通常较圆，优化容易）
    axs[0].set_xlabel('b')
    axs[0].set_ylabel('w')
    axs[0].set_title('Loss Surface - Original')

    # Loss Surface - ORIGINAL
    CS = axs[0].contour(scaled_bs[0, :], scaled_ws[:, 0], all_losses, cmap=plt.cm.jet)
    axs[0].clabel(CS, inline=1, fontsize=10)
    # Minimum point - ORIGINAL
    axs[0].scatter(b_minimum, w_minimum, c='k')

    # Annotations
    axs[0].annotate('Minimum', xy=(.3, 1.6), c='k')

    # 中间子图："坏"尺度数据的损失曲面（等高线狭长，优化困难）
    axs[1].set_xlabel('b')
    axs[1].set_ylabel('w')
    axs[1].set_title('Loss Surface - "Bad"')

    # Loss Surface - BAD
    CS = axs[1].contour(scaled_bs[0, :], scaled_ws[:, 0], bad_all_losses, cmap=plt.cm.jet)
    axs[1].clabel(CS, inline=1, fontsize=10)
    # Minimum point - BAD
    axs[1].scatter(bad_b_minimum, bad_w_minimum, c='k')

    # Annotations
    axs[1].annotate('Minimum', xy=(.3, -.25), c='k')

    # 右子图：标准化后数据的损失曲面（等高线恢复近圆形，优化效率恢复）
    axs[2].set_xlabel('b')
    axs[2].set_ylabel('w')
    axs[2].set_title('Loss Surface - Scaled')

    # Loss Surface - SCALED
    CS = axs[2].contour(scaled_bs[0, :], scaled_ws[:, 0], scaled_all_losses, cmap=plt.cm.jet)
    axs[2].clabel(CS, inline=1, fontsize=10)
    # Minimum point - SCALED
    axs[2].scatter(scaled_b_minimum, scaled_w_minimum, c='k')

    # Annotations
    axs[2].annotate('Minimum', xy=(1.3, .15), c='k')

    fig.tight_layout()
    return fig, axs


def figure18(x_train, y_train):
    # 展示最终模型的预测结果：使用解析最优参数（而非梯度下降的中间结果）
    # 该函数通常在训练结束后调用，展示模型最终的拟合效果
    b_minimum, w_minimum = fit_model(x_train, y_train)
    # Generates evenly spaced x feature
    # 生成均匀分布的 x 值用于绘制连续预测线
    x_range = np.linspace(0, 1, 101)
    # Computes yhat
    # 使用最优参数计算预测值
    yhat_range = b_minimum + w_minimum * x_range

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_ylim([0, 3.1])

    # Dataset
    # 绘制训练数据散点图
    ax.scatter(x_train, y_train)
    # Predictions
    # 绘制最终模型的预测线（黑色虚线）
    ax.plot(x_range, yhat_range, label='Final model\'s predictions', c='k', linestyle='--')

    # Annotations
    # 标注最优参数值，rotation=34 使文字沿预测线方向倾斜
    ax.annotate('b = {:.4f} w = {:.4f}'.format(b_minimum, w_minimum), xy=(.4, 1.5), c='k', rotation=34)
    ax.legend(loc=0)
    fig.tight_layout()
    return fig, ax
