# ============================================================
# 基础库导入
# ============================================================
import numpy as np                     # 数值计算库
import datetime                        # 日期时间处理
import torch                           # PyTorch 深度学习框架
import torch.nn as nn                  # 神经网络模块
import random                          # Python 随机数模块
import matplotlib.pyplot as plt        # 绘图库
from torch.utils.tensorboard import SummaryWriter  # TensorBoard 日志记录器

# 设置 matplotlib 绘图风格
plt.style.use('fivethirtyeight')

class StepByStep(object):
    """
    逐步训练封装类

    将模型训练、验证、可视化等流程封装在一起，简化 PyTorch 训练过程。
    支持 TensorBoard 日志、模型检查点、钩子函数可视化等功能。
    """
    def __init__(self, model, loss_fn, optimizer):
        # ============================================================
        # 将构造参数保存为实例属性，供后续方法使用
        # ============================================================
        self.model = model               # 神经网络模型
        self.loss_fn = loss_fn           # 损失函数
        self.optimizer = optimizer       # 优化器
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'  # 自动检测 GPU/CPU
        # 立即将模型移动到指定设备
        self.model.to(self.device)

        # ============================================================
        # 以下属性在初始化时暂不赋值，后续通过方法设置
        # ============================================================
        self.train_loader = None         # 训练数据加载器
        self.val_loader = None           # 验证数据加载器
        self.writer = None               # TensorBoard SummaryWriter

        # ============================================================
        # 内部计算属性（由 train 方法自动维护）
        # ============================================================
        self.losses = []                 # 训练损失历史
        self.val_losses = []             # 验证损失历史
        self.total_epochs = 0            # 累计训练 epoch 数

        # 可视化相关属性
        self.visualization = {}          # 存储钩子捕获的层输出
        self.handles = {}                # 存储钩子句柄（用于移除）

        # ============================================================
        # 创建训练步和验证步函数（闭包）
        # 注意：这些函数无需外部参数，直接使用类的属性
        # ============================================================
        self.train_step_fn = self._make_train_step_fn()  # 单步训练函数
        self.val_step_fn = self._make_val_step_fn()      # 单步验证函数

    def to(self, device):
        """
        将模型切换到指定设备（GPU/CPU）

        如果切换失败，自动回退到默认设备并给出提示。
        """
        try:
            self.device = device
            self.model.to(self.device)
        except RuntimeError:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            print(f"Couldn't send it to {device}, sending it to {self.device} instead.")
            self.model.to(self.device)

    def set_loaders(self, train_loader, val_loader=None):
        """
        设置训练和验证数据加载器

        Args:
            train_loader: 训练数据的 DataLoader
            val_loader: 验证数据的 DataLoader（可选）
        """
        self.train_loader = train_loader
        self.val_loader = val_loader

    def set_tensorboard(self, name, folder='runs'):
        """
        初始化 TensorBoard 日志记录器

        Args:
            name: 日志名称
            folder: 日志保存目录，默认为 'runs'
        """
        suffix = datetime.datetime.now().strftime('%Y%m%d%H%M%S')  # 时间戳后缀
        self.writer = SummaryWriter(f'{folder}/{name}_{suffix}')

    def _make_train_step_fn(self):
        """
        构造单步训练函数（闭包）

        闭包直接引用 self.model、self.loss_fn 和 self.optimizer，
        无需外部传入参数。

        Returns:
            perform_train_step_fn: 执行单步训练的函数
        """
        # 构建在训练循环中执行单步操作的函数
        def perform_train_step_fn(x, y):
            # 将模型设为训练模式（启用 Dropout、BatchNorm 等）
            self.model.train()

            # 第1步 — 前向传播：计算模型预测值
            yhat = self.model(x)
            # 第2步 — 计算损失
            loss = self.loss_fn(yhat, y)
            # 第3步 — 反向传播：计算所有参数的梯度
            loss.backward()
            # 第4步 — 更新参数（利用梯度 & 学习率）
            self.optimizer.step()
            self.optimizer.zero_grad()   # 清零梯度，避免累积

            # 返回损失值（item() 将标量张量转为 Python 数值）
            return loss.item()

        # 返回闭包函数，供训练循环调用
        return perform_train_step_fn

    def _make_val_step_fn(self):
        """
        构造单步验证函数（闭包）

        与训练步不同，验证步不需要计算梯度和更新参数。

        Returns:
            perform_val_step_fn: 执行单步验证的函数
        """
        # 构建在验证循环中执行单步操作的函数
        def perform_val_step_fn(x, y):
            # 将模型设为评估模式（禁用 Dropout、固定 BatchNorm 参数）
            self.model.eval()

            # 第1步 — 前向传播：计算模型预测值
            yhat = self.model(x)
            # 第2步 — 计算损失
            loss = self.loss_fn(yhat, y)
            # 验证阶段无需第3、4步（不更新参数）
            return loss.item()

        return perform_val_step_fn

    def _mini_batch(self, validation=False):
        """
        在完整数据集上执行一次 mini-batch 遍历

        Args:
            validation: 若为 True 使用验证集和 val_step_fn；
                       若为 False 使用训练集和 train_step_fn

        Returns:
            loss: 整个数据集的平均损失；若未设置 loader 则返回 None
        """
        # 根据 validation 标志选择对应的数据加载器和步函数
        if validation:
            data_loader = self.val_loader
            step_fn = self.val_step_fn
        else:
            data_loader = self.train_loader
            step_fn = self.train_step_fn

        # 若未设置数据加载器，直接返回
        if data_loader is None:
            return None

        # 标准的 mini-batch 循环
        mini_batch_losses = []
        for x_batch, y_batch in data_loader:
            # 将当前 batch 的数据发送到指定设备
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            # 执行一步（训练或验证）
            mini_batch_loss = step_fn(x_batch, y_batch)
            mini_batch_losses.append(mini_batch_loss)

        # 返回所有 batch 损失的平均值
        loss = np.mean(mini_batch_losses)
        return loss

    def set_seed(self, seed=42):
        """
        设置随机种子，确保实验可复现

        涵盖 PyTorch、NumPy、Python random、cuDNN 以及
        DataLoader 采样器的随机种子。

        Args:
            seed: 随机种子值，默认 42
        """
        # 设置 cuDNN 为确定性模式（可能略微降低性能）
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # 设置各库的随机种子
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        # 尝试设置 DataLoader 采样器的种子
        try:
            self.train_loader.sampler.generator.manual_seed(seed)
        except AttributeError:
            pass

    def train(self, n_epochs, seed=42):
        """
        执行模型训练

        Args:
            n_epochs: 训练轮数
            seed: 随机种子，确保可复现性，默认 42
        """
        # 设置随机种子，保证训练过程可复现
        self.set_seed(seed)

        for epoch in range(n_epochs):
            # 更新累计 epoch 计数器
            self.total_epochs += 1

            # ========== 训练阶段 ==========
            # 使用 mini-batch 进行训练
            loss = self._mini_batch(validation=False)
            self.losses.append(loss)

            # ========== 验证阶段 ==========
            # 验证过程不计算梯度（节省显存，加速计算）
            with torch.no_grad():
                # 使用 mini-batch 进行评估
                val_loss = self._mini_batch(validation=True)
                self.val_losses.append(val_loss)

            # ========== TensorBoard 日志记录 ==========
            if self.writer:
                scalars = {'training': loss}
                if val_loss is not None:
                    scalars.update({'validation': val_loss})
                # 在每个 epoch 记录训练和验证损失
                self.writer.add_scalars(main_tag='loss',
                                        tag_scalar_dict=scalars,
                                        global_step=epoch)

        # 训练结束后关闭 SummaryWriter
        if self.writer:
            self.writer.close()

    def save_checkpoint(self, filename):
        """
        保存模型检查点（用于中断后恢复训练）

        保存内容包括：epoch 数、模型参数、优化器状态、损失历史

        Args:
            filename: 检查点文件保存路径
        """
        # 构建包含所有恢复训练所需元素的字典
        checkpoint = {'epoch': self.total_epochs,
                      'model_state_dict': self.model.state_dict(),
                      'optimizer_state_dict': self.optimizer.state_dict(),
                      'loss': self.losses,
                      'val_loss': self.val_losses}

        torch.save(checkpoint, filename)

    def load_checkpoint(self, filename):
        """
        加载模型检查点，恢复训练状态

        恢复：模型参数、优化器状态、epoch 计数、损失历史

        Args:
            filename: 检查点文件路径
        """
        # 加载检查点字典
        checkpoint = torch.load(filename, weights_only=False)

        # 恢复模型和优化器的状态
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # 恢复训练进度
        self.total_epochs = checkpoint['epoch']
        self.losses = checkpoint['loss']
        self.val_losses = checkpoint['val_loss']

        # 恢复后始终设为训练模式
        self.model.train()

    def predict(self, x):
        """
        使用模型进行预测

        Args:
            x: NumPy 数组格式的输入数据

        Returns:
            NumPy 数组格式的预测结果
        """
        # 设为评估模式
        self.model.eval()
        # 将 NumPy 输入转为 float 张量
        x_tensor = torch.as_tensor(x).float()
        # 将输入发送到设备上进行预测
        y_hat_tensor = self.model(x_tensor.to(self.device))
        # 恢复训练模式
        self.model.train()
        # 将张量从计算图中分离 → 移至 CPU → 转回 NumPy
        return y_hat_tensor.detach().cpu().numpy()

    def plot_losses(self):
        """
        绘制训练和验证损失曲线

        使用对数坐标轴，便于观察损失的下降趋势。

        Returns:
            matplotlib Figure 对象
        """
        fig = plt.figure(figsize=(10, 4))
        plt.plot(self.losses, label='Training Loss', c='b')       # 蓝色：训练损失
        plt.plot(self.val_losses, label='Validation Loss', c='r') # 红色：验证损失
        plt.yscale('log')              # 使用对数尺度使变化更明显
        plt.xlabel('Epochs')           # X轴：训练轮数
        plt.ylabel('Loss')             # Y轴：损失值
        plt.legend()                   # 显示图例
        plt.tight_layout()             # 自动调整布局
        return fig

    def add_graph(self):
        """
        将模型计算图添加到 TensorBoard

        取一个 mini-batch 样本，通过 add_graph 记录模型结构。
        """
        if self.train_loader and self.writer:
            x_sample, y_sample = next(iter(self.train_loader))  # 取一个 batch
            self.writer.add_graph(self.model, x_sample.to(self.device))

    def count_parameters(self):
        """
        统计模型中需要训练的参数总数

        Returns:
            所有可训练参数的张量元素总数
        """
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    @staticmethod
    def _visualize_tensors(axs, x, y=None, yhat=None, layer_name='', title=None):
        """
        在指定子图上可视化张量（作为灰度图）

        Args:
            axs: matplotlib 子图数组
            x: 要可视化的张量数据
            y: 真实标签（可选）
            yhat: 预测标签（可选）
            layer_name: 图层名称
            title: 图像标题前缀
        """
        # 一行中子图的数量即为要显示的图像数量
        n_images = len(axs)
        # 获取最大值和最小值，用于统一灰度图的色彩范围
        minv, maxv = np.min(x[:n_images]), np.max(x[:n_images])
        # 逐张显示
        for j, image in enumerate(x[:n_images]):
            ax = axs[j]
            # 设置标题、标签，并移除刻度
            if title is not None:
                ax.set_title('{} #{}'.format(title, j), fontsize=12)
            ax.set_ylabel(
                '{}\n{}x{}'.format(layer_name, *np.atleast_2d(image).shape),
                rotation=0, labelpad=40
            )
            # 拼接真实标签和预测标签
            xlabel1 = '' if y is None else '\nLabel: {}'.format(y[j])
            xlabel2 = '' if yhat is None else '\nPredicted: {}'.format(yhat[j])
            xlabel = '{}{}'.format(xlabel1, xlabel2)
            if len(xlabel):
                ax.set_xlabel(xlabel, fontsize=12)
            ax.set_xticks([])
            ax.set_yticks([])

            # 将张量绘制为灰度图像
            ax.imshow(
                np.atleast_2d(image.squeeze()),
                cmap='gray',
                vmin=minv,
                vmax=maxv
            )
        return

    def visualize_filters(self, layer_name, **kwargs):
        """
        可视化卷积层的滤波器权重

        Args:
            layer_name: 层的名称（支持嵌套，如 'conv1' 或 'features.0'）

        Returns:
            matplotlib Figure 对象；若指定层不是 Conv2d 则返回 None
        """
        try:
            # 根据名称逐级获取层对象
            layer = self.model
            for name in layer_name.split('.'):
                layer = getattr(layer, name)
            # 仅支持 2D 卷积层
            if isinstance(layer, nn.Conv2d):
                # 提取权重（形状：[输出通道, 输入通道, H, W]）
                weights = layer.weight.data.cpu().numpy()
                n_filters, n_channels, _, _ = weights.shape

                # 创建画布：每行一个输入通道，每列一个滤波器
                size = (2 * n_channels + 2, 2 * n_filters)
                fig, axes = plt.subplots(n_filters, n_channels, figsize=size)
                axes = np.atleast_2d(axes).reshape(n_filters, n_channels)
                # 遍历每个滤波器（输出通道）
                for i in range(n_filters):
                    StepByStep._visualize_tensors(
                        axes[i, :],
                        weights[i],
                        layer_name='Filter #{}'.format(i),
                        title='Channel' if (i == 0) else None
                    )

                # 对外层子图只保留最外侧标签
                for ax in axes.flat:
                    ax.label_outer()

                fig.tight_layout()
                return fig
        except AttributeError:
            return

    def attach_hooks(self, layers_to_hook, hook_fn=None):
        """
        为指定层注册前向传播钩子

        钩子会捕获层的输出并存储到 self.visualization 字典中，
        供后续可视化使用。

        Args:
            layers_to_hook: 需要挂载钩子的层名称列表
            hook_fn: 自定义钩子函数；若为 None 则使用默认钩子
        """
        # 清空之前的可视化数据
        self.visualization = {}
        # 创建 层对象 → 层名称 的映射字典
        modules = list(self.model.named_modules())
        layer_names = {layer: name for name, layer in modules[1:]}  # [1:] 跳过模型本身

        if hook_fn is None:
            # 默认钩子函数：捕获前向传播的输出
            def hook_fn(layer, inputs, outputs):
                # 通过映射字典获取层名称
                name = layer_names[layer]
                # 将输出从计算图中分离并转为 NumPy
                values = outputs.detach().cpu().numpy()
                # 如果钩子被多次调用（多个 mini-batch），则拼接结果
                if self.visualization[name] is None:
                    self.visualization[name] = values
                else:
                    self.visualization[name] = np.concatenate([self.visualization[name], values])

        for name, layer in modules:
            # 如果该层在目标列表中
            if name in layers_to_hook:
                # 初始化字典中对应的键
                self.visualization[name] = None
                # 注册前向钩子并保留句柄（用于后续移除）
                self.handles[name] = layer.register_forward_hook(hook_fn)

    def remove_hooks(self):
        """
        移除所有已注册的前向钩子

        释放钩子占用的资源，避免内存泄漏。
        """
        # 遍历所有句柄并移除
        for handle in self.handles.values():
            handle.remove()
        # 清空句柄字典
        self.handles = {}

    def visualize_outputs(self, layers, n_images=10, y=None, yhat=None):
        """
        可视化指定层的输出特征图

        Args:
            layers: 要可视化的层名称列表（必须已通过 attach_hooks 捕获）
            n_images: 显示图像的数量，默认 10
            y: 真实标签（可选）
            yhat: 预测标签（可选）

        Returns:
            matplotlib Figure 对象
        """
        # 过滤出已捕获到数据的层
        layers = list(filter(lambda l: l in self.visualization.keys(), layers))
        # 计算每层的输出通道数
        shapes = [self.visualization[layer].shape for layer in layers]
        n_rows = [shape[1] if len(shape) == 4 else 1 for shape in shapes]
        total_rows = np.sum(n_rows)

        # 创建画布：总行数 = 所有输出通道数之和，列数 = 图像数量
        fig, axes = plt.subplots(total_rows, n_images, figsize=(1.5*n_images, 1.5*total_rows))
        axes = np.atleast_2d(axes).reshape(total_rows, n_images)

        # 逐层遍历，每层占用画布的一行或多行
        row = 0
        for i, layer in enumerate(layers):
            start_row = row
            # 获取该层的输出特征图
            output = self.visualization[layer]

            # 判断是否为向量形式（如全连接层的输出）
            is_vector = len(output.shape) == 2

            for j in range(n_rows[i]):
                StepByStep._visualize_tensors(
                    axes[row, :],
                    output if is_vector else output[:, j].squeeze(),
                    y,
                    yhat,
                    layer_name=layers[i] if is_vector else '{}\nfil#{}'.format(layers[i], row-start_row),
                    title='Image' if (row == 0) else None
                )
                row += 1

        # 对外层子图只保留最外侧标签
        for ax in axes.flat:
            ax.label_outer()

        plt.tight_layout()
        return fig

    def correct(self, x, y, threshold=.5):
        """
        统计模型在每个类别上的正确预测数

        自动处理多分类和二元分类两种场景。
        对于二元分类，会检测最后一层是否为 Sigmoid 来决定如何计算预测值。

        Args:
            x: 输入张量
            y: 真实标签张量
            threshold: 二元分类的阈值，默认 0.5

        Returns:
            形状为 (n_classes, 2) 的张量，每行是 [正确数, 总数]
        """
        self.model.eval()
        yhat = self.model(x.to(self.device))
        y = y.to(self.device)
        self.model.train()

        # 获取 batch 大小和类别数（二元分类时 n_dims=1）
        n_samples, n_dims = yhat.shape
        if n_dims > 1:
            # 多分类：直接取最大 logit 对应的类别
            # torch.max 返回元组：(最大值, 最大值索引)
            _, predicted = torch.max(yhat, 1)
        else:
            n_dims += 1
            # 二元分类：需要判断最后一层是否为 Sigmoid
            if isinstance(self.model, nn.Sequential) and \
               isinstance(self.model[-1], nn.Sigmoid):
                # Sigmoid 层已输出概率，直接与阈值比较
                predicted = (yhat > threshold).long()
            else:
                # 输出为 logits，需要使用 sigmoid 转换为概率后比较
                predicted = (torch.sigmoid(yhat) > threshold).long()

        # 按类别统计正确预测数
        result = []
        for c in range(n_dims):
            n_class = (y == c).sum().item()                        # 该类的样本总数
            n_correct = (predicted[y == c] == c).sum().item()      # 该类正确预测数
            result.append((n_correct, n_class))
        return torch.tensor(result)

    @staticmethod
    def loader_apply(loader, func, reduce='sum'):
        """
        对整个 DataLoader 中的数据批量应用函数，并汇总结果

        Args:
            loader: 数据加载器
            func: 应用的函数，接收 (x, y) 返回一个张量
            reduce: 汇总方式，'sum' 求和 或 'mean' 求平均，默认为 'sum'

        Returns:
            汇总后的结果张量
        """
        # 对每个 batch 应用函数
        results = [func(x, y) for i, (x, y) in enumerate(loader)]
        # 沿第0维堆叠所有结果
        results = torch.stack(results, axis=0)

        # 按指定方式汇总
        if reduce == 'sum':
            results = results.sum(axis=0)     # 求和汇总
        elif reduce == 'mean':
            results = results.float().mean(axis=0)  # 求平均汇总

        return results
