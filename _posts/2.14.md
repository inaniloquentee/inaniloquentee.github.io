---

# 从 0 到 1：PINN 的计算图抽取与 GitHub 贡献实战

在参与 PaddlePaddle 启航计划的过程中，我完成了热身打卡：为科研的流体力学模型 **PIGU_Hybrid** 抽取并提交 GraphNet 计算图。这篇文章记录了从环境报错、编译器维度对齐，到最终解决 CI 代码规范检查的完整技术细节。

## 1. 核心技术背景：AI Infra 与 GraphNet

作为 **AI Infra**（AI 基础设施）的一部分，**GraphNet** 旨在通过自动化的方式从原生框架（如 PyTorch）中“录制”模型的计算逻辑，生成标准化的计算图。这对于模型优化、跨平台部署以及流体力学（如 PINNs, URANS）等科学计算模型的标准化具有重要意义。

## 2. 计算图抽取：PIGU_Hybrid 模型实战

在抽取过程中，最头疼的是处理 **TorchDynamo** 编译器的符号追踪（Symbolic Tracing）限制。

### 遇到的挑战

* **维度不匹配**：编译器无法自动推导出动态网格（如 ）与输入尺寸（）在数学上的等价性。
* **现象**：编译器在处理网格维度时，将其标记为动态符号（例如  和 ），而不是具体的数字 **128** 和 **256**。
* **痛点**：虽然在数学上我们知道 ，但编译器在没有显式证据的情况下，无法“证明”这些动态符号在运行时一定会等于输入的固定尺寸。这导致在进行张量拼接（`cat`）或变换时，由于无法通过数学等价性检查而报错。


* **符号推导断裂**：使用 `reshape(-1)` 时，编译器会生成复杂的除法公式，导致卷积层通道数检查失败。
* **现象**：在代码中使用 `reshape(-1)` 这种“懒人写法”时，编译器为了推导那个 `-1` 到具体维度的关系，会生成极其复杂的除法公式。
* **后果**：这种复杂的内部公式在后续传递到卷积层时，会导致编译器对通道数（Channel）的检查失效，产生报错，阻断了计算图的顺利提取。



### 解决方案

**暴力对齐：将“变量”转为“常量”**

通过显式定义 `target_h, target_w = 128, 256`，你主动**切断了编译器的符号追踪**。

* 这相当于直接告诉编译器：“别猜了，这里就是固定的 128 和 256”。虽然这在一定程度上牺牲了模型处理任意分辨率的泛化性，但它极大地增强了静态图提取的稳定性。

**语义替换：用 `F.interpolate` 代替 `reshape**`

* **原理**：`reshape` 只是改变了看待数据的方式（View），对编译器来说它的逻辑是隐晦的。
* **优势**：`F.interpolate`（插值）是一个具有明确语义的操作。它明确告诉编译器：“我要将这个张量缩放到特定的空间维度”。
* 这种写法为编译器提供了一条清晰的路径，确保了空间维度在整个计算链条中的一致性，从而解决了通道数检查失败的问题。

## 3. GitHub 提交 PR 的具体命令与步骤

这是本次实战中最具通用性的部分，适用于所有开源项目的贡献流程。

### 第一阶段：本地准备与自查

在生成计算图后，必须通过本地验证工具：

1. **设置工作区**：`export GRAPH_NET_EXTRACT_WORKSPACE=$(pwd)/output`
2. **运行抽取脚本**：`python extract_pigu.py`
3. **执行自查验证**：
`python -m graph_net.torch.validate --model-path ./output/pigu_hybrid`
*只有看到 **Validation success** 才能继续。*

### 第二阶段：Git 提交与冲突处理

1. **配置身份**（防止 CLA 报错）：
```bash
git config --global user.name "你的用户名"
git config --global user.email "你的邮箱@example.com"

```


*注意：邮箱必须在 GitHub Settings -> Emails 中通过验证。*
2. **移动文件到 samples 目录**：
`mv ./output/pigu_hybrid ../samples/`
3. **合并提交 (Squash)**：
如果之前有错误的提交记录（如邮箱乱码），使用 `reset` 合并：
```bash
git reset --soft HEAD~3  # 回退最近3次提交
git commit -m "Add PIGU_Hybrid model computation graph"

```



### 第三阶段：推送与令牌验证

由于 GitHub 不再支持密码推送，需使用 **Personal Access Token (PAT)**：

```bash
git push -f https://<你的Token>@github.com/用户名/仓库名.git HEAD:分支名

```

### 第四阶段：CI/CD 代码规范修复 (Code Style)

提交 PR 后，通常会遇到 **Codestyle-Check** 失败的情况。在 AI Studio 等云端环境中，直接运行完整的 `pre-commit` 框架容易导致**连接超时 (Connection Refused)**，且本地工具版本过新（如 Black v26.x）会导致与 CI 服务器（如 Black v23.1）产生格式冲突。

**终极解决方案（手动降级 + 模块化运行）：**

1. **确认项目要求的版本**：
先查看配置文件，锁定项目规定的工具版本（防止神仙打架）。
```bash
grep -A 2 "psf/black" .pre-commit-config.yaml
# 输出示例：rev: 23.1.0

```


2. **手动安装指定版本**：
绕过笨重的 `pre-commit` 环境安装，直接安装轻量级的对应版本包。
```bash
pip install black==23.1.0

```


3. **运行格式化（解决路径问题）**：
由于系统环境变量问题，直接运行 `black` 可能会报错 `command not found`。使用 `python -m` 调用是最稳妥的方式。
```bash
python -m black samples/pigu_hybrid

```


4. **提交修复**：
```bash
git add samples/pigu_hybrid/model.py
git commit -m "style: fix codestyle using black 23.1.0 matching CI"
git push -f https://<你的Token>@github.com/用户名/仓库名.git HEAD:分支名

```



---

## 4. 经验总结

* **AI Infra 的严谨性**：底层编译器对代码的规范性要求极高，任何模糊的 `view` 或 `reshape` 都可能导致追踪失败。
* **云端环境的特殊性**：在 Web IDE 中，避免运行耗时过长的自动化环境构建命令（如 `pre-commit install`），改用轻量级的 `pip` 手动安装是防止断连的有效手段。
* **版本对齐的重要性**：本地工具越新不一定越好，CI/CD 流水线中必须严格遵守项目的版本规范（`.pre-commit-config.yaml`）。
* **开源贡献的细节**：**CLA (Contributor License Agreement)** 的签署与 Git 邮箱的匹配是新人最容易掉进去的坑。
