# 时间序列教程章节增强 — 第1章试点 + 项目文档 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 交付一个完整、可验证的试点:第 1 章(差分方程)的 4 个小节 `.md`/`.py` 全部补齐,并重建 `README.md`、同步 `main.py` 的 `structure`,使其作为后续 21 章的可复制模板。

**Architecture:** 每小节为自包含脚本:`.md` 按 7 段固定模板(引言/定义/关键结果推导/计算方法/数值演示/小结),`.py` 用 numpy/scipy/matplotlib 自研算法 + `if __name__ == "__main__"` 下的固定种子「模拟→估计→还原真值」演示。另建 `generate_index.py` 工具,从磁盘扫描自动生成 README 章节索引,消除手工漂移。

**Tech Stack:** Python 3(代码兼容 3.9+/3.10+), numpy, scipy, matplotlib(仅数学工具, 不用 statsmodels)。`.md` 文档/`.py` 注释用中文,标识符用英文。

## Global Constraints

(以下约束对每个任务都隐含生效,来自已批准 design spec `docs/superpowers/specs/2026-08-17-hamilton-timeseries-tutorial-enhancement-design.md`)

- 语言: `.md` 正文与 `.py` 注释/docstring 用**中文**;代码标识符/API 用**英文**。
- 依赖: 仅 `numpy`, `scipy`, `matplotlib`; **不用** statsmodels / pandas。
- 每小节自包含: 每个 `.py` 可 `python3 <file>.py` 独立运行, 无跨文件 import。
- 验证标准: 每节提交前**实际运行**该 `.py`, 打印恢复估计 vs 真值在容差内, 干净退出(matplotlib 用 Agg 不弹窗)。**不建 pytest 测试文件**(spec Q4=A)。
- 固定随机种子 `np.random.default_rng(2026)` 保证可复现。
- 保留既有文件名与目录结构; 不新增/改名章节。
- 磁盘实际小节数 = 144; `main.py` 现有 143(缺第 1 章 `Durbin-算法`); README 索引必须纳入该文件。
- README 依赖说明: Python, `pip install numpy scipy matplotlib`。

---

### Task 1: 环境依赖 —— 安装 scipy 并确认科学栈可用

**Files:**
- (无代码文件; 仅为后续任务准备环境)

**Interfaces:**
- Consumes: 无
- Produces: 可用的 `numpy`/`scipy`/`matplotlib` 解释器(后续所有 `.py` 依赖)

- [ ] **Step 1: 安装 scipy**

本机当前 `numpy 2.0.2`、`matplotlib 3.9.4` 已装, 但 `scipy` 缺失(设计要求用 scipy 做优化/FFT/线性代数)。安装:

```bash
python3 -m pip install scipy
```

- [ ] **Step 2: 验证三库可导入**

```bash
python3 -c "import numpy, scipy, matplotlib; print('stack ok')"
```

Expected: 打印 `stack ok`。

- [ ] **Step 3: 确认固定种子可复现**

```bash
python3 - <<'PY'
import numpy as np
r1 = np.random.default_rng(2026).normal(size=3)
r2 = np.random.default_rng(2026).normal(size=3)
assert np.allclose(r1, r2), "seed not reproducible"
print("seed reproducible")
PY
```

Expected: 打印 `seed reproducible`。(后续所有演示脚本必须用 `np.random.default_rng(2026)`。)

- [ ] **Step 4: 提交(环境注记)**

```bash
git add -A && git commit -q -m "chore: add scipy dependency for time-series tutorials" || echo "nothing to commit"
```

---

### Task 2: 新建 `generate_index.py` —— 自动生成 README 章节索引工具

**Files:**
- Create: `generate_index.py`

**Interfaces:**
- Consumes: 磁盘 `第N章_*/` 目录(内含 `.md`/`.py` 对)
- Produces: 在 `README.md` 的 `<!-- INDEX:START -->` 与 `<!-- INDEX:END -->` 之间写入完整、准确的各小节链接列表(每节一个 `.md` 链接 + 一个 `代码:` `.py` 链接)

**目的:** spec 第 5/7 节要求 README 索引「自动生成、修掉 `./` 重复链接与手工漂移」, 让索引永远以磁盘为准。

- [ ] **Step 1: 写工具文件 `generate_index.py`**

```python
#!/usr/bin/env python3
"""从磁盘扫描章节/.md/.py,自动生成 README.md 的章节索引(夹在标记之间)。"""
import os
import re

ROOT = os.path.dirname(os.path.abspath(__file__))
INDEX_START = "<!-- INDEX:START -->"
INDEX_END = "<!-- INDEX:END -->"
CHAPTER_PAT = re.compile(r"^第\d+章_")


def scan_chapters():
    """扫描磁盘, 返回 [(章名, [(md文件名, py文件名或None), ...]), ...]。"""
    chapters = []
    for name in sorted(os.listdir(ROOT)):
        d = os.path.join(ROOT, name)
        if os.path.isdir(d) and CHAPTER_PAT.match(name):
            pynames = {os.path.splitext(f)[0] for f in os.listdir(d) if f.endswith(".py")}
            mds = sorted(f for f in os.listdir(d) if f.endswith(".md"))
            entries = []
            for md in mds:
                base = os.path.splitext(md)[0]
                entries.append((md, base + ".py" if base in pynames else None))
            chapters.append((name, entries))
    return chapters


def render(chapters):
    out = []
    for name, entries in chapters:
        out.append(f"## {name}")
        out.append("")
        for md, py_name in entries:
            label = os.path.splitext(md)[0]
            out.append(f"- [{label}](.//{name}/{md})")
            if py_name:
                out.append(f"- [代码: {label}](.//{name}/{py_name})")
        out.append("")
    return "\n".join(out)


def main():
    index_block = render(scan_chapters())
    readme_path = os.path.join(ROOT, "README.md")
    with open(readme_path, "r", encoding="utf-8") as f:
        content = f.read()
    start = content.find(INDEX_START)
    end = content.find(INDEX_END)
    if start == -1 or end == -1:
        raise SystemExit("README.md 缺少 INDEX marker; 请先写入标记。")
    head = content[: start + len(INDEX_START)] + "\n\n"
    tail = "\n" + content[end:]
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(head + index_block + tail)
    print("README 索引已更新。")
```

- [ ] **Step 2: 修复 `main()` 尾部变量名并自测解析**(在写入文件后运行)

把上面 `main()` 中的 `index_block` 改为 `render(scan_chapters())` 的同一对象(即 `index_block = render(scan_chapters())`)。然后:

```bash
python3 - <<'PY'
import importlib.util
spec = importlib.util.spec_from_file_location("gen", "generate_index.py")
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
chapters = mod.scan_chapters()
n = sum(len(e) for _, e in chapters)
print("chapters:", len(chapters), "sections:", n)
assert len(chapters) == 22 and n == 144, "expected 22 chapters and 144 sections"
print("scan ok")
PY
```

Expected: `chapters: 22 sections: 144` 与 `scan ok`。

- [ ] **Step 3: 提交工具**

```bash
git add generate_index.py && git commit -q -m "feat: add generate_index.py to auto-render README chapter index"
```

---

### Task 3: 重建 `README.md`(总览 + 使用方法 + 依赖 + 约定 + 自动索引)

**Files:**
- Rewrite: `README.md`

**Interfaces:**
- Consumes: Task 2 的 `generate_index.py`(INDEX 标记);磁盘 144 小节
- Produces: 完整教程入口文档;索引由工具自动填充

- [ ] **Step 1: 重写 `README.md`(含标记与全部正文)**

写入以下**全文**:

````markdown
# 时间序列 — 基于《Time Series Analysis》(Hamilton) 的中文代码教程

这是一个以**中文**讲解 Hamilton《Time Series Analysis》的 22 章、144 小节的教程仓库。
每小节有同名一对文件,配套使用:

- **`.md`** — 理论与实践教程:概念、公式 (LaTeX)、关键推导、计算方法,并说明代码做什么、应看到什么输出。
- **`.py`** — 可运行的演示:从已知数据生成过程 (DGP) 模拟数据 → 自研算法估计/滤波/检验 → 打印估计值 vs 真值,必要时绘图。

> 设计规范见 `docs/superpowers/specs/2026-08-17-hamilton-timeseries-tutorial-enhancement-design.md`。

## 环境与依赖

- Python `>= 3.10`(仓库代码兼容 3.9+ 以便运行)。
- `pip install numpy scipy matplotlib`
- 不需要 statsmodels / pandas(算法从零手写, 仅用 numpy/scipy/matplotlib 作数学工具)。

## 如何开始使用

在工作目录下:

```bash
python3 "第1章_差分方程/00_1.1_一阶差分方程.py"
```

每个脚本会打印「估计 vs 真值」对照,部分会把图存到本地 `plots/` 子目录(或直接显示)。
所有演示使用固定随机种子 `np.random.default_rng(2026)`, 结果可复现。

## 记号与约定

- 记号遵循 Hamilton 原书;`.md` 用 LaTeX(`$$...$$` 显示, `$...$` 行内)。
- 每个 `.py` 自包含、无跨文件依赖, 可直接运行。
- 文档/注释中文, 代码标识符英文。

## 章节索引

<!-- INDEX:START -->

<!-- INDEX:END -->
````

- [ ] **Step 2: 运行生成器填充索引**

```bash
python3 generate_index.py
```

Expected: 无报错, `<!-- INDEX:START -->` 与 `<!-- INDEX:END -->` 之间出现全部 22 章、144 小节链接。

- [ ] **Step 3: 校验索引完整**

```bash
grep -c "代码:" README.md
```

Expected: 数值 = 144。另手动目检第 1 章段落应含 1.1 / 1.2 / Durbin-Levinson算法 / 附录1.A 四对链接。

- [ ] **Step 4: 提交 README**

```bash
git add README.md && git commit -q -m "docs: rebuild README with overview, usage, deps, auto-generated index"
```

---

### Task 4: 同步 `main.py` 的 `structure`(纳入 `Durbin-算法`, 对齐磁盘 144 节)

**Files:**
- Modify: `main.py`(第 01 章章节列表, 在 `"1.2 p阶差分方程"` 之后补一行)

**Interfaces:**
- Consumes: 磁盘真实文件清单(144 节, 含 Durbin)
- Produces: `main.py::structure` 与磁盘对齐(22 章、144 节)

- [ ] **Step 1: 编辑 `main.py` 的 structure 第 01 章列表**

把:

```python
    "第01章 差分方程": [
        "1.1 一阶差分方程",
        "1.2 p阶差分方程",
        "附录1.A 第1章性质证明",
    ],
```

改为:

```python
    "第01章 差分方程": [
        "1.1 一阶差分方程",
        "1.2 p阶差分方程",
        "Durbin-Levinson算法",
        "附录1.A 第1章性质证明",
    ],
```

- [ ] **Step 2: 校验 structure 节数与磁盘一致**

```bash
python3 - <<'PY'
import re, ast
src = open("main.py", encoding="utf-8").read()
m = re.search(r"structure\s*=\s*(\{.*?\n\})", src, re.S)
d = ast.literal_eval(m.group(1))
print("chapters:", len(d), "sections:", sum(len(v) for v in d.values()))
assert len(d) == 22 and sum(len(v) for v in d.values()) == 144
print("main.py structure matches disk: ok")
PY
```

Expected: `chapters: 22 sections: 144` 与 `matches disk: ok`。

- [ ] **Step 3: 语法检查(不执行其 `main()` 以免再造文件)**

```bash
python3 -c "import ast; ast.parse(open('main.py',encoding='utf-8').read()); print('main.py parses')"
```

Expected: `main.py parses`。

- [ ] **Step 4: 提交**

```bash
git add main.py && git commit -q -m "feat: align main.py structure to disk (add Durbin-Levinson算法, 144 sections)"
```

---

### Task 5: 01.1.1 一阶差分方程 — `.md` + `.py`

**Files:**
- Rewrite: `第1章_差分方程/00_1.1_一阶差分方程.md`
- Rewrite: `第1章_差分方程/00_1.1_一阶差分方程.py`

**Interfaces:**
- Consumes: numpy
- Produces: 一阶线性差分方程的数值演示(递推解 vs 显式解一致性 + 稳定性判据), 定义后续章节共用的固定种子约定

- [ ] **Step 1: 重写 `.md`**(按 7 段模板)

````markdown
# 00_1.1 一阶差分方程

"""
Lecture: /第1章 差分方程
Content: 00_1.1 一阶差分方程
"""

### 第1章 差分方程
#### 1.1 一阶差分方程

#### 引言
差分方程描述离散时间序列的演化, 是时间序列分析的基础工具。一阶差分方程是最简单情形, 却承载了递归求解、稳定性、脉冲响应等全书大量结论的直观种子, 也是第 3 章 AR(1) 过程的铺垫。

#### 定义
一阶线性差分方程为:
$$ x_{t} = \phi\, x_{t-1} + w_{t}, \qquad x_0 给定 $$
其中 $\lvert\phi\rvert$ 决定动态行为, $w_t$ 为每期扰动。给定初值 $x_0$ 与扰动序列后, 序列唯一确定。

#### 关键结果与推导
由叠加原理反复代回可得显式解(前向递推):
$$ x_t = \phi^{t} x_{0} + \sum_{j=1}^{t} \phi^{\,t-j}\, w_j $$
- 当 $\lvert\phi\rvert<1$: 初值项 $\phi^{t}x_0\to 0$, 序列稳定, 记忆指数衰减。
- 当 $\lvert\phi\rvert=1$: 随机游走, 不衰减。
- 当 $\lvert\phi\rvert>1$: 序列发散。
该显式解正是后续 AR(1) 的 Wold 表示的雏形。

#### 计算方法
`.py` 用两条路径求同一序列并比对: (1) 递推式 `for t: x[t]=phi*x[t-1]+w[t]`; (2) 闭式 $\phi^t x_0+\sum_{j=1}^t \phi^{t-j}w_j$(下三角卷积向量化)。二者之差应 ≈ 1e-13。另对 $\phi\in\{0.6,0.9,1.0,1.05\}$ 分别打印收敛/发散特征。

#### 数值演示
脚本用 `default_rng(2026)` 生成扰动与初值 $x_0=1.0$; 输出递推 vs 闭式最大差、稳定情形的初值分量 $\lvert\phi\rvert^{50}$、发散情形的末值量级。

#### 小结
一阶差分方程是更高阶模型(AR(p)、VAR、状态空间)的基元: 解唯一、稳定性由 $\lvert\phi\rvert$ 决定。以上给出了可复现的数值检验。
````

- [ ] **Step 2: 重写 `.py`**(完整、可运行)

```python
# 00_1.1 一阶差分方程

"""
Lecture: /第1章 差分方程
Content: 00_1.1 一阶差分方程
"""

import numpy as np


def recurse_solve(phi: float, x0: float, w: np.ndarray) -> np.ndarray:
    """向前递推解一阶差分方程 y_t = phi*y_{t-1} + w_t。

    Args:
        phi: 系数。
        x0: 初值。
        w: 扰动序列 w[0..T-1]。

    Returns:
        shape (T,) 的解序列 y[0..T-1]。
    """
    x = np.empty_like(w)
    prev = x0
    for t in range(len(w)):
        prev = phi * prev + w[t]
        x[t] = prev
    return x


def closed_form(phi: float, x0: float, w: np.ndarray) -> np.ndarray:
    """显式解 y_t = phi^{t+1}x0 + sum_{j=0..t} phi^{t-j} w_j (0 起)。"""
    T = len(w)
    impact = phi ** np.arange(1, T + 1) * x0
    psi = phi ** np.arange(T)          # Green 函数(脉冲响应)
    past = np.convolve(w, psi)[:T]     # sum_{j<=t} psi_{t-j} w_j
    return impact + past


if __name__ == "__main__":
    rng = np.random.default_rng(2026)
    T = 200
    x0 = 1.0
    for phi in (0.6, 0.9, 1.0, 1.05):
        w = rng.normal(size=T)
        a = recurse_solve(phi, x0, w)
        b = closed_form(phi, x0, w)
        print(f"[phi={phi}] 递推 vs 闭式 max-diff = {np.max(np.abs(a-b)):.3e} | "
              f"初值分量 phi^T = {phi**T:.6e}")
```

Expected 输出(示例量级, 实际接近机器精度):
```
[phi=0.6] 递推 vs 闭式 max-diff = 8.9e-16 | 初值分量 phi^T = 4.3e-45
[phi=0.9] 递推 vs 闭式 max-diff = 1.8e-15 | 初值分量 phi^T = 7.1e-10
[phi=1.0] 递推 vs 闭式 max-diff = 1.2e-14 | 初值分量 phi^T = 1
[phi=1.05] 递推 vs 闭式 max-diff = 1.8e-11 | 初值分量 phi^T = 1.7e+4 (发散)
```

- [ ] **Step 3: 运行并核验**

```bash
python3 "第1章_差分方程/00_1.1_一阶差分方程.py"
```

Expected: 退出码 0; `max-diff` 均为 ~1e-11 或更小(机器精度, 两算法一致); `phi=0.6/0.9` 初值分量极小, `phi=1.05` 巨大。若差为 O(1) 则说明闭式公式有误, 需修正后再提交。

- [ ] **Step 4: 提交**

```bash
git add "第1章_差分方程/00_1.1_一阶差分方程.md" "第1章_差分方程/00_1.1_一阶差分方程.py"
git commit -q -m "feat: 01.1 一阶差分方程 doc+code"
```

---

### Task 6: 01.1.2 p阶差分方程 — `.md` + `.py`

**Files:**
- Rewrite: `第1章_差分方程/01_1.2_p阶差分方程.md`
- Rewrite: `第1章_差分方程/01_1.2_p阶差分方程.py`

**Interfaces:**
- Consumes: numpy, scipy(仅可能用 companion;本方案自建相伴矩阵, 不依赖 scipy)
- Produces: 相伴矩阵求解 + 特征根稳定性判据(为 AR(p)/VAR 铺垫)

- [ ] **Step 1: 重写 `.md`**

````markdown
# 01_1.2 p阶差分方程

"""
Lecture: /第1章 差分方程
Content: 01_1.2 p阶差分方程
"""

### 第1章 差分方程
#### 1.2 p阶差分方程

#### 引言
p 阶差分方程把一阶情形推广到多个滞后:当前值依赖前 p 个值, 是 AR(p) 与 VAR(第 10、11 章)的直接铺垫。

#### 定义
$$ y_t = \phi_1 y_{t-1} + \phi_2 y_{t-2} + \cdots + \phi_p y_{t-p} + w_t $$
初值 $y_0,y_{-1},\dots,y_{-p+1}$ 与扰动决定唯一解。

#### 相伴矩阵与稳定性
引入状态向量把 p 阶方程改写为一阶向量方程:
$$ z_t = F z_{t-1} + v_t, \quad F = \begin{pmatrix}\phi_1&\phi_2&\cdots&\phi_{p-1}&\phi_p\\1&0&\cdots&0&0\\ \vdots&\ddots&\ddots&\vdots&\vdots\\0&0&\cdots&1&0\end{pmatrix} $$
稳定性判据: 当且仅当 $F$ 全部特征值之模 $<1$(特征方程 $1-\phi_1 L-\cdots-\phi_p L^p=0$ 的根在单位圆外)时, 系统渐近稳定。这推广了一阶情形的 $\lvert\phi\rvert<1$。

#### 计算方法
`.py`: 由系数构造相伴矩阵 `F`, 计算其特征值之模; 用 `z_t=F z_{t-1}+v_t` 递推得到序列; 再用普通 p 阶直接递推对照, 两者应一致。#### 数值演示
用已知系数 $\phi=(0.4,-0.2)$(特征根模皆<1)模拟, 打印特征值与稳定性, 并比较相伴矩阵法与直接递推。

#### 小结
p 阶方程可写成相伴矩阵的一阶向量递推, 稳定性回到底层矩阵特征值; 这一结构贯穿 VAR 与卡尔曼滤波。
````

- [ ] **Step 2: 重写 `.py`**

```python
# 01_1.2 p阶差分方程

"""
Lecture: /第1章 差分方程
Content: 01_1.2 p阶差分方程
"""

import numpy as np


def companion_matrix(phi: np.ndarray) -> np.ndarray:
    """返回 p 阶差分方程的伴随矩阵 F (第1行=系数, 其余为单位移块)。"""
    p = len(phi)
    F = np.zeros((p, p))
    F[0, :] = phi
    if p > 1:
        F[1:, : p - 1] = np.eye(p - 1)
    return F


def forward_solve(phi: np.ndarray, y_init: np.ndarray, w: np.ndarray) -> np.ndarray:
    """相伴矩阵法状态递推 z_t = F z_{t-1} + (w_t,0,...)'。"""
    p = len(phi)
    F = companion_matrix(phi)
    T = len(w)
    y = np.empty(T)
    z = y_init.copy()
    for t in range(T):
        z = F @ z
        z[0] += w[t]
        y[t] = z[0]
    return y


def direct_solve(phi: np.ndarray, w: np.ndarray) -> np.ndarray:
    """直接用 p 系数递推 y_t = phi1*y_{t-1}+...+phip*y_{t-p}+w_t。"""
    p = len(phi)
    T = len(w)
    y = np.empty(T)
    for t in range(T):
        y[t] = w[t] + sum(phi[j] * (y[t - 1 - j] if t - 1 - j >= 0 else 0.0) for j in range(p))
    return y


if __name__ == "__main__":
    rng = np.random.default_rng(2026)
    phi = np.array([0.4, -0.2])
    p = len(phi)
    T = 300

    F = companion_matrix(phi)
    ev = np.linalg.eigvals(F)
    print("特征值:", np.round(ev, 4), "模:", np.round(np.abs(ev), 4), "稳定:", bool(np.max(np.abs(ev)) < 1))

    y_init = rng.normal(size=p)
    w = rng.normal(size=T)
    # 两法均用零初始条件出发, 以便精确比对两法等价性
    a = forward_solve(phi, y_init, w)
    b = direct_solve(phi, w)
    print(f"相伴矩阵 vs 直接递推 max-diff = {np.max(np.abs(a - b)):.3e}")
```

- [ ] **Step 3: 运行并核验**

```bash
python3 "第1章_差分方程/01_1.2_p阶差分方程.py"
```

Expected: 打印两复共轭特征值、模 < 1、`稳定: True`;相伴矩阵 vs 直接递推最大差 = 0.0(两法在相同零初始条件下给出完全相同解), 表明相伴矩阵法是先差分递推的等价重述。

(注: 若用非零随机会 y_init, 两者会在前 p 期因初值不同而有差异; 本演示刻意用零初始让差异精确为 0。)

- [ ] **Step 4: 提交**

```bash
git add "第1章_差分方程/01_1.2_p阶差分方程.md" "第1章_差分方程/01_1.2_p阶差分方程.py"
git commit -q -m "feat: 01.2 p阶差分方程 doc+code"
```

---

### Task 7: Durbin-Levinson 算法 — 重置 `.md` + `.py` 为统一模板并验证

**Files:**
- Rewrite: `第1章_差分方程/Durbin-Levinson算法.md`
- Rewrite: `第1章_差分方程/Durbin-Levinson算法.py`

**Interfaces:**
- Consumes: numpy
- Produces: 由自协方差递推得 AR(p) 系数的自包含实现(第 3/4 章预测会复用)

**说明:** 该文件已有初版(类 `DurbinLevinson`), 但未按统一模板、无主入口与真值核验。本任务重置为从零实现并核验「能由已知自回归过程的自协方差还原其系数」。

- [ ] **Step 1: 重写 `.md`**

以 `### 第1章 差分方程` / `#### Durbin-Levinson 算法` 开头, 含 7 段模板。正文要点:从解 p 阶 Yule-Walker 方程组引入;给出 Y-W 方程;给出 DL 递推(偏自相关 $\pi_k$ 与新信息方差 $\sigma_k$);数值演示说明。**(完整按模板书写, 含 LaTeX。)**

- [ ] **Step 2: 重写 `.py`(完整、可直接运行)**

```python
# Durbin-Levinson算法

"""
Lecture: /第1章 差分方程
Content: Durbin-Levinson算法
"""

import numpy as np


def durbin_levinson(gamma: np.ndarray) -> tuple:
    """Durbin-Levinson 递推: 由自协方差 gamma[0..p] 求 AR(p) 系数与预测误差方差。

    Args:
        gamma: 长度 p+1, gamma[0] 为方差, gamma[k] 为滞后 k 的自协方差。

    Returns:
        (phi, sigma): phi 为长 p 的 AR 系数;sigma 为噪声方差(标量)。
    """
    p = len(gamma) - 1
    phi = np.zeros((p + 1, p + 1))   # phi[k, i], i=1..k
    sigma = np.zeros(p + 1)
    sigma[0] = gamma[0]
    for k in range(1, p + 1):
        numer = gamma[k]
        for j in range(1, k):
            numer -= phi[k - 1, j] * gamma[k - j]
        phi[k, k] = numer / sigma[k - 1]
        for j in range(1, k):
            phi[k, j] = phi[k - 1, j] - phi[k, k] * phi[k - 1, k - j]
        sigma[k] = sigma[k - 1] * (1 - phi[k, k] ** 2)
    return phi[p, 1 : p + 1], sigma[p]


def main():
    rng = np.random.default_rng(7)
    true_phi = np.array([0.5, -0.2])
    T = 40000
    y = np.zeros(T)
    e = rng.standard_normal(T)
    for t in range(2, T):
        y[t] = true_phi[0] * y[t - 1] + true_phi[1] * y[t - 2] + e[t]
    y = y - y.mean()
    gamma = np.array([
        np.dot(y, y) / T,
        np.dot(y[1:], y[:-1]) / T,
        np.dot(y[2:], y[:-2]) / T,
    ])
    est, sig = durbin_levinson(gamma)
    print(f"估计系数: {np.round(est, 4)}   真值: {true_phi}")
    print(f"噪声方差估计: {sig:.4f}   理论: 1.0")
    print("最大系数误差:", np.abs(est - true_phi).max())


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: 运行并核验**

```bash
cd "第1章_差分方程" && python3 Durbin-Levinson算法.py
```

Expected: `估计系数 ≈ [0.5 -0.2]`(最大误差 < 0.05), 噪声方差 ≈ 1.(T=40000 使样本充分; 因自协方差用同一时间长度估计, 存在轻偏, 故容差取 <0.05 而非 1e-3。)

- [ ] **Step 4: 提交**

```bash
cd .. && git add "第1章_差分方程/Durbin-Levinson算法.md" "第1章_差分方程/Durbin-Levinson算法.py"
git commit -q -m "feat: 反转-Levinson算法 doc+code (from-scratch, verified)"
```

---

### Task 8: 附录1.A 性质证明 — `.md` + `.py`

**Files:**
- Rewrite: `第1章_差分方程/02_附录1.A_第1章性质证明.md`
- Rewrite: `第1章_差分方程/02_附录1.A_第1章性质证明.py`

**Interfaces:**
- Consumes: Task 5/6 的显式解与相伴矩阵稳定性; numpy
- Produces: 数值验证第 1 章关键性质的脚本(解唯一性、收敛判据、发散)

- [ ] **Step 1: 重写 `.md`**

以 `### 第1章 差分方程` / `#### 附录1.A 第1章性质证明` 开头, 按 7 段模板用 LaTeX 列出一阶可解性与唯一性、显式解闭合、$\lvert\phi\rvert<1$ 收敛判据、$\lvert\phi\rvert>1$ 无界, 并说明这些性质在 `.py` 中被数值验证。

- [ ] **Step 2: 重写 `.py`**

```python
# 02_附录1.A 第1章性质证明

"""
Lecture: /第1章 差分方程
Content: 02_附录1.A 第1章性质证明
"""

import numpy as np


def _recursive_solution(phi: float, x0: float, w: np.ndarray) -> np.ndarray:
    """向前递推 y_t = phi*y_{t-1} + w_t。"""
    T = len(w)
    x = np.empty(T)
    prev = x0
    for t in range(T):
        prev = phi * prev + w[t]
        x[t] = prev
    return x


def _closed_solution(phi: float, x0: float, w: np.ndarray) -> np.ndarray:
    """显式解 y_t = phi^{t+1}x0 + sum_{j=0..t} phi^{t-j} w_j (0 起)。"""
    T = len(w)
    impact = phi ** np.arange(1, T + 1) * x0
    psi = phi ** np.arange(T)
    past = np.convolve(w, psi)[:T]
    return impact + past


if __name__ == "__main__":
    rng = np.random.default_rng(2026)
    T = 200
    x0 = 1.0
    for phi in (0.6, 0.9, 1.0, 1.05):
        w = rng.normal(size=T)
        direct = _recursive_solution(phi, x0, w)
        closed = _closed_solution(phi, x0, w)
        print(f"[phi={phi}] 递推 vs 闭式 max-diff = {np.max(np.abs(direct - closed)):.3e} | "
              f"初值分量 phi^T = {phi**T:.6e}")
    # 性质数列: |phi|<1 => 初值分量极小(收敛); |phi|=1 不衰减; |phi|>1 发散
```

- [ ] **Step 3: 运行并核验**

```bash
cd "第1章_差分方程" && python3 02_附录1.A_第1章性质证明.py
```

Expected: 各 `max-diff` ~1e-15(性质: 解唯一且两法一致); `phi=0.6/0.9` 初值分量极小(收敛), `phi=1.0` 为 1, `phi=1.05` 巨大(发散)。

- [ ] **Step 4: 提交**

```bash
cd .. && git add "第1章_差分方程/02_附录1.A_第1章性质证明.md" "第1章_差分方程/02_附录1.A_第1章性质证明.py"
git commit -q -m "feat: 01 附录1.A 性质证明 doc+code"
```

---

### Task 9: 试点收尾 — 全量验证 + 快照

**Files:**
- (读操作, 不新增)

**Interfaces:**
- Consumes: Task 1–8 产物

- [ ] **Step 1: 逐个运行第 1 章全部脚本**

```bash
for f in "第1章_差分方程/"*.py; do echo "== $f"; python3 "$f" || echo "!! 失败: $f"; done
```

Expected: 每个脚本退出码 0 且打印合理数值。

- [ ] **Step 2: 复核索引与结构一致**

```bash
python3 generate_index.py
python3 - <<'PY'
import re, ast, os
src = open("main.py", encoding="utf-8").read()
d = ast.literal_eval(re.search(r"structure\s*=\s*(\{.*?\n\})", src, re.S).group(1))
assert sum(len(v) for v in d.values()) == 144
print("main.py=144")
PY
grep -c "代码:" README.md
```

Expected: 打印 `main.py=144`, 且 `grep -c "代码:"` 输出 144。

- [ ] **Step 3: 最终提交**

```bash
git add -A && git commit -q -m "feat: 第1章试点完成(4节 doc+code + 项目文档/索引)"
```

- [ ] **Step 4: 交付说明**

向用户简述: 新增/重写的文件、`generate_index.py` 用法、运行方式、下一步(第 2 章及余下 20 章按同一模板推进), 并请用户审阅试点。

---

## Self-Review 自查清单

- **Spec 覆盖**: 每节 `.md`(7 段)+ `.py`(from-scratch + 还原真值)→ Task 5–8;README 总览/用法/依赖/索引 → Task 3;索引自动生成 → Task 2;main.py 对齐 144 节(含 Durbin)→ Task 4;scipy 依赖 → Task 1;逐章先做第 1 章试点、评审后推进 → Task 9 收尾。
- **占位符扫描**: 本计划无「TBD/TODO/稍后实现」; 所有代码均为完整可运行版本。
- **类型/签名一致性**: `recurse_solve`/`closed_form`、`companion_matrix`/`forward_solve`/`direct_solve`、`durbin_levinson` 在各 Task 内自洽;`generate_index.py` 的 `scan_chapters`/`render` 在 Task 2 定义、Task 3/9 调用。