# 果蝇种群沙盒

**连接组 + 孟德尔遗传 + 自然规律** —— 一个零依赖的单文件 HTML 演化沙盒。

每只果蝇都有一套独立的 LIF 脉冲神经网络大脑，遵循真实的孟德尔遗传规律交配繁殖，并在 Gompertz-Makeham 死亡律下被自然选择筛汰。点几下鼠标，就能看着一个种群在几十代里漂变、固定、或是被一次突变改写命运。

---

## 快速开始

页面通过 `fetch("data/connectome_demo.json")` 载入连接组，因此**直接双击 HTML 会被浏览器的 CORS 策略拦住**，需要起一个本地服务器：

```bash
python -m http.server 8000
# 然后打开 http://localhost:8000
```

> 只想体验内置大脑、不载入 MaleCNS 连接组的话，直接双击 `fly_sandbox.html` 也能跑。

---

## 三大系统

### 🧠 连接组驱动的大脑

- 每只果蝇一张**独立的 LIF 网络**，互不共享
- 内置一套轻量模板连接组，开箱即用
- 点「🧠 载入 MaleCNS 连接组」可一键切换到真实规模拓扑，**所有果蝇的大脑同时被接管**（可随时 ↺ 还原）
- 通路结构：感觉入口（层 0 神经元按类型分配模态）→ 6 层传播 → 运动输出（层 4 的 DN 按左右半球分侧）
- 突触用 **CSR 稀疏出边索引**存储，每步只遍历发放神经元的突触，两万神经元规模也能实时跑

### 🧬 孟德尔遗传 v2.0

- **20 个基因座 × 4 条染色体**：X 连锁 6 个、第 2 染色体 4 个、第 3 染色体 5 个、第 4 染色体 5 个（第 4 染色体极小且无交换）
- 基因座位置采用 **FlyBase 实测遗传图距（cM）**，相邻基因座按真实重组率发生交换
- 每代每基因座突变率 **1.2e-5**，且正向失活远高于回复突变（4:1）——所以坏突变容易攒、极难自发修复
- 支持**上位效应**：例如 `st`（朱红，缺褐色素）与 `bw`（褐眼，缺红色素）单突变各有眼色，双突变则两类色素皆无 → 白眼
- 雄性 X 染色体半合子，隐性等位基因直接表达

### 🌿 自然规律

- **Gompertz-Makeham 死亡律**，叠加温度、表型适合度、以及交配代价（已交配雌蝇风险 ×1.22，对应精液蛋白/性肽缩短寿命的真实效应）
- **求偶状态机**按真实果蝇序列推进：定向 → 触碰 → 振翅鸣唱 → 舔舐 → 尝试交配
- 雌蝇若处于**性肽拒配期**会拒绝交配；一次交配储精约 400–600 枚，供后续所有卵受精
- 点击放入的是**野生型杂合携带者**——模拟野外捕捞，自然种群本就携带隐性等位基因

### ⏩ 跳代演化

点「⏩ 快进 5000 代」直接把演化推到远期：大脑按 1/6 采样（LIF 的 τ=10ms 远小于行为决策时间尺度，采样后行为统计等效）、轨迹冻结。种群存活数 <4 时自动迁入野生型一雌一雄补种，避免整群灭绝。

---

## 操作

| 操作 | 说明 |
| --- | --- |
| 点击地面 | 放置目标 |
| 拖动 | 旋转视角 |
| 滚轮 | 缩放 |
| <kbd>1</kbd>–<kbd>6</kbd> | 切换目标类型 |
| <kbd>Tab</kbd> | 切换跟随视角 |
| <kbd>空格</kbd> | 暂停 |
| <kbd>R</kbd> | 重置 |
| <kbd>S</kbd> | 加速 |

右侧面板另有「载入 MaleCNS 连接组」和「快进 5000 代」两个按钮。

---

## 目录结构

```
.
├── fly_sandbox.html           单文件应用：UI + 仿真 + 渲染，无任何外部依赖
├── src/connectome_import.py   连接组解析：feather/CSV → 6 层映射 → 浏览器可加载 JSON
├── data/connectome_demo.json  MaleCNS 风格演示数据（20,000 神经元 / 347,377 突触，7.6 MB）
├── README.md
└── LICENSE                    MIT
```

---

## 连接组数据

### 用仓库自带的演示数据

`data/connectome_demo.json` 已随仓库提供，是按 MaleCNS 统计特性合成的：

- **幂律突触分布** —— 少数高度连接的 hub 神经元，符合真实连接组
- **同侧连接偏置** —— 约 70% 突触强制同半球（真实果蝇脑特征，视交叉除外）
- **层间前向为主** —— 80% 连接前向或同层，20% 任意（含稀疏反馈）
- **递质比例** —— ACh 60%、GABA 20%、Glu 15%、调制类 5%

> ⚠️ 这是**合成数据**，遵循真实统计特性但不是 MaleCNS 的实测发布数据。

### 从真实数据生成

[MaleCNS v1.0](https://codex.janelia.org)（*Cell* 2026-09-03，HHMI Janelia + Cambridge + Google）：166,700 神经元、125,000,000 突触，feather（Arrow IPC）格式，约 100 GB，需自行下载。

```bash
pip install pandas numpy pyarrow

# 完整管线：feather → 过滤 → 6 层映射 → 导出
python src/connectome_import.py \
    --input synapses.feather --neurons neurons.feather \
    --output data/connectome.json \
    --top-neurons 20000 --min-syn 5

# 无真实数据时，重新合成一份演示数据
python src/connectome_import.py --synthetic --output data/connectome_demo.json
```

脚本也接受任意含 `pre_root_id / post_root_id / syn_count` 三列的 feather 或 CSV（自动兼容 FlyWire 的 `pre / post` 命名）。

### 6 层架构映射

| 层 | 对应 | 典型神经元类型 |
| --- | --- | --- |
| 0 | V1 感觉 | R1–R8、L1–L5、ORN、GRN、JO、机械/温度/信息素受体 |
| 1 | V2 联合 | Tm、TmY、T4/T5、Mi、PN、iACT |
| 2 | PE 前额叶 | MB（Kenyon cell）、CX、AOTU、TuBu |
| 3 | IFC 符号 | LH（lateral horn）、saddle、flange、LC |
| 4 | 决策输出 | DN（descending neuron）、AN、GF |
| 5 | TC 时间 | FB、EB、PB、NO、生物钟 LN |

突触权重按递质带符号：**GABA 与 glutamate 为抑制**（果蝇的 glutamate 主要起抑制作用），ACh / DA / 5HT / OA 为兴奋。

### 输出格式

紧凑数组结构，浏览器端 `fetch` 后构建 CSR 索引：

```json
{
  "meta": { "source": "...", "n_neurons": 20000, "n_synapses": 347377, "min_syn": 5 },
  "neurons": [ { "id": 0, "type": "PFL2", "layer": 3, "side": "L" } ],
  "synapses": { "src": [...], "tgt": [...], "w": [...] }
}
```

---

## 部署到 GitHub Pages

仓库根目录即是站点根目录，`fly_sandbox.html` 与 `data/` 的相对路径关系天然可用：

1. Settings → Pages → Build and deployment
2. Source 选 **Deploy from a branch**
3. Branch 选 **main**、目录选 **/(root)** → Save

部署完成后访问：`https://<用户名>.github.io/fly_sandbox.html/`

---

## License

MIT © 2026 Agnes
