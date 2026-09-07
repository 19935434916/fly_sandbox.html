"""
MaleCNS v1.0 连接组数据解析器 (feather/Arrow IPC 格式)

MaleCNS v1.0 (Cell, 2026-09-03, HHMI Janelia + Cambridge + Google):
  - 166,700 神经元 (雄性果蝇全中枢神经系统)
  - 125,000,000 突触
  - 数据格式: feather (Arrow IPC), 每文件一个表

支持的数据源:
  1. MaleCNS 官方发布 (需下载, ~100GB):
     https://codex.janelia.org (男蝇 CNS release)
     - neurons.feather     : 神经元表 (root_id, type, side, hemisphere, n_syn...)
     - synapses.feather    : 突触表 (pre_root_id, post_root_id, syn_count, nt_type...)
     - proofreading.feather: 校对状态表

  2. FlyWire (雌蝇, 已发布, cleft 形式):
     https://codex.flywire.ai/api/download (材料: 2026-03 Eon Systems 曾用)

  3. 本工具也接受任意含列 (pre, post, weight) 的 feather/CSV

用法:
  # 完整管线: feather -> 过滤 -> 映射到6层架构 -> 导出 JS 可加载数据
  python connectome_import.py --input synapses.feather --neurons neurons.feather \
      --output ../data/connectome.json --top-neurons 20000 --min-syn 5

  # 用自带合成数据演示 (无真实数据时)
  python connectome_import.py --synthetic --output ../data/connectome_demo.json

输出 JSON 结构 (浏览器端直接 fetch):
  {
    "meta": { "source", "n_neurons", "n_synapses", "generated_at" },
    "neurons": [ { "id", "type", "layer", "side", "pos" } ... ],
    "synapses": { "src": [...], "tgt": [...], "w": [...] }   // 紧凑数组格式
  }
"""
import argparse
import json
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# 6 层皮层映射 (与 fly_sandbox.html / fly_final.html 的架构对齐)
# 层 0 V1 感觉: 视觉/嗅觉/味觉/触觉/听觉外周 + 视叶 (optic lobe)
# 层 1 V2 联合: 中间神经元密集区 (optic glomeruli, AL 输出, MB calyx)
# 层 2 PE 前额叶: mushroom body vertical/beta-lobe, central complex
# 层 3 IFC 符号: lateral horn, saddle, flange (高级关联)
# 层 4 决策: descending neurons (DN), neck motor
# 层 5 TC 时间: fan-shaped body, ellipsoid body (central complex 时间积分)
# ---------------------------------------------------------------------------
LAYER_RULES = [
    # (层号, 匹配关键词列表) —— 按 MaleCNS/FlyWire 的 neuron type 命名约定
    (5, ["FB", "EB", "fan-shaped", "ellipsoid", "PB", "NO", "clock", "DN1", "DN2", "DN3", "LN", "s-LNv", "l-LNv"]),
    (4, ["DN", "descending", "neck", "AN", "giant fiber", "GF"]),
    (3, ["LH", "lateral horn", "saddle", "flange", "clamp", "EPG", "PFL", "APE", "LC", "LT"]),
    (2, ["MB", "mushroom", "KC", "Kenyon", "CX", "noduli", "AOTU", "TuBu", "TuL", "PB"]),
    (1, ["Tm", "TmY", "T4", "T5", "Mi", "CT", "C2", "C3", "PLC", "PS", "AL", "PN", "ORN", "olf", "iACT", "oACT", "mACT"]),
    (0, ["R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8", "L1", "L2", "L3", "L4", "L5", "Lawf", "Amac",
         "GRN", "GR", "BR", "mechanoreceptor", "JO", "chordotonal", "WGRN", "water", "TRP", "thermal",
         "AVR", "aversive", "PHER", "pheromone", "ORN", "sensory", "receptor"]),
]

NT_WEIGHT = {  # 神经递质 -> 突触符号 (兴奋 +, 抑制 -)
    "acetylcholine": 1.0, "ACH": 1.0,
    "glutamate": -0.8, "GLUT": -0.8,          # 果蝇 glutamate 主要为抑制
    "GABA": -1.0,
    "dopamine": 0.5, "DA": 0.5,
    "serotonin": 0.4, "5HT": 0.4,
    "octopamine": 0.6, "OA": 0.6,             # 强化/唤醒
    "unknown": 0.8,
}


def classify_layer(neuron_type: str) -> int:
    """按 type 名称映射到 6 层架构 (先匹配高层层号规则)"""
    if not neuron_type:
        return 1
    t = neuron_type.upper()
    for layer, kws in LAYER_RULES:          # 从高层(5)向下匹配
        for kw in kws:
            if kw.upper() in t:
                return layer
    return 1                                # 默认归联合层


def load_feather(path: str, columns=None):
    """读 feather/arrow IPC 文件 -> pandas DataFrame"""
    import pandas as pd
    return pd.read_feather(path, columns=columns)


def load_csv(path: str):
    import pandas as pd
    return pd.read_csv(path)


def synthetic_dataset(n_neurons=20000, n_syn=400000, seed=42):
    """合成 MaleCNS 风格数据集 (无真实数据时演示用)

    遵循真实果蝇的连接统计:
    - 层间前向连接为主, 反馈稀疏
    - 幂律突触数量分布 (少数高度连接的 hub 神经元)
    - 递质比例: ACh 60%, GABA 20%, Glu 15%, 调制类 5%
    """
    import numpy as np
    rng = np.random.default_rng(seed)

    # 每层神经元数比例 (仿真实: 感觉层最大)
    layer_frac = [0.40, 0.28, 0.15, 0.10, 0.04, 0.03]
    layers = rng.choice(6, size=n_neurons, p=layer_frac)

    # 神经元 type 名称池 (真实类型命名)
    type_pools = {
        0: ["R1-R6", "L1", "L2", "L3", "Mi1", "Tm3", "GRN", "ORN", "JO", "WGRN",
            "TRP", "AVR", "PHER-DA1"],
        1: ["Tm1", "TmY", "T4", "T5", "PN", "iACT", "KC", "CT1", "C2"],
        2: ["KC", "MB-V", "MB-b", "AOTU", "TuBu", "EPG", "PFL1", "PFL2"],
        3: ["LC10", "LC11", "LT11", "LH", "saddle", "clamp", "AVLP"],
        4: ["DNge1", "DNam1", "DNge104", "DNg13", "DNp11", "AN"],
        5: ["FB4", "FB5", "EB", "PB", "NO1", "DN1p", "s-LNv", "l-LNv"],
    }
    types = [rng.choice(type_pools[l]) for l in layers]

    # 半球侧 (真实果蝇: 大多数通路同侧连接, 视叶经视交叉对侧)
    sides = rng.choice(["L", "R"], n_neurons)

    # 递质
    nt_choices = rng.choice(
        ["acetylcholine", "GABA", "glutamate", "dopamine", "octopamine"],
        size=n_neurons, p=[0.60, 0.20, 0.15, 0.03, 0.02])

    # 突触: 层间前向 + 层内 + 少量反馈
    src_idx = rng.integers(0, n_neurons, n_syn)
    src_layers = layers[src_idx]
    # 目标层: 80% 前向(或同层), 20% 任意
    tgt_layers = np.where(
        rng.random(n_syn) < 0.8,
        np.minimum(src_layers + (rng.random(n_syn) < 0.6).astype(int), 5),
        rng.integers(0, 6, n_syn))
    tgt_idx = rng.integers(0, n_neurons, n_syn)
    tgt_idx = np.where(layers[tgt_idx] == tgt_layers, tgt_idx,
                       rng.integers(0, n_neurons, n_syn))

    # 同侧连接偏置: 70% 突触强制同半球 (真实果蝇脑特征; 视交叉除外)
    idxL = np.where(sides == "L")[0]
    idxR = np.where(sides == "R")[0]
    flip = (rng.random(n_syn) < 0.70) & (sides[tgt_idx] != sides[src_idx])
    for sv, pool_idx in (("L", idxL), ("R", idxR)):
        m = flip & (sides[src_idx] == sv)
        if m.any():
            tgt_idx[m] = pool_idx[rng.integers(0, len(pool_idx), int(m.sum()))]

    # 幂律突触权重
    w = (1.0 / (rng.pareto(2.5, n_syn) + 1.0)) * 0.9 + 0.1
    # 递质符号 (抑制性: GABA/glutamate)
    sign = np.where(np.isin(nt_choices[src_idx], ["GABA", "glutamate"]), -1.0, 1.0)
    w = w * sign

    df_syn = {
        "pre_root_id": src_idx,
        "post_root_id": tgt_idx,
        # 注意: 计数取绝对值 (真实数据 syn_count 恒正, 符号由 nt_type 决定)
        "syn_count": (np.abs(w) * 10).astype(int).clip(1),
        "nt_type": nt_choices[src_idx],
    }
    df_neu = {
        "root_id": np.arange(n_neurons),
        "type": types,
        "layer": layers,
        "side": sides,
        "nt": nt_choices,
    }
    return df_syn, df_neu


def build_connectome(df_syn, df_neu, top_neurons=None, min_syn=5, source=None):
    """过滤 + 映射 -> 浏览器可加载的紧凑结构"""
    import numpy as np

    # 1) 取突触数最多的 top_neurons (hub 神经元), 保证连通性
    if top_neurons:
        in_deg = df_syn.groupby("post_root_id")["syn_count"].sum()
        out_deg = df_syn.groupby("pre_root_id")["syn_count"].sum()
        deg = (in_deg.add(out_deg, fill_value=0)).sort_values(ascending=False)
        keep = set(deg.head(top_neurons).index.astype(int))
    else:
        keep = set(df_neu["root_id"].astype(int))

    # 2) 过滤突触 (保留递质类型用于符号)
    pre = df_syn["pre_root_id"].astype(int).values
    post = df_syn["post_root_id"].astype(int).values
    cnt = df_syn["syn_count"].astype(float).values
    nt = (df_syn["nt_type"].astype(str).values
          if "nt_type" in df_syn.columns else None)
    mask = [(p in keep and q in keep and c >= min_syn)
            for p, q, c in zip(pre, post, cnt)]
    pre_f, post_f, cnt_f = pre[mask], post[mask], cnt[mask]
    nt_f = nt[mask] if nt is not None else None

    # 3) 重编号为 0..n-1
    id_map = {old: new for new, old in enumerate(sorted(keep))}
    src = np.array([id_map[p] for p in pre_f])
    tgt = np.array([id_map[q] for q in post_f])

    # 4) 神经元元数据
    neu_rows = df_neu[df_neu["root_id"].isin(keep)].set_index("root_id")
    neurons = []
    for old, new in sorted(id_map.items(), key=lambda kv: kv[1]):
        row = neu_rows.loc[old] if old in neu_rows.index else None
        ntype = str(row["type"]) if row is not None and "type" in row else "?"
        layer = (int(row["layer"]) if row is not None and "layer" in row
                 else classify_layer(ntype))
        side = str(row["side"]) if row is not None and "side" in row else "C"
        neurons.append({"id": new, "type": ntype, "layer": layer, "side": side})

    # 5) 权重归一化到 LIF 可用范围 (0.2~0.9) + 递质符号
    #    GABA/Glutamate = 抑制(-1); ACh/DA/OA/unknown = 兴奋(+1)
    w = cnt_f / (cnt_f.max() + 1e-9)
    w = 0.2 + 0.7 * w
    if nt_f is not None:
        sign = np.where(np.isin(nt_f, ["GABA", "glutamate", "GLUT", "Gly",
                                        "glycine"]), -1.0, 1.0)
        w = w * sign

    return {
        "meta": {
            "source": source or "MaleCNS v1.0 (Cell 2026-09-03)",
            "n_neurons": len(neurons),
            "n_synapses": int(len(src)),
            "min_syn": min_syn,
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "neurons": neurons,
        "synapses": {
            "src": src.tolist(),
            "tgt": tgt.tolist(),
            "w": [round(float(x), 3) for x in w],
        },
    }


def main():
    global args
    ap = argparse.ArgumentParser(description="MaleCNS feather 连接组解析")
    ap.add_argument("--input", help="synapses feather/CSV 文件")
    ap.add_argument("--neurons", help="neurons feather/CSV 文件 (可选)")
    ap.add_argument("--output", default="../data/connectome.json")
    ap.add_argument("--top-neurons", type=int, default=20000,
                    help="保留突触数最多的前 N 神经元")
    ap.add_argument("--min-syn", type=int, default=5, help="最小突触数过滤")
    ap.add_argument("--synthetic", action="store_true",
                    help="无真实数据时用合成 MaleCNS 风格数据演示")
    args = ap.parse_args()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    if args.synthetic or not args.input:
        print("[1/3] 生成合成 MaleCNS 风格数据集 ...")
        syn_d, neu_d = synthetic_dataset()
        import pandas as pd
        df_syn, df_neu = pd.DataFrame(syn_d), pd.DataFrame(neu_d)
    else:
        print(f"[1/3] 读取 {args.input} ...")
        loader = load_feather if args.input.endswith((".feather", ".arrow")) else load_csv
        df_syn = loader(args.input)
        need = {"pre_root_id", "post_root_id", "syn_count"}
        if not need.issubset(df_syn.columns):
            # 兼容 FlyWire 命名: pre_root_id / post_root_id / syn_count
            if {"pre", "post"}.issubset(df_syn.columns):
                df_syn = df_syn.rename(columns={"pre": "pre_root_id",
                                                "post": "post_root_id"})
            else:
                sys.exit(f"缺少必需列 {need}; 实际列: {list(df_syn.columns)}")
        if args.neurons:
            nloader = (load_feather if args.neurons.endswith((".feather", ".arrow"))
                       else load_csv)
            df_neu = nloader(args.neurons)
        else:
            # 从突触表推断神经元集合
            import numpy as np
            ids = np.union1d(df_syn["pre_root_id"].unique(),
                             df_syn["post_root_id"].unique())
            df_neu = {"root_id": ids}

    print(f"[2/3] 过滤 (top {args.top_neurons} 神经元, >= {args.min_syn} 突触) + 6层映射 ...")
    src_label = ("synthetic (MaleCNS-style statistics)" if (args.synthetic or not args.input)
                 else "MaleCNS v1.0 (Cell 2026-09-03)")
    conn = build_connectome(df_syn, df_neu, args.top_neurons, args.min_syn,
                            source=src_label)

    print(f"[3/3] 导出 -> {out}")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(conn, f, ensure_ascii=False)

    m = conn["meta"]
    layers = {}
    for n in conn["neurons"]:
        layers[n["layer"]] = layers.get(n["layer"], 0) + 1
    print(f"      神经元: {m['n_neurons']:,}  突触: {m['n_synapses']:,}")
    print(f"      6层分布: V1={layers.get(0,0):,} V2={layers.get(1,0):,} "
          f"PE={layers.get(2,0):,} IFC={layers.get(3,0):,} "
          f"决策={layers.get(4,0):,} TC={layers.get(5,0):,}")
    size_mb = out.stat().st_size / 1048576
    print(f"      文件大小: {size_mb:.1f} MB")
    print("完成。浏览器端: fetch('data/connectome.json') 即可加载真实拓扑。")


if __name__ == "__main__":
    main()
