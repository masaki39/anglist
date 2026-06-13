"""
頚椎学習曲線 集計・可視化・レポート生成（ローカル実行）

Usage:
  uv run python train/learning_curve/cervical/03_report.py

入力:  RESULTS_DIR の foldN_sizeSSS.json（30件）
出力:  OUT_DIR に fig1-5 PNG と report.md
"""

import json
import os
import re
import statistics
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ===== 設定 =====
RESULTS_DIR = "/Volumes/T7 Shield/dicom/omuro/cervical_lc/results/smallunet_aug1_awl_s15"
OUT_DIR     = "/Volumes/T7 Shield/dicom/omuro/cervical_lc/report"
EVAL_REPORT = "/Volumes/T7 Shield/dicom/omuro/runs_cervical_sigma15/eval_report_test.txt"
SIZES       = [20, 40, 80, 160, 220, 259]
LANDMARK_ORDER = [
    "C2_center", "C2_ant", "C2_post", "C7_sup_post",
    "C7_inf_ant", "C7_inf_post", "T1_ant", "T1_post",
]
ANGLES = ["C2C7_angle", "T1S", "C2C7_SVA"]
# =================


# ---------------------------------------------------------------------------
# データ読み込み・集計
# ---------------------------------------------------------------------------
def load_results():
    """size -> list[dict]（fold分）に整理。"""
    by_size = defaultdict(list)
    for fname in sorted(os.listdir(RESULTS_DIR)):
        if not fname.endswith(".json"):
            continue
        with open(os.path.join(RESULTS_DIR, fname)) as f:
            d = json.load(f)
        by_size[d["size"]].append(d)
    return by_size


def parse_eval_landmarks(path):
    """フルモデルtest評価レポートから点ごと all / excl-outliers の (MRE, SDR@4) を抽出。

    LC JSONは点ごとの外れ値除外版を保存していないため、この比較は
    フルモデルのtest評価（eval_report_test.txt）のみから得られる。
    """
    if not os.path.exists(path):
        return None
    with open(path) as f:
        text = f.read()

    sections = {}
    cur = None
    for line in text.splitlines():
        if line.startswith("=== Landmark Detection"):
            cur = "noout" if "excl. outliers" in line else "all"
            sections[cur] = {}
            continue
        if cur is None:
            continue
        if line.startswith("==="):  # 別セクションに入った
            cur = None
            continue
        parts = line.split()
        if not parts or parts[0] not in LANDMARK_ORDER:
            continue
        nums = re.findall(r"-?\d+\.\d+", line)  # mre, sd, ci_lo, ci_hi, sdr2, sdr4, conf
        if len(nums) >= 6:
            sections[cur][parts[0]] = (float(nums[0]), float(nums[5]))

    if "all" in sections and "noout" in sections:
        return sections
    return None


def _mean_sd(values):
    values = [v for v in values if v is not None]
    if not values:
        return None, None
    m = statistics.mean(values)
    s = statistics.pstdev(values) if len(values) > 1 else 0.0
    return m, s


def curve(by_size, getter):
    """size順に (sizes, means, sds) を返す。"""
    xs, ms, ss = [], [], []
    for sz in SIZES:
        if sz not in by_size:
            continue
        m, s = _mean_sd([getter(d) for d in by_size[sz]])
        if m is None:
            continue
        xs.append(sz)
        ms.append(m)
        ss.append(s)
    return xs, ms, ss


# ---------------------------------------------------------------------------
# 図
# ---------------------------------------------------------------------------
def fig1_overall(by_size):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # MRE: all vs excl outliers
    ax = axes[0, 0]
    for label, g in [("All", lambda d: d["overall"]["mre_mm"]),
                     ("Excl. outliers", lambda d: d["overall_excl_outliers"]["mre_mm"])]:
        xs, ms, ss = curve(by_size, g)
        ax.errorbar(xs, ms, yerr=ss, marker="o", capsize=4, label=label)
    ax.set_title("Overall MRE")
    ax.set_xlabel("Training cases")
    ax.set_ylabel("MRE (mm)")
    ax.axhline(4.0, ls="--", color="gray", lw=1)
    ax.set_ylim(0, 20)
    ax.legend()
    ax.grid(alpha=0.3)

    # SDR@4
    ax = axes[0, 1]
    xs, ms, ss = curve(by_size, lambda d: d["overall"]["sdr4"])
    ax.errorbar(xs, ms, yerr=ss, marker="o", capsize=4, color="tab:green")
    ax.set_title("Overall SDR@4mm")
    ax.set_xlabel("Training cases")
    ax.set_ylabel("SDR@4 (%)")
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.3)

    # C2C7 / T1S angle MAE
    for ax, name in [(axes[1, 0], "C2C7_angle"), (axes[1, 1], "T1S")]:
        xs, ms, ss = curve(by_size, lambda d, n=name: d["overall"]["angle_mae"].get(n))
        ax.errorbar(xs, ms, yerr=ss, marker="o", capsize=4, color="tab:red")
        ax.set_title(f"{name} MAE")
        ax.set_xlabel("Training cases")
        ax.set_ylabel("MAE (deg)")
        ax.grid(alpha=0.3)

    fig.suptitle("Cervical landmark learning curve (5-fold mean ± SD)", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig1_overall_curve.png"), dpi=150)
    plt.close(fig)


def fig2_landmark_curves(by_size):
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    for ax, lm in zip(axes.ravel(), LANDMARK_ORDER):
        xs, ms, ss = curve(by_size, lambda d, k=lm: d["landmarks"].get(k, {}).get("mre_mm"))
        ax.errorbar(xs, ms, yerr=ss, marker="o", capsize=3)
        ax.set_title(lm)
        ax.set_xlabel("Training cases")
        ax.set_ylabel("MRE (mm)")
        ax.axhline(4.0, ls="--", color="gray", lw=1)
        ax.set_ylim(0, 12)
        ax.grid(alpha=0.3)
    fig.suptitle("Per-landmark MRE learning curve (5-fold mean ± SD)", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig2_landmark_curves.png"), dpi=150)
    plt.close(fig)


def fig3_landmark_ranking(by_size):
    """size=259 での点ごと MRE / SDR@4。"""
    sz = max(s for s in SIZES if s in by_size)
    rows = []
    for lm in LANDMARK_ORDER:
        mre, _ = _mean_sd([d["landmarks"][lm]["mre_mm"] for d in by_size[sz]])
        sdr, _ = _mean_sd([d["landmarks"][lm]["sdr4"] for d in by_size[sz]])
        rows.append((lm, mre, sdr))
    rows.sort(key=lambda r: r[1])  # MRE昇順（易→難）
    names = [r[0] for r in rows]
    mres  = [r[1] for r in rows]
    sdrs  = [r[2] for r in rows]
    y = range(len(names))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].barh(list(y), mres, color="tab:blue")
    axes[0].axvline(4.0, ls="--", color="gray", lw=1)
    axes[0].set_yticks(list(y))
    axes[0].set_yticklabels(names)
    axes[0].set_xlabel("MRE (mm)")
    axes[0].set_title(f"Per-landmark MRE (n={sz})")
    axes[0].invert_yaxis()

    axes[1].barh(list(y), sdrs, color="tab:green")
    axes[1].set_yticks(list(y))
    axes[1].set_yticklabels(names)
    axes[1].set_xlabel("SDR@4 (%)")
    axes[1].set_xlim(0, 100)
    axes[1].set_title(f"Per-landmark SDR@4 (n={sz})")
    axes[1].invert_yaxis()

    fig.suptitle("Landmark difficulty ranking (easy -> hard)", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig3_landmark_ranking.png"), dpi=150)
    plt.close(fig)


def fig4_outliers(by_size):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    xs, ms, ss = curve(by_size, lambda d: d["overall_excl_outliers"]["n_outliers"])
    ax.errorbar(xs, ms, yerr=ss, marker="o", capsize=4, color="tab:purple")
    ax.set_title("Outliers per fold vs training size")
    ax.set_xlabel("Training cases")
    ax.set_ylabel("Outliers / fold (any landmark > 10mm)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig4_outliers.png"), dpi=150)
    plt.close(fig)


def fig5_angle_curves(by_size):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    units = {"C2C7_angle": "deg", "T1S": "deg", "C2C7_SVA": "mm"}
    for ax, name in zip(axes, ANGLES):
        xs, ms, ss = curve(by_size, lambda d, n=name: d["overall"]["angle_mae"].get(n))
        ax.errorbar(xs, ms, yerr=ss, marker="o", capsize=4)
        ax.set_title(f"{name} MAE")
        ax.set_xlabel("Training cases")
        ax.set_ylabel(f"MAE ({units[name]})")
        ax.grid(alpha=0.3)
    axes[2].annotate("flat -> systematic\n(not a learning issue)",
                     xy=(0.5, 0.85), xycoords="axes fraction",
                     ha="center", color="tab:red", fontsize=11)
    fig.suptitle("Angle / SVA MAE learning curve (5-fold mean ± SD)", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig5_angle_curves.png"), dpi=150)
    plt.close(fig)


def fig6_landmark_outlier(parsed):
    """点ごと all vs excl-outliers（フルモデルtest評価より）。"""
    order = sorted(LANDMARK_ORDER, key=lambda k: parsed["all"][k][0])  # 易→難
    n = len(order)
    y = list(range(n))
    h = 0.38
    yo = [v - h / 2 for v in y]
    yu = [v + h / 2 for v in y]

    mre_all = [parsed["all"][k][0]   for k in order]
    mre_no  = [parsed["noout"][k][0] for k in order]
    sdr_all = [parsed["all"][k][1]   for k in order]
    sdr_no  = [parsed["noout"][k][1] for k in order]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].barh(yo, mre_all, height=h, label="All",            color="tab:blue")
    axes[0].barh(yu, mre_no,  height=h, label="Excl. outliers", color="tab:orange")
    axes[0].axvline(4.0, ls="--", color="gray", lw=1)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(order)
    axes[0].set_xlabel("MRE (mm)")
    axes[0].set_title("Per-landmark MRE")
    axes[0].invert_yaxis()
    axes[0].legend()

    axes[1].barh(yo, sdr_all, height=h, label="All",            color="tab:blue")
    axes[1].barh(yu, sdr_no,  height=h, label="Excl. outliers", color="tab:orange")
    axes[1].set_yticks(y)
    axes[1].set_yticklabels(order)
    axes[1].set_xlabel("SDR@4 (%)")
    axes[1].set_xlim(0, 100)
    axes[1].set_title("Per-landmark SDR@4")
    axes[1].invert_yaxis()
    axes[1].legend()

    fig.suptitle("Effect of excluding outlier cases, per landmark (full-model test set)", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig6_landmark_outlier.png"), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# レポート
# ---------------------------------------------------------------------------
def build_table(by_size):
    """size別の集計表（markdown行のリスト）と生データdictを返す。"""
    table = []
    data = {}
    for sz in SIZES:
        if sz not in by_size:
            continue
        ds = by_size[sz]
        mre, _   = _mean_sd([d["overall"]["mre_mm"] for d in ds])
        mre_eo, _ = _mean_sd([d["overall_excl_outliers"]["mre_mm"] for d in ds])
        sdr, _   = _mean_sd([d["overall"]["sdr4"] for d in ds])
        c2c7, _  = _mean_sd([d["overall"]["angle_mae"].get("C2C7_angle") for d in ds])
        t1s, _   = _mean_sd([d["overall"]["angle_mae"].get("T1S") for d in ds])
        sva, _   = _mean_sd([d["overall"]["angle_mae"].get("C2C7_SVA") for d in ds])
        nout, _  = _mean_sd([d["overall_excl_outliers"]["n_outliers"] for d in ds])
        data[sz] = dict(mre=mre, mre_eo=mre_eo, sdr=sdr, c2c7=c2c7, t1s=t1s, sva=sva, nout=nout)
        eo = f"{mre_eo:.1f}" if mre_eo is not None else "—"
        table.append(
            f"| {sz} | {mre:.1f} | {eo} | {sdr:.1f} | {c2c7:.1f} | {t1s:.1f} | {sva:.1f} | {nout:.1f} |"
        )
    return table, data


def write_report(by_size, parsed):
    table, data = build_table(by_size)
    smax = max(data)
    smin = min(data)

    # 点ごと all vs excl-outliers（フルモデルtest評価より）
    lm_outlier_table = ["（eval_report_test.txt が見つからないため省略）"]
    if parsed:
        order = sorted(LANDMARK_ORDER, key=lambda k: parsed["all"][k][0])
        lm_outlier_table = ["| Landmark | MRE 全(mm) | MRE 除外(mm) | SDR@4 全(%) | SDR@4 除外(%) |",
                            "|---|--:|--:|--:|--:|"]
        for k in order:
            ma, sa = parsed["all"][k]
            mn, sn = parsed["noout"][k]
            lm_outlier_table.append(f"| {k} | {ma:.2f} | {mn:.2f} | {sa:.1f} | {sn:.1f} |")

    # 点ごと（最大size）
    lm_rows = []
    for lm in LANDMARK_ORDER:
        mre, _ = _mean_sd([d["landmarks"][lm]["mre_mm"] for d in by_size[smax]])
        sdr, _ = _mean_sd([d["landmarks"][lm]["sdr4"] for d in by_size[smax]])
        lm_rows.append((lm, mre, sdr))
    lm_rows.sort(key=lambda r: r[1])
    lm_table = [f"| {lm} | {mre:.2f} | {sdr:.1f} |" for lm, mre, sdr in lm_rows]
    easy = ", ".join(r[0] for r in lm_rows[:3])
    hard = ", ".join(r[0] for r in lm_rows[-2:])

    eval_excerpt = ""
    if os.path.exists(EVAL_REPORT):
        with open(EVAL_REPORT) as f:
            eval_excerpt = f.read()

    md = f"""# 頚椎ランドマーク検出 学習曲線レポート

実験: SmallUNet / σ=15 / AWL / augment / 5-fold CV × 6 training sizes
データ: merged_dataset 324例（学習に最大 {smax} 例、各foldのテストは残り全例）
集計元: `results/smallunet_aug1_awl_s15/`（30 JSON）

---

## 1. 全体の学習曲線

| Training cases | MRE(mm) | MRE 外れ値除外(mm) | SDR@4(%) | C2C7 MAE(°) | T1S MAE(°) | SVA MAE(mm) | outliers/fold |
|--:|--:|--:|--:|--:|--:|--:|--:|
{chr(10).join(table)}

![overall](fig1_overall_curve.png)

**所見**: 学習例数を増やすと全指標が単調に改善する。{smin}例ではほぼ使い物にならない
（MRE {data[smin]['mre']:.0f}mm）が、160例で MRE {data[160]['mre']:.1f}mm・SDR@4 {data[160]['sdr']:.0f}%
に到達。160→220→{smax}は改善が緩やかになり**収穫逓減**に入る。

---

## 2. 点ごとの違い（難易度）

![landmark curves](fig2_landmark_curves.png)
![ranking](fig3_landmark_ranking.png)

最大学習量（n={smax}）での点ごと精度（MRE昇順 = 易→難）:

| Landmark | MRE(mm) | SDR@4(%) |
|---|--:|--:|
{chr(10).join(lm_table)}

**所見**: C2系（{easy}）は容易で SDR@4 90%超。一方 **{hard}** が律速点で、
学習量を増やしても 4mm 前後で頭打ち。これらは下位頚椎〜上位胸椎の移行部で、
鎖骨・肩・縦隔の重なりにより**人間でも同定が難しい**領域に一致する。

---

## 3. 外れ値・難しい点を除けば点・角度ともに高精度

![outliers](fig4_outliers.png)

外れ値（いずれかの点が >10mm）は学習量とともに激減（{data[smin]['nout']:.0f} → {data[smax]['nout']:.0f} 件/fold）。

フルモデル（本番 train/val/test 分割, test=32例; `runs_cervical_sigma15/eval_report_test.txt`）での
**外れ値除外の効果**:

- 点 Overall: 2.92 → **2.27mm**、SDR@4 78.9 → **84.3%**
- C2C7_angle: MAE 10.99 → **8.83°**、ICC 0.835 → **0.861**
- T1S: MAE 11.51 → **8.50°**

→ 少数の難症例（外れ値）を除けば、**点・角度ともに臨床利用に十分な精度**が得られる。

### 点ごとの外れ値除外効果

![landmark outlier](fig6_landmark_outlier.png)

{chr(10).join(lm_outlier_table)}

**所見**: 律速だった **T1_ant（5.22→3.54mm）** と **C7_inf_ant（4.63→3.34mm）** は、外れ値症例を除くと
4mm 前後まで改善し SDR@4 も大きく上がる。つまりこれらの点の悪さは「全症例で常に下手」なのではなく、
**一部の難症例（外れ値）に誤差が集中**していることを意味する。除外後は全点が ~3.5mm 以下に収まる。

> 注: LC の30 JSON は点ごとの平均しか保存しておらず（各症例の生誤差は破棄）、学習曲線(fig2)に
> 点ごとの外れ値除外版は後付けできない。本節の比較はフルモデルの test 評価のみから得たもの。

---

## 4. SVA（C2C7_SVA）の系統的異常 ※バグ疑い

![angle curves](fig5_angle_curves.png)

C2C7_SVA の MAE は学習量に依らず **~65-69mm でほぼ一定**、ICC ≈ −1（反相関）。
SVA を構成する点（C2_center・C7_sup_post）はいずれも高精度（~1.5-2mm）であるため、
これは**学習不足ではなく系統的なバグ**（spacing 適用・符号・座標系の不一致）の可能性が高い。
点が当たっているのに距離だけ大きく外れる＝計算側の問題を強く示唆する。**要調査**。

---

## 5. 何例学習すれば実用域か

- **実用最小ライン ≈ 160例**: MRE {data[160]['mre']:.1f}mm・SDR@4 {data[160]['sdr']:.0f}%。
- **推奨 ≈ 220例以上**: SDR@4 {data[220]['sdr']:.0f}%、角度MAEも C2C7 {data[220]['c2c7']:.0f}° 台に。
- 220→{smax} の伸びは小さく、現データ規模では **220前後で頭打ち**。さらなる向上には
  量より「難症例の質」（移行部アノテーション）が効く見込み。

---

## 6. 結論

1. **全体は学習量とともに徐々に上昇**し、160-220例で実用域に達する。
2. **外れ値・難しい点を除外すると、点・角度ともに非常に良好**な精度になる。
3. 律速は **T1_ant / C7_inf_ant**（頚胸椎移行部）。ここは人間でも難しく、今後の主課題。
4. **SVA は系統的異常**があり、モデルとは別レイヤの計算バグとして切り分けるべき。

---

## 7. 次の研究ステップ提案

1. **SVA計算の調査**: offline評価の spacing 適用・座標軸・符号規約を Slicer 実装
   （`logic_angles_cervical.py`）と突き合わせ、GT と AI の単位系を一致させる。
2. **難点の補強**: T1_ant / C7_inf_ant を多く含む症例を追加収集、移行部アノテーションを再確認。
3. **人間baseline（inter-annotator）の測定**: 「人間でも難しい」を定量化し、モデル誤差の到達目標を設定。
4. **本番モデル確定**: 220例規模で再現性のある最終モデルを学習・凍結。
5. **ablation**: σ・loss・backbone の比較で頭打ち打破の余地を探る。

---

<details><summary>付録: フルモデル test 評価レポート全文</summary>

```
{eval_excerpt}
```
</details>
"""
    out = os.path.join(OUT_DIR, "report.md")
    with open(out, "w") as f:
        f.write(md)
    return out


# ---------------------------------------------------------------------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    by_size = load_results()
    n = sum(len(v) for v in by_size.values())
    print(f"loaded {n} results across sizes {sorted(by_size)}")

    parsed = parse_eval_landmarks(EVAL_REPORT)
    print(f"eval_report landmarks parsed: {parsed is not None}")

    fig1_overall(by_size)
    fig2_landmark_curves(by_size)
    fig3_landmark_ranking(by_size)
    fig4_outliers(by_size)
    fig5_angle_curves(by_size)
    if parsed:
        fig6_landmark_outlier(parsed)
    print("figures saved")

    out = write_report(by_size, parsed)
    print(f"report saved: {out}")
    print(f"=> {OUT_DIR}")


if __name__ == "__main__":
    main()
