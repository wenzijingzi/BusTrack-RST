# -*- coding: utf-8 -*-
"""
06grid_search_oip_tcr.py
- 不训练：只扫 12~20 组 Soft OIP/TCR 超参（通过 ENV 注入 matching/bot_sort）
- 自动：追踪(04) -> 评估(05) -> 汇总每序列增益 + overall 最优组合
- 不改 04/05：04 只传 --det/--cmc；超参全部走 ENV

用法（在 BoT-SORT-main 根目录运行）：
  python tools/BFront_MOT/06grid_search_oip_tcr.py --det tempX --cmc busfront --runs 12
  python tools/BFront_MOT/06grid_search_oip_tcr.py --det tempX --cmc busfront --runs 12 --baseline_tracker m2la_botsort_tempX_busfront_v2
  python tools/BFront_MOT/06grid_search_oip_tcr.py --det tempX --cmc busfront --runs 12 --baseline_tracker m2la_botsort_tempX_busfront_v2

"""

import os
import sys
import time
import shutil
import argparse
import subprocess
from datetime import datetime
from pathlib import Path

import pandas as pd


# -------------------- utils --------------------
def abspath(p: str) -> str:
    return os.path.abspath(os.path.expanduser(p))


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def safe_rmtree(p: str):
    if os.path.isdir(p):
        shutil.rmtree(p, ignore_errors=True)


def move_dir(src: str, dst: str):
    if not os.path.exists(src):
        raise FileNotFoundError(f"Source folder not found: {src}")
    if os.path.exists(dst):
        safe_rmtree(dst)
    shutil.move(src, dst)


def run_cmd(cmd, env=None, cwd=None):
    print("\n[CMD]", " ".join(map(str, cmd)))
    p = subprocess.run(list(map(str, cmd)), env=env, cwd=cwd)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed (ret={p.returncode}): {cmd}")


def infer_default_out_name(det: str, cmc: str) -> str:
    # 04 的默认 tracker 输出目录名
    det = det.strip()
    cmc = cmc.strip()
    return f"m2la_botsort_{det}_{cmc}"


def dynamic_import_eval_module(eval_py_path: str):
    """
    动态 import 05eval_all_trackers_final.py
    并在运行前把 TRACKERS/tracker_names 强制设置为 [tracker_name]
    """
    import importlib.util

    eval_py_path = abspath(eval_py_path)
    spec = importlib.util.spec_from_file_location("eval_m2la_dynamic", eval_py_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore
    return mod


def run_eval_one_tracker(eval_mod, tracker_name: str):
    if hasattr(eval_mod, "TRACKERS"):
        eval_mod.TRACKERS = [tracker_name]
    elif hasattr(eval_mod, "tracker_names"):
        eval_mod.tracker_names = [tracker_name]
    else:
        raise AttributeError("eval script has neither TRACKERS nor tracker_names")
    eval_mod.main()


def read_overall_row(summary_csv: str) -> dict:
    """
    兼容两类 summary.csv：
    A) 第一列是序列名（BF_01... OVERALL），其余是指标列
    B) 没有 index，最后一行是 OVERALL
    """
    df = pd.read_csv(summary_csv)
    # 如果第一列像 seq 名，就把它设成 index
    if df.shape[1] >= 2:
        first_col = df.columns[0]
        if df[first_col].astype(str).str.startswith("BF_").any() or df[first_col].astype(str).isin(["OVERALL", "COMBINED"]).any():
            df = df.set_index(first_col)
            if "OVERALL" in df.index:
                return df.loc["OVERALL"].to_dict()
            # 有些版本是 COMBINED
            if "COMBINED" in df.index:
                return df.loc["COMBINED"].to_dict()
            return df.iloc[-1].to_dict()

    # fallback：直接最后一行
    return df.iloc[-1].to_dict()


def read_hota_overall(hota_csv: str):
    """
    hota_per_seq.csv：一般含 BF_01... + COMBINED/OVERALL，且含列 HOTA/DetA/AssA
    """
    if not os.path.isfile(hota_csv):
        return float("nan"), float("nan"), float("nan")

    df = pd.read_csv(hota_csv)
    if df.shape[1] >= 2:
        first_col = df.columns[0]
        df = df.set_index(first_col)

    if "OVERALL" in df.index:
        r = df.loc["OVERALL"]
    elif "COMBINED" in df.index:
        r = df.loc["COMBINED"]
    else:
        r = df.iloc[-1]

    def get_any(row, keys):
        for k in keys:
            if k in row.index:
                return float(row[k])
        return float("nan")

    hota = get_any(r, ["HOTA", "hota", "HOTA_mean", "HOTA(0)"])
    deta = get_any(r, ["DetA", "deta"])
    assa = get_any(r, ["AssA", "assa"])
    return hota, deta, assa


def pick_key(d: dict, keys):
    for k in keys:
        if k in d:
            return k
    return None


# -------------------- grid (最小 12 组，更贴合你当前尺度) --------------------
def build_param_grid(max_runs: int):
    """
    你当前最有效的点在：alpha≈0.25, beta≈0.20, trigger_iou≈0.35, small_norm≈0.010
    所以用“围绕最优点的小范围微调”最划算，不要跑到 0.05/0.03 那种太弱的区间。
    """
    grid = [
        # 以你 v1 为中心（主峰附近）
        {"OIP_ALPHA": 0.22, "TCR_BETA": 0.18, "TRIGGER_IOU": 0.35, "SMALL_NORM": 0.010},
        {"OIP_ALPHA": 0.25, "TCR_BETA": 0.18, "TRIGGER_IOU": 0.35, "SMALL_NORM": 0.010},
        {"OIP_ALPHA": 0.28, "TCR_BETA": 0.18, "TRIGGER_IOU": 0.35, "SMALL_NORM": 0.010},

        {"OIP_ALPHA": 0.22, "TCR_BETA": 0.20, "TRIGGER_IOU": 0.35, "SMALL_NORM": 0.010},
        {"OIP_ALPHA": 0.25, "TCR_BETA": 0.20, "TRIGGER_IOU": 0.35, "SMALL_NORM": 0.010},  # 你当前最优点
        {"OIP_ALPHA": 0.28, "TCR_BETA": 0.20, "TRIGGER_IOU": 0.35, "SMALL_NORM": 0.010},

        {"OIP_ALPHA": 0.22, "TCR_BETA": 0.22, "TRIGGER_IOU": 0.35, "SMALL_NORM": 0.010},
        {"OIP_ALPHA": 0.25, "TCR_BETA": 0.22, "TRIGGER_IOU": 0.35, "SMALL_NORM": 0.010},
        {"OIP_ALPHA": 0.28, "TCR_BETA": 0.22, "TRIGGER_IOU": 0.35, "SMALL_NORM": 0.010},

        # trigger_iou & small_norm 的小幅扰动（更贴合 BusFront 小目标/遮挡）
        {"OIP_ALPHA": 0.25, "TCR_BETA": 0.20, "TRIGGER_IOU": 0.30, "SMALL_NORM": 0.010},
        {"OIP_ALPHA": 0.25, "TCR_BETA": 0.20, "TRIGGER_IOU": 0.40, "SMALL_NORM": 0.010},
        {"OIP_ALPHA": 0.25, "TCR_BETA": 0.20, "TRIGGER_IOU": 0.35, "SMALL_NORM": 0.012},
    ]
    return grid[:max_runs]


# -------------------- main --------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--det", type=str, default="tempX", choices=["raw", "tempX"], help="det variant")
    ap.add_argument("--cmc", type=str, default="busfront", help="cmc method (busfront/none/orb/ecc/...)")
    ap.add_argument("--runs", type=int, default=12, help="grid size (12~20 recommended)")
    ap.add_argument("--baseline_tracker", type=str, default="", help="baseline tracker name for delta computation")
    ap.add_argument("--track_py", type=str, default="tools/BFront_MOT/04track_external_dets_factorial.py")
    ap.add_argument("--eval_py", type=str, default="tools/BFront_MOT/05eval_all_trackers_final.py")
    ap.add_argument("--work_dir", type=str, default="", help="BoT-SORT-main root. empty = current cwd")
    ap.add_argument("--python", type=str, default=sys.executable, help="python executable")
    args = ap.parse_args()

    work_dir = abspath(args.work_dir) if args.work_dir else os.getcwd()
    track_py = abspath(os.path.join(work_dir, args.track_py))
    eval_py = abspath(os.path.join(work_dir, args.eval_py))

    if not os.path.isfile(track_py):
        raise FileNotFoundError(f"track script not found: {track_py}")
    if not os.path.isfile(eval_py):
        raise FileNotFoundError(f"eval script not found: {eval_py}")

    # 关键：04 的输出在 work_dir/YOLOX_outputs/ 下
    yolox_root = os.path.join(work_dir, "YOLOX_outputs/grid_search_oip_tcr")
    ensure_dir(yolox_root)

    default_out_name = infer_default_out_name(args.det, args.cmc)
    default_out_dir = os.path.join(yolox_root, default_out_name)

    # 关键：05 的真实输出目录（你现在用的是 eval_m2la_tempX_busfront）
    # 如果你 05 写死了 OUT_DIR，则它会自动写这里；我们只需要保证读取这里。
    eval_out_dir = os.path.join(yolox_root, "E:/track/BoT-SORT-main/YOLOX_outputs/grid_search_oip_tcr/eval_m2la_tempX_busfront")
    ensure_dir(eval_out_dir)

    # grid-search 保存目录
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    gs_root = os.path.join(yolox_root, "grid_search_oip_tcr", f"gridsearch_{args.det}_{args.cmc}_{stamp}")
    ensure_dir(gs_root)

    # baseline
    baseline_overall = None
    baseline_hota = None
    if args.baseline_tracker:
        base_sum = os.path.join(eval_out_dir, f"{args.baseline_tracker}_summary.csv")
        base_hota_csv = os.path.join(eval_out_dir, f"{args.baseline_tracker}_hota_per_seq.csv")
        if os.path.isfile(base_sum):
            baseline_overall = read_overall_row(base_sum)
        if os.path.isfile(base_hota_csv):
            baseline_hota, _, _ = read_hota_overall(base_hota_csv)

    # 动态加载 eval
    eval_mod = dynamic_import_eval_module(eval_py)

    # build grid
    runs = max(1, min(int(args.runs), 30))
    grid = build_param_grid(runs)
    if not grid:
        raise RuntimeError("Empty grid.")

    all_rows = []

    print("\n========== GRID SEARCH START ==========")
    print("work_dir        :", work_dir)
    print("track_py        :", track_py)
    print("eval_py         :", eval_py)
    print("yolox_root      :", yolox_root)
    print("default_out_dir :", default_out_dir)
    print("eval_out_dir    :", eval_out_dir)
    print("gs_root         :", gs_root)
    print("runs            :", len(grid))
    if args.baseline_tracker:
        print("baseline        :", args.baseline_tracker)

    for idx, hp in enumerate(grid, 1):
        run_tag = f"gs{idx:02d}_a{hp['OIP_ALPHA']}_b{hp['TCR_BETA']}_tiou{hp['TRIGGER_IOU']}_sn{hp['SMALL_NORM']}"
        tracker_name = f"{default_out_name}_{run_tag}"

        print("\n--------------------------------------")
        print(f"[{idx}/{len(grid)}] {tracker_name}")
        print("HP:", hp)

        # 1) 清理默认输出目录（避免旧 tracks 叠加）
        if os.path.isdir(default_out_dir):
            safe_rmtree(default_out_dir)

        # 2) env 注入（matching/bot_sort 必须读取这些 env）
        env = os.environ.copy()
        env["BFRONT_SOFT_ENABLE"] = "1"
        env["BFRONT_SOFT_OIP_ALPHA"] = str(hp["OIP_ALPHA"])
        env["BFRONT_SOFT_TCR_BETA"] = str(hp["TCR_BETA"])
        env["BFRONT_SOFT_TRIGGER_IOU"] = str(hp["TRIGGER_IOU"])
        env["BFRONT_SMALL_NORM"] = str(hp["SMALL_NORM"])

        # 3) 跑追踪（注意：04 只传 det/cmc）
        cmd_track = [args.python, track_py, "--det", args.det, "--cmc", args.cmc]
        t0 = time.time()
        run_cmd(cmd_track, env=env, cwd=work_dir)
        t1 = time.time()

        # 4) 把默认输出目录重命名为唯一 tracker_name（避免覆盖）
        if not os.path.isdir(default_out_dir):
            raise RuntimeError(f"Tracking finished but output folder missing: {default_out_dir}")

        run_out_dir = os.path.join(yolox_root, tracker_name)
        move_dir(default_out_dir, run_out_dir)

        # 5) 跑评估（只评估当前 tracker）
        run_eval_one_tracker(eval_mod, tracker_name)

        # 6) 读取评估结果
        sum_csv = os.path.join(eval_out_dir, f"{tracker_name}_summary.csv")
        hota_csv = os.path.join(eval_out_dir, f"{tracker_name}_hota_per_seq.csv")

        if not os.path.isfile(sum_csv):
            raise FileNotFoundError(f"Missing summary: {sum_csv}")

        overall = read_overall_row(sum_csv)

        # 兼容列名
        hota_key = pick_key(overall, ["HOTA", "hota", "HOTA_mean", "HOTA(0)"])
        idf1_key = pick_key(overall, ["IDF1", "idf1"])
        ids_key = pick_key(overall, ["num_switches", "IDs", "ids", "num_ids"])
        mota_key = pick_key(overall, ["mota", "MOTA"])
        fp_key = pick_key(overall, ["num_false_positives", "FP", "fp"])
        fn_key = pick_key(overall, ["num_misses", "FN", "fn"])

        hota_val = float(overall[hota_key]) if hota_key else float("nan")
        idf1_val = float(overall[idf1_key]) if idf1_key else float("nan")
        ids_val = float(overall[ids_key]) if ids_key else float("nan")
        mota_val = float(overall[mota_key]) if mota_key else float("nan")
        fp_val = float(overall[fp_key]) if fp_key else float("nan")
        fn_val = float(overall[fn_key]) if fn_key else float("nan")

        hota_overall2, deta_overall2, assa_overall2 = read_hota_overall(hota_csv)

        row = {
            "tracker": tracker_name,
            "run_tag": run_tag,
            "OIP_ALPHA": hp["OIP_ALPHA"],
            "TCR_BETA": hp["TCR_BETA"],
            "TRIGGER_IOU": hp["TRIGGER_IOU"],
            "SMALL_NORM": hp["SMALL_NORM"],
            "time_sec": round(t1 - t0, 2),
            "MOTA": mota_val,
            "IDF1": idf1_val,
            "IDs": ids_val,
            "FP": fp_val,
            "FN": fn_val,
            "HOTA_from_summary": hota_val,
            "HOTA": hota_overall2,
            "DetA": deta_overall2,
            "AssA": assa_overall2,
        }

        # overall 增益
        if baseline_overall is not None:
            base_mota = float(baseline_overall.get(mota_key, float("nan"))) if mota_key else float("nan")
            base_idf1 = float(baseline_overall.get(idf1_key, float("nan"))) if idf1_key else float("nan")
            base_ids = float(baseline_overall.get(ids_key, float("nan"))) if ids_key else float("nan")

            row["dMOTA"] = row["MOTA"] - base_mota
            row["dIDF1"] = row["IDF1"] - base_idf1
            row["dIDs"] = base_ids - row["IDs"]  # IDs 越小越好：用“减少量”表示增益

        if baseline_hota is not None and not pd.isna(row["HOTA"]):
            row["dHOTA"] = row["HOTA"] - float(baseline_hota)

        all_rows.append(row)

        # 7) 保存运行中结果（防止中断丢数据）
        pd.DataFrame(all_rows).to_csv(os.path.join(gs_root, "grid_results_running.csv"), index=False, encoding="utf-8-sig")

    # 排序选最优：优先 HOTA，其次 IDs（小优），再 IDF1、MOTA
    df = pd.DataFrame(all_rows)
    df_sorted = df.sort_values(by=["HOTA", "IDs", "IDF1", "MOTA"], ascending=[False, True, False, False])

    out_all = os.path.join(gs_root, "grid_results_all.csv")
    out_best = os.path.join(gs_root, "grid_best.csv")
    df_sorted.to_csv(out_all, index=False, encoding="utf-8-sig")
    df_sorted.head(1).to_csv(out_best, index=False, encoding="utf-8-sig")

    print("\n========== GRID SEARCH DONE ==========")
    print("Saved:", out_all)
    print("Best :", out_best)
    print("\n[BEST]\n", df_sorted.head(1).to_string(index=False))


if __name__ == "__main__":
    main()
