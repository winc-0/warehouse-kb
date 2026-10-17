#!/usr/bin/env python3
# 从扫描结果中挑选待导入文档：按品类配额 + 体积(厚度)优先，排除已知失败文件
# 输出到待导入目录（硬链接，不占额外空间）+ 清单日志
#
# 用法（路径可用环境变量覆盖，默认落在项目内 logs/ 与 data/doc_pick/）：
#   SCAN_CSV=logs/scan_doc.csv PICK_DIR=data/doc_pick python scripts/pick_doc.py
import os, csv, shutil, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCAN_CSV = os.environ.get("SCAN_CSV", str(ROOT / "logs" / "scan_doc.csv"))
PICK_DIR = os.environ.get("PICK_DIR", str(ROOT / "data" / "doc_pick"))
PICK_LOG = os.environ.get("PICK_LOG", str(ROOT / "logs" / "pick_doc.txt"))

# 各品类配额（网络设备/路由器已在库 11 份，域饱和，不再新增）
QUOTA = {
    "打印机(联想/奔图/兄弟)": 6,
    "华为笔记本": 4,
    "平板": 2,
    "显示器": 3,
    "华为打印机": 1,
    "华为台式/一体机": 4,
    "华为台式": 2,
    "其他": 3,
}
# 已知 MinerU 解析必失败的文件（不要浪费额度）
EXCLUDE_KEYWORDS = ["CS2610DNW"]

def main():
    rows = []
    with open(SCAN_CSV, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if r["in_library"].strip().lower() in ("true", "1", "yes"):
                continue
            if any(k.lower() in r["filename"].lower() for k in EXCLUDE_KEYWORDS):
                continue
            rows.append(r)

    # 按品类分组，组内按体积降序（体积=厚度代理，页数估算对部分PDF不可靠）
    from collections import defaultdict
    by_cat = defaultdict(list)
    for r in rows:
        by_cat[r["category"]].append(r)
    for c in by_cat:
        by_cat[c].sort(key=lambda x: -float(x["size_mb"]))

    picked = []
    for cat, n in QUOTA.items():
        picked.extend(by_cat.get(cat, [])[:n])

    os.makedirs(PICK_DIR, exist_ok=True)
    made, missing = [], []
    for r in picked:
        src = os.path.join(r"D:\doc", r["filename"])
        dst = os.path.join(PICK_DIR, r["filename"])
        if not os.path.exists(src):
            missing.append(r["filename"]); continue
        if os.path.exists(dst):
            os.remove(dst)
        try:
            os.link(src, dst)      # 硬链接，同盘不占空间
        except Exception:
            shutil.copy2(src, dst) # 不支持则复制
        made.append(r)

    with open(PICK_LOG, "w", encoding="utf-8") as f:
        f.write(f"共挑选 {len(made)} 个文档 -> {PICK_DIR}\n")
        f.write("规则：排除已在库13份 + 排除MinerU失败文件；按品类配额、组内按体积降序\n\n")
        for r in made:
            f.write(f'{r["category"]:<24}{r["size_mb"]:>8}MB  ~{r["est_pages"]:>5}页  {r["filename"]}\n')
        if missing:
            f.write("\n缺失未链接：\n")
            for m in missing:
                f.write(" - " + m + "\n")

    print(f"挑选完成：{len(made)} 个硬链接到 {PICK_DIR}")
    if missing:
        print(f"[warn] {len(missing)} 个源文件缺失")
    for r in made:
        print(f'{r["category"]:<24}{r["size_mb"]:>8}MB  ~{r["est_pages"]:>5}页  {r["filename"]}')
    print(f"\n清单已写入 {PICK_LOG}")

if __name__ == "__main__":
    main()
