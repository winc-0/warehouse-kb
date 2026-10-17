#!/usr/bin/env python3
# 扫描候选语料目录的 PDF：体积 / 估算页数 / 品类 / 是否已在库
# 纯标准库实现，不依赖第三方包。
#
# 用法（语料目录与库内标题清单因机器而异，均通过环境变量传入）：
#   DOC_DIR=/path/to/doc  FILE_DIR=/path/to/file  python scripts/scan_doc.py
#   LIB_TITLES_JSON=logs/lib_titles.json 指定库内标题清单（由 dump_lib_titles.py 生成）
import os, re, glob, csv, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOC_DIR = os.environ.get("DOC_DIR", "")
FILE_DIR = os.environ.get("FILE_DIR", "")
OUT_CSV = os.environ.get("OUT_CSV", str(ROOT / "logs" / "scan_doc.csv"))

def norm_name(basename: str) -> str:
    # 去掉扩展名 + 折叠空白 + 转小写
    n = os.path.splitext(basename)[0]
    n = re.sub(r"\s+", "", n).lower()
    return n

def collect_file_stems(root: str):
    stems = set()
    for p in glob.glob(os.path.join(root, "**", "*.pdf"), recursive=True):
        stems.add(norm_name(os.path.basename(p)))
    return stems

def load_lib_titles(json_path: str):
    # 从 Milvus 导出的真实库内标题做精确去重（比 D:\file 文件名比对更准）
    import json
    try:
        with open(json_path, encoding="utf-8") as f:
            d = json.load(f)
        titles = set()
        for t in d.get("file_titles", []) + d.get("item_names", []):
            titles.add(norm_name(t))
        return titles
    except Exception as e:
        print(f"[warn] 读取库内标题失败: {e}", file=sys.stderr)
        return set()

def est_pages(path: str) -> int:
    try:
        with open(path, "rb") as f:
            data = f.read()
        # /Type /Page 出现次数（排除 /Pages 树节点）
        cnt = len(re.findall(rb"/Type\s*/?\s*Page(?!s)", data))
        return cnt
    except Exception as e:
        return -1

def categorize(name: str) -> str:
    n = name.lower()
    if any(k in n for k in ["aolynk", "h3c", "er2100", "mer系列", "msr", "ner", "nr-", "ug系列", "la2608"]):
        return "网络设备/路由器"
    if "matebook" in n:
        return "华为笔记本"
    if "matestation" in n or "matestation" in n:
        return "华为台式"
    if "pixlab" in n:
        return "华为打印机"
    if "平板" in n or "matepad" in n or "pad" in n:
        return "平板"
    if "显示器" in n or "monitor" in n or "b3-211h" in n:
        return "显示器"
    if "擎云" in n or "qingyun" in n or "b530" in n or "g540" in n:
        return "华为台式/一体机"
    if any(k in n for k in ["打印机", "lj", "pantum", "panda", "z26", "hl30",
                            "m3070", "m7208", "m7268", "m3系列", "t3", "m3470"]):
        return "打印机(联想/奔图/兄弟)"
    if any(k in n for k in ["万用表", "网关", "迅饶", "海豚", "hl30"]):
        return "其他"
    return "其他"

def main():
    if not DOC_DIR or not os.path.isdir(DOC_DIR):
        print("请先通过环境变量指定语料目录，例如：")
        print("  DOC_DIR=/path/to/doc FILE_DIR=/path/to/file python scripts/scan_doc.py")
        sys.exit(1)
    lib_titles = load_lib_titles(os.environ.get(
        "LIB_TITLES_JSON", str(ROOT / "logs" / "lib_titles.json")))
    rows = []
    for p in glob.glob(os.path.join(DOC_DIR, "**", "*.pdf"), recursive=True):
        base = os.path.basename(p)
        size_mb = os.path.getsize(p) / (1024 * 1024)
        pages = est_pages(p)
        cat = categorize(base)
        in_lib = norm_name(os.path.splitext(base)[0]) in lib_titles
        rows.append((base, cat, round(size_mb, 2), pages, in_lib))

    # 排序：品类 -> 体积降序
    cat_order = ["网络设备/路由器", "华为笔记本", "华为台式", "华为台式/一体机",
                 "平板", "显示器", "华为打印机", "打印机(联想/奔图/兄弟)", "其他"]
    rows.sort(key=lambda r: (cat_order.index(r[1]) if r[1] in cat_order else 99, -r[2]))

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["filename", "category", "size_mb", "est_pages", "in_library"])
        for r in rows:
            w.writerow(r)

    # 打印汇总
    new_rows = [r for r in rows if not r[4]]
    print(f"共扫描 {len(rows)} 个 PDF；已在库(跳过) {len(rows)-len(new_rows)} 个；待传(新) {len(new_rows)} 个\n")
    print(f"{'品类':<22}{'数量':>4}{'总页数(估)':>12}{'总体积MB':>12}")
    from collections import defaultdict
    agg = defaultdict(lambda: [0, 0, 0.0])
    for base, cat, sz, pg, in_lib in new_rows:
        agg[cat][0] += 1
        agg[cat][1] += max(pg, 0)
        agg[cat][2] += sz
    for cat in cat_order:
        if cat in agg:
            c, p, s = agg[cat]
            print(f"{cat:<22}{c:>4}{p:>12}{s:>12.1f}")
    other = [c for c in agg if c not in cat_order]
    for cat in other:
        c, p, s = agg[cat]
        print(f"{cat:<22}{c:>4}{p:>12}{s:>12.1f}")

    print("\n=== 全部明细（已在库标 *）===")
    for base, cat, sz, pg, in_lib in rows:
        mark = "*" if in_lib else " "
        print(f"{mark} {cat:<20} {sz:>7.2f}MB  ~{pg:>4}页  {base}")

if __name__ == "__main__":
    main()
