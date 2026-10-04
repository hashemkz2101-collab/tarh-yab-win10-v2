# -*- coding: utf-8 -*-
"""سنجش دقت روی نمونه‌های خودتان.
هر فایل کلاژ باید دو تصویر کنار هم داشته باشد: سمت چپ عکس مشتری، سمت راست شیتی از کتابخانه.

اجرا (داخل پوشه‌ی برنامه، با venv فعال):
    python evaluate.py پوشه_کلاژها [small|base]

برنامه شیت سمت راست را در کتابخانه پیدا می‌کند (پاسخ درست)، بعد عکس سمت چپ را با چند تنظیم مختلف
جستجو می‌کند و می‌گوید پاسخ درست در چه رتبه‌ای آمده است. با این جدول می‌توانید بهترین تنظیم را انتخاب کنید."""
import os, sys, glob
import cv2, numpy as np
import core

CONFIGS = [   # (دقت, وزن تطبیق خطوط)  — وزن صفر یعنی بدون بازبینی خطوط
    ("سریع", 0.0), ("دقیق", 0.0), ("خیلی دقیق", 0.0),
    ("دقیق", 0.2), ("دقیق", 0.4),
]


def split_collage(im):
    H, W = im.shape[:2]
    g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    white = (g > 245).mean(0)
    mid = np.arange(int(W * .40), int(W * .60))
    gap = mid[white[mid] > 0.97]
    cut = int(np.median(gap)) if len(gap) else W // 2

    def trim(a):
        gg = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
        rows = np.where((gg < 245).mean(1) > 0.05)[0]; cols = np.where((gg < 245).mean(0) > 0.05)[0]
        return a[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]
    return trim(im[:, :cut]), trim(im[:, cut:])


def run(folder, embed, lib_dir="library", cache="index_eval.pkl", configs=CONFIGS, out=print):
    idx = core.sync_index(cache, core.list_sheets(lib_dir), embed,
                          lambda i, n: out(f"  ایندکس: {i}/{n}") if i % 20 == 0 else None)
    files = sorted(f for f in glob.glob(os.path.join(folder, "*")) if f.lower().endswith(core.EXTS))
    ranks = {c: [] for c in configs}
    for f in files:
        im = core.imread(f, cv2.IMREAD_COLOR)
        if im is None:
            continue
        query, sheet = split_collage(im)
        truth, _ = core.search(sheet, idx, embed, "سریع", True, 1)
        if not truth:
            continue
        t = truth[0]["name"]
        out(f"\n{os.path.basename(f)}  ←  شیت درست: {t} (امتیاز {truth[0]['dino']:.2f})")
        for (mode, w) in configs:
            res, _ = core.search(query, idx, embed, mode, True, 50, lib_dir=lib_dir if w > 0 else None, line_weight=w, n_cand=50)
            names = [r["name"] for r in res]
            rk = names.index(t) + 1 if t in names else None
            ranks[(mode, w)].append(rk)
            out(f"  {mode:10s} خطوط {w:.1f}  →  رتبه: {rk if rk else 'بیرون از ۵۰'}")
    out("\n===== خلاصه =====")
    for c, r in ranks.items():
        n = len(r)
        if not n: continue
        top = lambda k: sum(1 for x in r if x and x <= k)
        out(f"{c[0]:10s} خطوط {c[1]:.1f}:  رتبه ۱: {top(1)}/{n}   ۵ تای اول: {top(5)}/{n}   ۱۰ تای اول: {top(10)}/{n}")
    return ranks


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(1)
    import embed as E
    run(sys.argv[1], E.make_embedder(sys.argv[2] if len(sys.argv) > 2 else "small"))
