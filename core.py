# -*- coding: utf-8 -*-
"""هسته‌ی طرح‌یاب: پیش‌پردازش، ایندکس و رتبه‌بندی (بدون وابستگی به torch و streamlit)"""
import os, pickle
import cv2, numpy as np

SHEET_SIDE = 738                 # اندازه‌ی استاندارد ضلع کوتاه شیت
LIB_SIZES = (200, 300, 450)      # اندازه‌ی تکه‌ها روی شیت ۷۳۸ پیکسلی
EXTS = (".png", ".jpg", ".jpeg", ".webp")
MAX_QUERY_TILES = 2400           # سقف تعداد تکه‌های عکس مشتری (برای جلوگیری از کندی زیاد)

# حالت‌های جستجو: اندازه‌ی تکه‌ها نسبت به عکس برش‌خورده، و زاویه‌های اضافه
MODES = {
    "سریع":      dict(scales=(0.35, 0.6, 1.0),                              angles=(0,)),
    "دقیق":      dict(scales=(0.2, 0.3, 0.5, 0.8, 1.0),                     angles=(0,)),
    "خیلی دقیق": dict(scales=(0.15, 0.2, 0.3, 0.4, 0.5, 0.65, 0.8, 1.0),    angles=(-12, 0, 12)),
}


def list_sheets(lib_dir):
    if not os.path.isdir(lib_dir):
        return []
    return sorted(os.path.join(lib_dir, f) for f in os.listdir(lib_dir) if f.lower().endswith(EXTS))


def imread(path, flag):
    try:
        return cv2.imdecode(np.fromfile(path, np.uint8), flag)      # مسیرهای فارسی هم کار می‌کند
    except Exception:
        return None


# ---------- پیش‌پردازش شیت‌ها ----------
def crop_panel(g):
    """پیدا کردن صفحه‌ی خاکستری شیت. خروجی: تصویر برش‌خورده و جابه‌جایی (x, y) آن در تصویر اصلی"""
    m = ((g > 185) & (g < 242)).astype(np.float32)
    rows = np.where(m.mean(1) > 0.35)[0]; cols = np.where(m.mean(0) > 0.35)[0]
    if len(rows) < 100 or len(cols) < 100:
        return g, (0, 0)
    return g[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1], (int(cols[0]), int(rows[0]))


def clean_sheet(path):
    """شیت کتابخانه: حذف لوگو، عنوان کد، QR، حاشیه و سایه؛ خروجی سیاه روی سفید.
    علاوه بر تصویر، تبدیل مختصات (ox, oy, r) هم برمی‌گردد تا محل تطابق روی شیت اصلی درست رسم شود:
    مختصات_اصلی = مختصات_تمیز / r + (ox, oy)"""
    g = imread(path, cv2.IMREAD_GRAYSCALE)
    if g is None:
        return None, None
    g, (ox, oy) = crop_panel(g)
    H, W = g.shape; s = min(H, W)
    b = np.where(g < 100, 0, 255).astype(np.uint8)
    m = int(.05 * s)
    b[:m] = b[-m:] = 255; b[:, :m] = b[:, -m:] = 255             # لبه‌ها و گوشه‌های گرد
    b[:int(.22 * s), :int(.26 * s)] = 255                         # لوگو (گوشه‌ی بالا-چپ)
    b[:int(.13 * s), int(.26 * s):W - int(.24 * s)] = 255         # «کد»
    b[H - int(.22 * s):, W - int(.24 * s):] = 255                 # QR و متن (پایین-راست)
    r = 1.0
    if s > 0 and abs(s - SHEET_SIDE) / SHEET_SIDE > 0.10:         # هم‌اندازه کردن شیت‌های با اندازه‌ی متفاوت
        r = SHEET_SIDE / s
        b = cv2.resize(b, None, fx=r, fy=r, interpolation=cv2.INTER_AREA)
        b = np.where(b < 170, 0, 255).astype(np.uint8)
    return b, (ox, oy, r)


def tile_boxes(W, H, sizes, overlap=0.5):
    for s in sizes:
        s = int(min(s, W, H)); step = max(1, int(s * (1 - overlap)))
        for y in range(0, max(H - s, 0) + 1, step):
            for x in range(0, max(W - s, 0) + 1, step):
                yield (x, y, s, s)


# ---------- ایندکس (افزایشی: فقط طرح‌های جدید یا تغییرکرده دوباره خوانده می‌شوند) ----------
def _save(path, entries):
    tmp = path + ".tmp"
    with open(tmp, "wb") as fh:
        pickle.dump({"files": entries}, fh)
    os.replace(tmp, path)


def assemble(entries):
    names, V, boxes, starts, tfs = [], [], [], [], []
    pos = 0
    for n in sorted(entries):
        e = entries[n]
        if e["vecs"] is None or len(e["vecs"]) == 0:
            continue
        names.append(n); starts.append(pos); tfs.append(e["tf"])
        V.append(e["vecs"].astype("float32")); boxes += e["boxes"]; pos += len(e["vecs"])
    if not V:
        return None
    return {"names": names, "starts": np.array(starts), "V": np.vstack(V), "boxes": boxes, "tf": tfs}


def sync_index(cache_path, files, embed_fn, progress=None):
    entries = {}
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "rb") as fh:
                entries = pickle.load(fh).get("files", {})
        except Exception:
            entries = {}
    present = {os.path.basename(f): f for f in files}
    changed = False
    for n in list(entries):
        if n not in present:
            del entries[n]; changed = True
    todo = sorted(n for n, p in present.items()
                  if n not in entries or entries[n]["mtime"] != os.path.getmtime(p))
    for k, n in enumerate(todo):
        p = present[n]
        b, tf = clean_sheet(p)
        mt = os.path.getmtime(p)
        if b is None:
            entries[n] = {"mtime": mt, "vecs": None, "boxes": [], "tf": None}
        else:
            H, W = b.shape
            tiles, boxes = [], []
            for (x, y, w, h) in tile_boxes(W, H, LIB_SIZES):
                t = b[y:y + h, x:x + w]
                if (t < 128).mean() < 0.03:
                    continue                                    # تکه‌ی خالی
                tiles.append(t); boxes.append((x, y, w, h))
            if tiles:
                entries[n] = {"mtime": mt, "vecs": embed_fn(tiles).astype("float16"), "boxes": boxes, "tf": tf}
            else:
                entries[n] = {"mtime": mt, "vecs": None, "boxes": [], "tf": tf}
        changed = True
        if progress: progress(k + 1, len(todo))
        if (k + 1) % 50 == 0:
            _save(cache_path, entries)                          # ذخیره‌ی میانی تا با قطع شدن برق کارها از دست نرود
    if changed:
        _save(cache_path, entries)
    return assemble(entries)


# ---------- عکس مشتری ----------
def trim_bars(img):
    """حذف نوارهای سیاه بالا/پایین اسکرین‌شات‌ها"""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rows = np.where(g.mean(1) > 25)[0]; cols = np.where(g.mean(0) > 25)[0]
    if len(rows) < 50 or len(cols) < 50:
        return img
    return img[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]


def binarize_variants(img_bgr):
    """دو نسخه‌ی سیاه‌وسفید از یک تکه: روشنایی و اشباع رنگ"""
    outs = []
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    for i, ch in enumerate((cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY), hsv[:, :, 1])):
        c = cv2.createCLAHE(2.0, (8, 8)).apply(ch)
        c = cv2.GaussianBlur(c, (5, 5), 0)
        b = cv2.adaptiveThreshold(c, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 41, 10)
        if i == 1:
            b = 255 - b
        outs.append(b)
    return outs


def _rotate_small(t, ang):
    h, w = t.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
    return cv2.warpAffine(t, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=255)


def query_tiles(img_bgr, mode="دقیق", rotations=True):
    h, w = img_bgr.shape[:2]
    if min(h, w) > 720:
        r = 720 / min(h, w); img_bgr = cv2.resize(img_bgr, None, fx=r, fy=r, interpolation=cv2.INTER_AREA)
    elif min(h, w) < 240:
        r = 240 / min(h, w); img_bgr = cv2.resize(img_bgr, None, fx=r, fy=r, interpolation=cv2.INTER_CUBIC)
    h, w = img_bgr.shape[:2]; base = min(h, w)
    side = max(h, w)
    square = cv2.copyMakeBorder(img_bgr, 0, side - h, 0, side - w, cv2.BORDER_REPLICATE)  # برای تکه‌ی «کل طرح»
    cfg = MODES[mode]
    tiles, meta = [], []
    for frac in cfg["scales"]:
        if frac >= 1.0:
            src, boxes = square, [(0, 0, side)]
        else:
            src = img_bgr
            boxes = [(x, y, s) for (x, y, s, _) in tile_boxes(w, h, [int(base * frac)], overlap=0.4)]
        for (x, y, s) in boxes:
            if s < 32:
                continue
            for t in binarize_variants(src[y:y + s, x:x + s]):
                if (t < 128).mean() < 0.03:
                    continue
                for ang in cfg["angles"]:
                    ta = t if ang == 0 else _rotate_small(t, ang)
                    for k in (range(4) if rotations else [0]):
                        tiles.append(np.rot90(ta, k).copy()); meta.append((x, y, s, ang, k))
    if len(tiles) > MAX_QUERY_TILES:                              # نمونه‌برداری یکنواخت
        sel = np.linspace(0, len(tiles) - 1, MAX_QUERY_TILES).astype(int)
        tiles = [tiles[i] for i in sel]; meta = [meta[i] for i in sel]
    return tiles, meta


# ---------- رتبه‌بندی ----------
def rank(q, idx, top_k=12, chunk=24000):
    """امتیاز هر شیت = ترکیب سه چیز:
       ۱) بهترین تکه‌ی شیت نسبت به عکس مشتری، ۲) میانگین سه تکه‌ی برتر شیت،
       ۳) میانگین بهترین تکه‌های عکس مشتری که روی این شیت جا افتاده‌اند (تطابق از سمت عکس مشتری)"""
    V, starts = idx["V"], idx["starts"]
    n = len(starts); ends = np.append(starts[1:], len(V))
    scores = np.zeros(n, "float32"); best_col = np.zeros(n, int); best_q = np.zeros(n, int)
    kq = min(5, len(q))
    i = 0
    while i < n:
        j = i + 1
        while j < n and ends[j] - starts[i] < chunk:
            j += 1
        a, b = int(starts[i]), int(ends[j - 1])
        S = q @ V[a:b].T                                           # (تکه‌های مشتری) × (تکه‌های کتابخانه)
        lib_best = S.max(0); q_arg = S.argmax(0)
        loc = starts[i:j] - a
        R = np.maximum.reduceat(S, loc, axis=1)                    # بهترین تطابق هر تکه‌ی مشتری در هر شیت
        qside = np.partition(R, -kq, axis=0)[-kq:].mean(0)
        for m in range(j - i):
            s0 = int(loc[m]); s1 = s0 + int(ends[i + m] - starts[i + m])
            lb = lib_best[s0:s1]
            t3 = np.sort(lb)[-3:].mean()
            c = s0 + int(lb.argmax())
            scores[i + m] = 0.5 * lb.max() + 0.2 * t3 + 0.3 * qside[m]
            best_col[i + m] = a + c; best_q[i + m] = q_arg[c]
        i = j
    out = []
    for si in np.argsort(-scores)[:top_k]:
        x, y, w, h = idx["boxes"][best_col[si]]
        ox, oy, r = idx["tf"][si]
        out.append(dict(name=idx["names"][si], score=float(scores[si]), dino=float(scores[si]), line=None,
                        box=(int(x / r + ox), int(y / r + oy), int(w / r), int(h / r)),
                        tf=(ox, oy, r), qtile=int(best_q[si])))
    return out


# ---------- بازبینی دقیق خطوط (مرحله‌ی دوم) ----------
# فاصله‌ی «چمفر» بین خطوط عکس مشتری و خطوط شیت: به ضخامت خط، سایه و رنگ حساس نیست
# و فقط وقتی کم می‌شود که خطوط واقعاً روی هم بیفتند. در هر دو جهت سنجیده می‌شود.
CH_SCALE = 0.3                                   # شیت برای این مرحله کوچک می‌شود تا سریع باشد
CH_FRACS = (0.18, 0.26, 0.36, 0.48, 0.62, 0.8)   # اندازه‌ی احتمالی طرح نسبت به شیت
CH_TAU = 10.0                                    # سقف فاصله (پیکسل در مقیاس کوچک‌شده)
CH_WORK = 360


def _query_ink_masks(img_bgr):
    h, w = img_bgr.shape[:2]
    r = CH_WORK / max(h, w)
    img = cv2.resize(img_bgr, None, fx=r, fy=r, interpolation=cv2.INTER_AREA if r < 1 else cv2.INTER_CUBIC)
    masks = []
    for t in binarize_variants(img):
        ink = (t < 128).astype(np.uint8)
        n, lab, st, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
        keep = np.zeros(n, np.uint8); keep[1:] = st[1:, cv2.CC_STAT_AREA] >= 12      # حذف لکه‌های ریز
        ink = keep[lab]
        if 0.01 < ink.mean() < 0.6:                                                  # کاملاً خالی یا کاملاً سیاه بی‌فایده است
            masks.append(ink.astype(np.float32))
    return masks


def _templates(masks, side, rotations):
    out = []
    for m in masks:
        for frac in CH_FRACS:
            L = frac * side; hh, ww = m.shape
            f = L / max(hh, ww)
            th, tw = max(8, int(hh * f)), max(8, int(ww * f))
            T = (cv2.resize(m, (tw, th), interpolation=cv2.INTER_AREA) > 0.3).astype(np.float32)
            for k in (range(4) if rotations else [0]):
                out.append(np.rot90(T, k).copy())
    return out


def _chamfer_best(ink, dts, templates):
    """بهترین محل و اندازه‌ی طرح روی شیت با فاصله‌ی چمفر دوطرفه (میانگین فاصله‌ی خطوط، با سقف CH_TAU):
       خطوط مشتری تا نزدیک‌ترین خط شیت، و خطوط آن ناحیه‌ی شیت تا نزدیک‌ترین خط مشتری"""
    H, W = ink.shape; best = (CH_TAU, None)
    for T in templates:
        th, tw = T.shape
        if th >= H or tw >= W:
            continue
        n = float(T.sum())
        if n < 30:
            continue
        dtq = np.minimum(cv2.distanceTransform((T < 0.5).astype(np.uint8), cv2.DIST_L2, 3), CH_TAU)
        A = cv2.matchTemplate(dts, T, cv2.TM_CCORR) / n
        cnt = cv2.matchTemplate(ink, np.ones_like(T), cv2.TM_CCORR)
        B = cv2.matchTemplate(ink, dtq, cv2.TM_CCORR) / np.maximum(cnt, 1)
        B = np.where(cnt > 0.25 * n, B, CH_TAU)
        D = (A + np.minimum(B, CH_TAU)) / 2
        y, x = np.unravel_index(D.argmin(), D.shape)
        if D[y, x] < best[0]:
            best = (float(D[y, x]), (int(x), int(y), tw, th))
    return best


def rerank(img_bgr, cands, lib_dir, rotations=True, weight=0.4, progress=None):
    """cands خروجی rank است. امتیاز ظاهری (dino) و تطابق خطوط (line) را با وزن weight ترکیب می‌کند"""
    masks = _query_ink_masks(img_bgr)
    if not masks or not cands:
        return cands
    for i, c in enumerate(cands):
        b, tf = clean_sheet(os.path.join(lib_dir, c["name"]))
        if b is None:
            c["line"] = 0.0
        else:
            bs = cv2.resize(b, None, fx=CH_SCALE, fy=CH_SCALE, interpolation=cv2.INTER_AREA)
            ink = (bs < 200).astype(np.float32)
            dts = np.minimum(cv2.distanceTransform((ink < 0.5).astype(np.uint8), cv2.DIST_L2, 3), CH_TAU)
            d, box = _chamfer_best(ink, dts, _templates(masks, min(ink.shape), rotations))
            c["line"] = float(1 - d / CH_TAU)
            if box is not None:
                x, y, w, h = [v / CH_SCALE for v in box]; ox, oy, r = tf
                c["box"] = (int(x / r + ox), int(y / r + oy), int(w / r), int(h / r))
        if progress: progress(i + 1, len(cands))
    z = lambda a: (np.array(a) - np.mean(a)) / (np.std(a) + 1e-6)
    f = (1 - weight) * z([c["dino"] for c in cands]) + weight * z([c["line"] for c in cands])
    for c, s in zip(cands, f):
        c["score"] = float(s)
    return sorted(cands, key=lambda c: -c["score"])


def search(img_bgr, idx, embed_fn, mode="دقیق", rotations=True, top_k=12, progress=None,
           lib_dir=None, line_weight=0.4, n_cand=30, progress2=None):
    tiles, meta = query_tiles(img_bgr, mode, rotations)
    if not tiles:
        return [], 0
    q = embed_fn(tiles, progress)
    if lib_dir and line_weight > 0:
        cands = rank(q, idx, max(top_k, n_cand))
        try:
            cands = rerank(img_bgr, cands, lib_dir, rotations, line_weight, progress2)
        except Exception:
            pass                                                   # اگر مرحله‌ی دوم خطا داد، همان نتیجه‌ی مرحله‌ی اول
        return cands[:top_k], len(tiles)
    return rank(q, idx, top_k), len(tiles)
