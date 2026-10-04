# -*- coding: utf-8 -*-
"""طرح‌یاب: پیدا کردن شیت‌های مشابه در کتابخانه‌ی طرح‌های سیاه‌وسفید با عکس مشتری
اجرا:  streamlit run app.py"""
import os, time
import cv2, numpy as np, streamlit as st
from PIL import Image, ImageOps
import core

LIB_DIR = "library"
MODELS = {
    "small": dict(label="سریع‌تر", local="model_dinov2_small", hub="facebook/dinov2-small"),
    "base":  dict(label="دقیق‌تر (کندتر)", local="model_dinov2_base", hub="facebook/dinov2-base"),
}

st.set_page_config(page_title="طرح‌یاب", page_icon="🔍", layout="wide")
st.markdown("""
<style>
html, body, .stApp, [data-testid="stSidebar"] { direction: rtl; font-family: Vazirmatn, "Segoe UI", Tahoma, sans-serif; }
h1, h2, h3, p, label, .stMarkdown, .stCaption { text-align: right; }
[data-testid="stImage"] img { border: 1px solid #d5d9e2; border-radius: 4px; }
.small-note { color: #5b6270; font-size: .9rem; }
</style>
""", unsafe_allow_html=True)


# ---------- مدل ----------
@st.cache_resource(show_spinner="در حال بارگذاری مدل…")
def load_model(key):
    import torch
    from transformers import AutoImageProcessor, AutoModel
    m = MODELS[key]; src = m["local"] if os.path.isdir(m["local"]) else m["hub"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return AutoImageProcessor.from_pretrained(src), AutoModel.from_pretrained(src).to(device).eval(), device


def make_embedder(key):
    import torch
    proc, model, device = load_model(key)

    def embed(tiles, progress=None):
        out, n = [], len(tiles)
        with torch.inference_mode():
            for i in range(0, n, 32):
                batch = [Image.fromarray(t).convert("RGB") for t in tiles[i:i + 32]]
                # بدون برش گوشه‌ها: کل تکه به ۲۲۴×۲۲۴ تبدیل می‌شود
                inp = proc(images=batch, size={"shortest_edge": 224}, do_center_crop=False,
                           return_tensors="pt").to(device)
                h = model(**inp).last_hidden_state
                v = torch.cat([h[:, 0], h[:, 1:].mean(1)], dim=1)          # CLS + میانگین پچ‌ها
                out.append(torch.nn.functional.normalize(v, dim=-1).cpu().numpy())
                if progress: progress(min(i + 32, n), n)
        return np.vstack(out).astype("float32")
    return embed


@st.cache_resource(show_spinner=False)
def get_index(key, token):
    """ایندکس را می‌خواند و فقط طرح‌های جدید یا تغییرکرده را به آن اضافه می‌کند"""
    if not core.list_sheets(LIB_DIR):
        return None
    embed = make_embedder(key)
    bar = st.progress(0.0, text="در حال بررسی کتابخانه…")

    def cb(i, n):
        bar.progress(i / n, text=f"در حال خواندن طرح‌های جدید… {i} از {n}")
    idx = core.sync_index(f"index_{key}_v2.pkl", core.list_sheets(LIB_DIR), embed, cb)
    bar.empty()
    return idx


@st.cache_data(show_spinner=False, max_entries=64)
def load_sheet(path, mtime):
    return cv2.cvtColor(core.imread(path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


# ---------- نوار کناری ----------
with st.sidebar:
    st.header("تنظیمات جستجو")
    mode = st.radio("دقت جستجو", ["سریع", "دقیق", "خیلی دقیق"], index=1,
                    help="«خیلی دقیق» زاویه‌های کج و تکه‌های بیشتری را هم بررسی می‌کند و چند برابر کندتر است.")
    rot = st.checkbox("بررسی چرخش‌های ۹۰ درجه", True)
    top_k = st.slider("تعداد نتیجه", 4, 24, 12, step=4)
    zoom = st.checkbox("نمایش بزرگ‌شده‌ی بخش مطابق", True)
    line_check = st.checkbox("بازبینی خطوط (آزمایشی)", False,
                             help="۳۰ طرح برتر را یک بار دیگر با تطبیق خط‌به‌خط می‌سنجد. روی نمونه‌های سوزن‌دوزی "
                                  "که با آن‌ها امتحان شد کمکی نکرد؛ فقط اگر با evaluate.py بهتر شدن را دیدید روشن کنید.")
    line_w = st.slider("اهمیت تطبیق خطوط (٪)", 10, 90, 40, step=10, disabled=not line_check,
                       help="اگر عکس‌ها نویزی‌اند (پارچه‌ی رنگی، سایه، بازتاب) کمترش کنید؛ "
                            "اگر عکس‌ها تمیز و دور طرح بریده‌شده‌اند بیشترش کنید.") / 100

    model_keys = [k for k in MODELS if k == "small" or os.path.isdir(MODELS[k]["local"])]
    mkey = "small"
    if len(model_keys) > 1:
        mkey = st.radio("مدل", model_keys, format_func=lambda k: MODELS[k]["label"],
                        help="با تغییر مدل، ایندکس یک بار از نو ساخته می‌شود.")

    st.divider()
    st.subheader("کتابخانه")
    token = st.session_state.setdefault("token", 0)
    c1, c2 = st.columns(2)
    if c1.button("به‌روزرسانی", help="طرح‌های تازه‌اضافه‌شده را به ایندکس اضافه می‌کند"):
        get_index.clear(); st.session_state["token"] += 1; st.rerun()
    if c2.button("ساخت از نو", help="ایندکس را پاک می‌کند و همه‌ی طرح‌ها را دوباره می‌خواند"):
        p = f"index_{mkey}_v2.pkl"
        if os.path.exists(p): os.remove(p)
        get_index.clear(); st.session_state["token"] += 1; st.rerun()
    if hasattr(os, "startfile") and st.button("باز کردن پوشه‌ی طرح‌ها"):
        os.makedirs(LIB_DIR, exist_ok=True); os.startfile(os.path.abspath(LIB_DIR))

    with st.expander("افزودن طرح جدید"):
        adds = st.file_uploader("فایل شیت‌ها", type=["png", "jpg", "jpeg", "webp"], accept_multiple_files=True,
                                key=f"add_{st.session_state.get('add_n', 0)}")
        if adds and st.button("ذخیره در کتابخانه"):
            os.makedirs(LIB_DIR, exist_ok=True)
            for f in adds:
                with open(os.path.join(LIB_DIR, os.path.basename(f.name)), "wb") as fh:
                    fh.write(f.getbuffer())
            st.session_state["add_n"] = st.session_state.get("add_n", 0) + 1
            get_index.clear(); st.session_state["token"] += 1; st.rerun()


# ---------- صفحه‌ی اصلی ----------
st.title("طرح‌یاب")
st.markdown("<p class='small-note'>عکس مشتری را بیندازید، دور طرح را ببُرید و دکمه‌ی جستجو را بزنید.</p>",
            unsafe_allow_html=True)

if not core.list_sheets(LIB_DIR):
    st.warning("کتابخانه خالی است. فایل شیت‌ها (PNG یا JPG) را در پوشه‌ی «library» کنار برنامه کپی کنید "
               "یا از «افزودن طرح جدید» در نوار کناری استفاده کنید.")
    st.stop()

idx = get_index(mkey, st.session_state["token"])
if idx is None:
    st.error("هیچ طرح قابل‌خواندنی در کتابخانه پیدا نشد.")
    st.stop()
st.sidebar.caption(f"{len(idx['names'])} طرح در ایندکس")

st.subheader("۱. عکس مشتری")
up = st.file_uploader("عکس را بکشید و رها کنید", type=["jpg", "jpeg", "png", "webp"], label_visibility="collapsed")
if not up:
    st.info("هنوز عکسی انتخاب نشده است.")
    st.stop()

ukey = f"{up.name}:{up.size}"
if st.session_state.get("ukey") != ukey:
    st.session_state["ukey"] = ukey; st.session_state.pop("results", None)

pil = ImageOps.exif_transpose(Image.open(up)).convert("RGB")            # عکس‌های گوشی که چرخیده‌اند درست می‌شوند
bgr = core.trim_bars(cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR))
pil = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def pick_region(img):
    disp = img.copy(); disp.thumbnail((1600, 1600))
    try:
        from streamlit_cropper import st_cropper
        return st_cropper(disp, realtime_update=True, box_color="#1f3a8a", aspect_ratio=None, key=f"crop_{ukey}")
    except Exception:
        pass
    st.caption("برش با موس در دسترس نیست؛ با لغزنده‌ها محدوده را مشخص کنید.")
    a, b = st.columns(2)
    left = a.slider("حذف از چپ (٪)", 0, 90, 0); right = b.slider("حذف از راست (٪)", 0, 90, 0)
    top = a.slider("حذف از بالا (٪)", 0, 90, 0); bottom = b.slider("حذف از پایین (٪)", 0, 90, 0)
    W, H = disp.size
    x0, x1 = int(W * left / 100), int(W * (1 - right / 100)); y0, y1 = int(H * top / 100), int(H * (1 - bottom / 100))
    if x1 - x0 < 40 or y1 - y0 < 40:
        return disp
    prev = np.array(disp).copy(); cv2.rectangle(prev, (x0, y0), (x1, y1), (31, 58, 138), max(3, W // 200))
    st.image(prev)
    return disp.crop((x0, y0, x1, y1))


ca, cb = st.columns([3, 2])
with ca:
    crop = pick_region(pil)
with cb:
    st.image(crop, caption="بخشی که جستجو می‌شود")
    st.markdown("<p class='small-note'>هر چه برش به خود طرح نزدیک‌تر باشد، نتیجه دقیق‌تر است. "
                "پس‌زمینه، پوست و نوشته‌ها را تا جای ممکن کنار بگذارید.</p>", unsafe_allow_html=True)
    go = st.button("جستجو", type="primary")

if go:
    embed = make_embedder(mkey)
    bar = st.progress(0.0, text="در حال بررسی…")
    t0 = time.time()
    q_bgr = cv2.cvtColor(np.array(crop.convert("RGB")), cv2.COLOR_RGB2BGR)
    bar2 = st.empty()

    def prog2(i, m):
        bar2.progress(i / m, text=f"بازبینی دقیق خطوط… {i} از {m} طرح")
    res, n = core.search(q_bgr, idx, embed, mode, rot, top_k,
                         progress=lambda i, m: bar.progress(i / m, text=f"در حال بررسی… {i} از {m} تکه"),
                         lib_dir=LIB_DIR if line_check else None, line_weight=line_w, progress2=prog2)
    bar.empty(); bar2.empty()
    st.session_state["results"] = dict(res=res, n=n, secs=time.time() - t0)

R = st.session_state.get("results")
if R:
    st.subheader("۲. شبیه‌ترین طرح‌ها")
    st.caption(f"{R['n']} تکه از عکس بررسی شد ({R['secs']:.0f} ثانیه). شماره‌ی ۱ شبیه‌ترین است.")
    cols = st.columns(4)
    for i, r in enumerate(R["res"]):
        path = os.path.join(LIB_DIR, r["name"])
        if not os.path.exists(path):
            continue
        sheet = load_sheet(path, os.path.getmtime(path)).copy()
        x, y, w, h = r["box"]
        region = sheet[max(y, 0):y + h, max(x, 0):x + w].copy()
        cv2.rectangle(sheet, (x, y), (x + w, y + h), (220, 30, 30), max(3, min(sheet.shape[:2]) // 150))
        with cols[i % 4]:
            st.markdown(f"**{i + 1}. {r['name']}**")
            st.image(sheet)
            if zoom and region.size:
                st.image(region, caption="بخش مطابق")
            if r.get("line") is not None:
                st.caption(f"شباهت ظاهری {r['dino']:.2f} | تطابق خطوط {r['line']:.2f}")
            else:
                st.progress(float(min(max(r["dino"], 0), 1)), text=f"امتیاز {r['dino']:.2f}")
