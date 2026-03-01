"""
Rekentekening — Streamlit app
Genereert een PDF met rekenoefeningen gekoppeld aan een kleurplaat.

Benodigde mapstructuur:
    rekentekening_app.py  /  requirements.txt  /  afbeeldingen/1.png … 117.png
"""

import io
import os
import random
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Iterator

import fitz  # PyMuPDF
import numpy as np
import qrcode
import streamlit as st
from PIL import Image

# ── CONSTANTEN ────────────────────────────────────────────────────────────────

IMAGES_DIR       = os.path.join(os.path.dirname(os.path.abspath(__file__)), "afbeeldingen")
MAX_IMAGE_SIZE   = 15
NUM_EXERCISES    = 40
MAX_IMAGE_NUMBER = 117
QR_URL           = "https://rekentekenen.streamlit.app/"
FONT_CANDIDATES  = [
    "/Library/Fonts/Comic Sans MS.ttf",
    "/System/Library/Fonts/Supplemental/Comic Sans MS.ttf",
    "/Library/Fonts/Chalkboard.ttc",
    "C:/Windows/Fonts/comic.ttf",
]
EXISTING_IMAGES: list[int] = [
    i for i in range(1, MAX_IMAGE_NUMBER + 1)
    if os.path.exists(os.path.join(IMAGES_DIR, f"{i}.png"))
]

# ── DATACLASS: PAGINA CONFIGURATIE ───────────────────────────────────────────

@dataclass
class PageConfig:
    """Alle instellingen voor het tekenen van één PDF-pagina."""
    page_w:         int = 595
    page_h:         int = 842
    margin:         int = 50
    ex_fontsize:    int = 11
    row_height:     int = 24
    gap:            int = 15
    instr_fontsize: int = 13
    line_h:         int = 19
    qr_size:        int = 70
    measure_font:   str = "helv"
    measure_font_b: str = "hebo"
    title_fontsize: int = field(init=False)

    def __post_init__(self) -> None:
        self.title_fontsize = self.instr_fontsize + 5

# ── STREAMLIT PAGINA CONFIG ───────────────────────────────────────────────────

st.set_page_config(page_title="Rekentekening", page_icon="🎨", layout="centered")
st.markdown("""<style>
    @import url('https://fonts.googleapis.com/css2?family=Fredoka+One&family=Nunito:wght@400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Nunito', sans-serif; }
    h1, h2, h3 { font-family: 'Fredoka One', cursive !important; }
    .stButton > button {
        font-family: 'Fredoka One', cursive; font-size: 1.2rem;
        background: linear-gradient(135deg, #f9a825, #e53935);
        color: white; border: none; border-radius: 16px;
        padding: 0.6rem 2rem; width: 100%; transition: transform 0.1s;
    }
    .stButton > button:hover { transform: scale(1.03); color: white; }
    .stDownloadButton > button {
        font-family: 'Fredoka One', cursive; font-size: 1.1rem;
        background: linear-gradient(135deg, #43a047, #00897b);
        color: white; border: none; border-radius: 16px;
        padding: 0.6rem 2rem; width: 100%;
    }
    .stDownloadButton > button:hover { color: white; }
    .title-block { text-align: center; padding: 1.5rem 0 0.5rem 0; }
    .title-block h1 {
        font-size: 3rem;
        background: linear-gradient(135deg, #f9a825, #e53935);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 0;
    }
    .subtitle { font-size: 1.1rem; color: #666; text-align: center; margin-top: 0; }
</style>""", unsafe_allow_html=True)

# ── AFBEELDING VERWERKEN ──────────────────────────────────────────────────────

@st.cache_data
def process_image(image_data: bytes | str, max_size: int = MAX_IMAGE_SIZE) -> tuple[np.ndarray, dict]:
    """Laad afbeelding (pad of bytes), reduceer naar max_size en extraheer kleurclusters."""
    source = io.BytesIO(image_data) if isinstance(image_data, bytes) else image_data
    image  = Image.open(source)
    image  = image.resize((min(image.width, max_size), min(image.height, max_size)))

    if image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info):
        bg = Image.new("RGB", image.size, (255, 255, 255))
        rgba = image.convert("RGBA")
        bg.paste(rgba, mask=rgba.split()[3])
        image = bg
    else:
        image = image.convert("RGB")

    image   = image.convert("P", palette=Image.ADAPTIVE, colors=8)
    matrix  = np.array(image)
    palette = image.getpalette()
    unique_vals = sorted(set(matrix.flatten()))
    remap   = {v: i + 1 for i, v in enumerate(unique_vals)}
    matrix  = np.vectorize(remap.get)(matrix)
    cluster_colors = {
        cn: tuple(palette[oi * 3 + k] / 255.0 for k in range(3))
        for oi, cn in remap.items()
    }
    return matrix, cluster_colors

# ── WILLEKEURIGE AFBEELDINGEN ZONDER HERHALINGEN ─────────────────────────────

def build_image_queue(num_pages: int, img_choice: int) -> tuple[list[int], list[str]]:
    """Shuffled wachtrij zonder directe herhalingen. Geeft (queue, warnings) terug."""
    if not EXISTING_IMAGES:
        return [], ["Geen afbeeldingen gevonden in de map 'afbeeldingen'."]

    warnings: list[str] = []
    if img_choice > 0:
        if img_choice not in EXISTING_IMAGES:
            warnings.append(f"Afbeelding {img_choice}.png bestaat niet — een willekeurige afbeelding wordt gebruikt.")
            first = random.choice(EXISTING_IMAGES)
        else:
            first = img_choice
        rest = [i for i in EXISTING_IMAGES if i != first]
    else:
        rest = EXISTING_IMAGES[:]
        random.shuffle(rest)
        first = rest.pop(0)

    queue = [first]
    while len(queue) < num_pages:
        if not rest:
            rest = [i for i in EXISTING_IMAGES if i != queue[-1]]
            random.shuffle(rest)
        queue.append(rest.pop(0))

    return queue[:num_pages], warnings

# ── CYCLING GENERATOR ─────────────────────────────────────────────────────────

def _cycling(answers: list) -> Iterator:
    """Herhaalt de lijst in willekeurige volgorde, telkens opnieuw geschud."""
    while True:
        shuffled = answers[:]
        random.shuffle(shuffled)
        yield from shuffled

# ── OEFENINGEN GENEREREN ──────────────────────────────────────────────────────

def generate_math_exercises(
    mult_numbers: list[int], div_numbers: list[int],
    num_clusters: int = 8, num_exercises: int = NUM_EXERCISES,
) -> tuple[list[tuple], str | None]:
    """
    Genereer unieke rekenoefeningen. Geeft (exercises, warning | None) terug.
    Raises ValueError als er te weinig unieke antwoorden zijn.
    """
    available_ops = (["multiplication"] if mult_numbers else []) + (["division"] if div_numbers else [])
    if not available_ops:
        return [], None

    seen: set[str] = set()
    exercises: list[tuple] = []
    answer_to_group: dict[int, int] = {}
    group_counter = 1

    for _ in range(num_exercises * 50):
        if len(exercises) >= num_exercises:
            break
        op = random.choice(available_ops)
        if op == "multiplication":
            n1, n2 = random.choice(mult_numbers), random.randint(1, 10)
            answer, expr = n1 * n2, f"{n1} x {n2}"
        else:
            n2 = random.choice(div_numbers)
            n1 = random.randint(1, n2 * 10)
            while n1 % n2 != 0:
                n1 = random.randint(1, n2 * 10)
            answer, expr = n1 // n2, f"{n1} : {n2}"

        if expr in seen:
            continue
        seen.add(expr)
        if answer not in answer_to_group:
            answer_to_group[answer] = group_counter
            group_counter += 1
        exercises.append((answer_to_group[answer], answer, f"{expr} = ?"))

    if len(set(ans for _, ans, _ in exercises)) < 2:
        raise ValueError(
            "De gekozen tafels leveren te weinig unieke antwoorden op. "
            "Voeg meer tafels toe zodat de kleurplaat zinvol is."
        )

    warning = (
        f"Slechts {len(exercises)} unieke oefeningen gevonden. Selecteer meer tafels voor een voller werkblad."
        if len(exercises) < num_exercises else None
    )

    unique_groups = sorted(set(g for g, _, _ in exercises))
    cluster_map = {g: (i % num_clusters) + 1 for i, g in enumerate(unique_groups)}
    exercises = [(cluster_map[g], ans, ex) for g, ans, ex in exercises]
    random.shuffle(exercises)
    return exercises, warning

# ── MATRIX INVULLEN MET ANTWOORDEN ────────────────────────────────────────────

def populate_matrix(matrix: np.ndarray, exercises: list[tuple], num_clusters: int = 8) -> np.ndarray:
    """Vul elke cel van de matrix met een antwoord van het bijhorende cluster."""
    cluster_answers: dict[int, list] = {i: [] for i in range(1, num_clusters + 1)}
    for cluster, answer, _ in exercises:
        if answer not in cluster_answers[cluster]:
            cluster_answers[cluster].append(answer)
    cycles = {c: _cycling(ans) for c, ans in cluster_answers.items() if ans}
    result = np.full(matrix.shape, "?", dtype=object)
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            cluster = int(matrix[row][col])
            if cluster in cycles:
                result[row][col] = next(cycles[cluster])
    return result

# ── KLEURNAAM BEPALEN ─────────────────────────────────────────────────────────

# Geordende regels: (test_functie, naam) — eerste match wint
def _color_rules(r: float, g: float, b: float, max_c: float, min_c: float, sat: float) -> str:
    if max_c > 220 and sat < 0.06:                                        return "Wit"
    if max_c < 60:                                                         return "Zwart"
    if r > g and r > b and (r - b) > 10 and min_c > 100:                 return "Lichtroze"
    if sat < 0.15:                                                         return "Grijs"
    if r > 100 and g > 50 and b < 80 and r > g and max_c < 200 and sat > 0.2: return "Bruin"
    if max_c == r:
        if g > 200 and b < 80:   return "Geel"
        if g > r * 0.75:         return "Geel"
        if g > 80:               return "Oranje"
        if b > r * 0.7:          return "Paars"
        if b > 120:              return "Roze"
        return "Rood"
    if max_c == g:
        if r > 150:              return "Geel"
        if b > g * 0.85:         return "Lichtblauw"
        if r > g * 0.6:          return "Lichtgroen"
        return "Donkergroen"
    return "Paars" if r > 150 else ("Lichtblauw" if g > 150 else "Blauw")

@lru_cache(maxsize=256)
def get_color_name(r: float, g: float, b: float) -> str:
    """Geeft een Nederlandse kleurnaam terug op basis van RGB-waarden (0–1 bereik)."""
    r255, g255, b255 = r * 255, g * 255, b * 255
    max_c = max(r255, g255, b255)
    min_c = min(r255, g255, b255)
    sat   = (max_c - min_c) / max_c if max_c > 0 else 0
    return _color_rules(r255, g255, b255, max_c, min_c, sat)

@lru_cache(maxsize=1024)
def _text_len(text: str, fontname: str, fontsize: float) -> float:
    return fitz.get_text_length(text, fontname=fontname, fontsize=fontsize)

# ── QR CODE & FONT (eenmalig bij opstarten) ───────────────────────────────────

def _build_qr_png(url: str, size: int = 70) -> bytes:
    qr = qrcode.QRCode(version=1, error_correction=qrcode.constants.ERROR_CORRECT_M, box_size=4, border=2)
    qr.add_data(url)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white").convert("RGB")
    buf = io.BytesIO()
    img.resize((size, size), Image.NEAREST).save(buf, format="PNG")
    return buf.getvalue()

QR_PNG = _build_qr_png(QR_URL)
_PLAYFUL_FONT_PATH: str | None = next((p for p in FONT_CANDIDATES if os.path.exists(p)), None)

def _load_playful_font(page) -> tuple[str, str]:
    if _PLAYFUL_FONT_PATH:
        try:
            page.insert_font(fontname="playfont", fontfile=_PLAYFUL_FONT_PATH)
            return "playfont", "playfont"
        except Exception:
            pass
    return "helv", "hebo"

# ── OEFENING TEKENEN ─────────────────────────────────────────────────────────

def draw_exercise(
    page, ex: tuple, x_start: float, ex_y: float, col_w: float,
    cluster_colors: dict, color_col_w: float,
    font_r: str, font_br: str, cfg: PageConfig, show_answers: bool,
) -> None:
    """Teken één rekenoefening: kleurvak + naam + som + antwoord of stippellijn."""
    cluster, answer, text = ex
    cr, cg, cb = cluster_colors.get(cluster, (0, 0, 0))
    color_name = get_color_name(cr, cg, cb)
    name_w     = _text_len(color_name, fontname=cfg.measure_font_b, fontsize=cfg.ex_fontsize)

    sh = page.new_shape()
    sh.draw_rect(fitz.Rect(x_start, ex_y + 2, x_start + color_col_w, ex_y + cfg.row_height - 2))
    sh.finish(color=None, fill=(cr, cg, cb), width=0)
    sh.commit()

    lum = 0.299 * cr + 0.587 * cg + 0.114 * cb
    page.insert_text((x_start + (color_col_w - name_w) / 2, ex_y + 16),
                     color_name, fontname=font_br, fontsize=cfg.ex_fontsize,
                     color=(1, 1, 1) if lum < 0.5 else (0, 0, 0))

    ex_x    = x_start + color_col_w + 6
    label   = text.replace("= ?", "=")
    label_w = _text_len(label, fontname=cfg.measure_font, fontsize=cfg.ex_fontsize)
    page.insert_text((ex_x, ex_y + 16), label, fontname=font_r, fontsize=cfg.ex_fontsize, color=(0, 0, 0))

    if show_answers:
        page.insert_text((ex_x + label_w + 4, ex_y + 16), str(answer),
                         fontname=font_br, fontsize=cfg.ex_fontsize, color=(0, 0, 0))
    else:
        page.draw_line((ex_x + label_w + 4, ex_y + 15), (x_start + col_w - 8, ex_y + 15),
                       color=(0, 0, 0), width=0.5, dashes="[2 3]")

# ── PDF PAGINA TEKENEN ────────────────────────────────────────────────────────

def draw_page(
    doc, matrix: np.ndarray, answer_matrix: np.ndarray, exercises: list[tuple],
    user_numbers: dict, cluster_colors: dict,
    show_colors: bool = True, show_answers: bool = False, cfg: PageConfig | None = None,
) -> None:
    """Voeg één pagina toe aan het PDF-document."""
    cfg  = cfg or PageConfig()
    page = doc.new_page(width=cfg.page_w, height=cfg.page_h)
    font_r, font_br = _load_playful_font(page)

    rows, cols   = matrix.shape
    half         = (len(exercises) + 1) // 2
    available_h  = cfg.page_h - 2 * cfg.margin
    ex_height    = half * cfg.row_height
    cell_size    = max(10, min(23, int((available_h - cfg.gap - ex_height) / rows)))
    grid_y       = cfg.margin + max(0, (available_h - (rows * cell_size + cfg.gap + ex_height)) // 2)
    grid_x       = cfg.margin
    fs           = cell_size * 0.55

    # ── Matrix ───────────────────────────────────────────────────────────────
    shape = page.new_shape()
    for r in range(rows):
        for c in range(cols):
            x0   = grid_x + c * cell_size
            y0   = grid_y + r * cell_size
            fill = cluster_colors.get(int(matrix[r][c]), (1, 1, 1)) if show_colors else (1, 1, 1)
            shape.draw_rect(fitz.Rect(x0, y0, x0 + cell_size, y0 + cell_size))
            shape.finish(color=(0.5, 0.5, 0.5), fill=fill, width=0.5)
    shape.commit()

    for r in range(rows):
        for c in range(cols):
            ans = str(answer_matrix[r][c])
            x0  = grid_x + c * cell_size
            y0  = grid_y + r * cell_size
            tw  = _text_len(ans, fontname=cfg.measure_font_b, fontsize=fs)
            page.insert_text((x0 + (cell_size - tw) / 2, y0 + cell_size / 2 + fs * 0.35),
                             ans, fontname=font_r, fontsize=fs, color=(0, 0, 0))

    # ── Instructietekst ───────────────────────────────────────────────────────
    instr_x    = grid_x + cols * cell_size + 20
    instr_width = cfg.page_w - cfg.margin - instr_x
    all_nums   = sorted(set(user_numbers.get("mult", []) + user_numbers.get("div", [])))

    # numbers_str in één regel met join
    numbers_str = (
        str(all_nums[0]) if len(all_nums) == 1
        else " en ".join(str(n) for n in all_nums) if len(all_nums) == 2
        else ", ".join(str(n) for n in all_nums[:-1]) + f" en {all_nums[-1]}"
    )
    tafel_word = "tafel" if len(all_nums) == 1 else "tafels"

    # Titel + instructieregels als één lijst
    title_lines = [
        (cfg.title_fontsize, font_br, "Rekentekenen met de"),
        (cfg.title_fontsize, font_br, f"{tafel_word} van {numbers_str}!"),
        (cfg.instr_fontsize, font_r,  ""),
        (cfg.instr_fontsize, font_r,  "Los alle oefeningen op en kleur"),
        (cfg.instr_fontsize, font_r,  "daarna de getallen hiernaast"),
        (cfg.instr_fontsize, font_r,  "in de juiste kleur!"),
    ]
    y_cursor = grid_y + 14
    for fs_line, fn, txt in title_lines:
        page.insert_text((instr_x, y_cursor), txt, fontname=fn, fontsize=fs_line, color=(0, 0, 0))
        y_cursor += fs_line + 4 if fn == font_br else cfg.line_h

    # ── QR Code ───────────────────────────────────────────────────────────────
    qr_y = y_cursor + 6
    if instr_width >= cfg.qr_size and (qr_y + cfg.qr_size + 12) < (grid_y + rows * cell_size):
        page.insert_image(fitz.Rect(instr_x, qr_y, instr_x + cfg.qr_size, qr_y + cfg.qr_size), stream=QR_PNG)
        page.insert_text((instr_x, qr_y + cfg.qr_size + 8), QR_URL,
                         fontname=font_r, fontsize=7, color=(0.4, 0.4, 0.4))

    # ── Oefeningen in 2 kolommen ──────────────────────────────────────────────
    y         = grid_y + rows * cell_size + cfg.gap
    col_width = (cfg.page_w - 2 * cfg.margin) / 2
    all_names = [get_color_name(*cluster_colors.get(cl, (0, 0, 0))) for cl, _, _ in exercises]
    ccw       = max(_text_len(n, fontname=cfg.measure_font_b, fontsize=cfg.ex_fontsize) for n in all_names) + 8

    for i, ex in enumerate(exercises[:half]):
        draw_exercise(page, ex, cfg.margin, y + i * cfg.row_height, col_width,
                      cluster_colors, ccw, font_r, font_br, cfg, show_answers)
    for i, ex in enumerate(exercises[half:]):
        draw_exercise(page, ex, cfg.margin + col_width + 4, y + i * cfg.row_height, col_width,
                      cluster_colors, ccw, font_r, font_br, cfg, show_answers)

# ── PDF GENEREREN ─────────────────────────────────────────────────────────────

def generate_pdf(
    user_numbers: dict, img_choice: int, num_pages: int, show_colors: bool,
    uploaded_bytes: bytes | None = None, show_answers: bool = False,
    num_exercises: int = NUM_EXERCISES, progress_bar=None,
) -> bytes:
    """Genereer het volledige PDF-document en geef het terug als bytes."""
    doc = fitz.open()
    cfg = PageConfig()
    image_queue, queue_warnings = build_image_queue(num_pages, img_choice)
    for w in queue_warnings:
        st.warning(w)

    for page_num in range(num_pages):
        if progress_bar is not None:
            progress_bar.progress(page_num / num_pages, text=f"Pagina {page_num + 1} van {num_pages}...")
        try:
            if uploaded_bytes is not None and page_num == 0:
                image_data: bytes | str = uploaded_bytes
            else:
                if page_num >= len(image_queue):
                    st.warning(f"Geen afbeelding beschikbaar voor pagina {page_num + 1}, overgeslagen.")
                    continue
                img_num    = image_queue[page_num]
                image_path = os.path.join(IMAGES_DIR, f"{img_num}.png")
                if not os.path.exists(image_path):
                    st.warning(f"Afbeelding {img_num}.png niet gevonden, pagina {page_num + 1} overgeslagen.")
                    continue
                image_data = image_path

            matrix, cluster_colors = process_image(image_data)
            num_clusters = len(cluster_colors)
            exercises, ex_warning = generate_math_exercises(
                user_numbers["mult"], user_numbers["div"],
                num_clusters=num_clusters, num_exercises=num_exercises,
            )
            if ex_warning:
                st.warning(ex_warning)
            answer_matrix = populate_matrix(matrix, exercises, num_clusters=num_clusters)

            if show_answers:
                draw_page(doc, matrix, answer_matrix, exercises, user_numbers, cluster_colors,
                          show_colors=False, show_answers=False, cfg=cfg)
                draw_page(doc, matrix, answer_matrix, exercises, user_numbers, cluster_colors,
                          show_colors=True,  show_answers=True,  cfg=cfg)
            else:
                draw_page(doc, matrix, answer_matrix, exercises, user_numbers, cluster_colors,
                          show_colors=show_colors, show_answers=False, cfg=cfg)

        except ValueError as e:
            st.error(str(e))
            doc.close()
            raise
        except Exception as e:
            st.warning(f"Pagina {page_num + 1} kon niet worden gegenereerd en is overgeslagen: {e}")

    buf = io.BytesIO()
    doc.save(buf)
    doc.close()
    buf.seek(0)
    return buf.read()

# ── STREAMLIT INTERFACE ───────────────────────────────────────────────────────

st.markdown("""<div class="title-block">
    <h1>🎨 Rekentekening</h1>
    <p class="subtitle">Maak een kleurplaat met rekenoefeningen!</p>
</div>""", unsafe_allow_html=True)

st.markdown("### 📚 Stap 1 — Welke tafels wil je oefenen?")
col1, col2 = st.columns(2)
with col1:
    mult_selected = st.multiselect("✖️ Maaltafels:", options=list(range(1, 11)), default=[2, 5],
                                   format_func=lambda x: f"Tafel van {x}")
with col2:
    div_selected  = st.multiselect("➗ Deeltafels:",  options=list(range(1, 11)), default=[2, 5],
                                   format_func=lambda x: f"Tafel van {x}")
selected = {"mult": mult_selected, "div": div_selected}

st.markdown("### 🖼️ Stap 2 — Kies een afbeelding")
img_mode = st.radio("", ["Willekeurig", "Specifiek nummer", "Eigen afbeelding uploaden"], horizontal=True)
img_choice     = 0
uploaded_bytes: bytes | None = None

if img_mode == "Specifiek nummer":
    img_choice   = st.slider("Afbeeldingsnummer:", min_value=1, max_value=MAX_IMAGE_NUMBER, value=1)
    preview_path = os.path.join(IMAGES_DIR, f"{img_choice}.png")
    st.image(preview_path) if os.path.exists(preview_path) else st.warning(f"Afbeelding {img_choice} niet gevonden.")
elif img_mode == "Eigen afbeelding uploaden":
    uploaded_file = st.file_uploader("Upload een afbeelding (PNG of JPG):", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        uploaded_bytes = uploaded_file.read()
        st.image(uploaded_bytes, caption="Geüploade afbeelding", width=200)
        st.info("De geüploade afbeelding wordt gebruikt voor de eerste pagina. Extra pagina's gebruiken willekeurige afbeeldingen.")

st.markdown("### 📄 Stap 3 — Hoeveel pagina's?")
num_pages = st.slider("Aantal pagina's:", min_value=1, max_value=40, value=1)

st.markdown("### ⚙️ Opties")
show_answers = st.toggle("Genereer ook een antwoordblad (2 pagina's per afbeelding)", value=False)
if show_answers:
    st.info("📄 Per afbeelding worden 2 pagina's gegenereerd: een werkblad (zonder kleur) en een antwoordblad (met kleur en oplossingen).")
show_colors   = st.toggle("Toon kleuren op werkblad", value=False)
num_exercises = st.slider("Aantal oefeningen per pagina:", min_value=10, max_value=40, value=40, step=5)

st.markdown("---")

if st.button("✏️ Genereer Rekentekening!"):
    if not selected["mult"] and not selected["div"]:
        st.error("Kies minstens één maaltafel of deeltafel!")
    elif img_mode == "Eigen afbeelding uploaden" and uploaded_bytes is None:
        st.error("Upload eerst een afbeelding!")
    else:
        try:
            progress  = st.progress(0, text="Bezig met genereren...")
            pdf_bytes = generate_pdf(selected, img_choice, num_pages, show_colors,
                                     uploaded_bytes=uploaded_bytes, show_answers=show_answers,
                                     num_exercises=num_exercises, progress_bar=progress)
            progress.empty()
            tafel_str = ", ".join(str(n) for n in sorted(set(selected["mult"] + selected["div"])))
            st.success(f"✅ {num_pages} pagina('s) klaar voor de tafels van {tafel_str}!")
            st.download_button("⬇️ Download Rekentekening.pdf", data=pdf_bytes,
                               file_name="Rekentekening.pdf", mime="application/pdf")
        except ValueError:
            pass
        except Exception as e:
            st.error(f"Er is een onverwachte fout opgetreden: {e}")
