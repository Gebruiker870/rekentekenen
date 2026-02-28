"""
Rekentekening — Streamlit app
Genereert een PDF met rekenoefeningen gekoppeld aan een kleurplaat.

Benodigde mapstructuur:
    rekentekening_app.py
    requirements.txt
    afbeeldingen/
        1.png  ...  117.png
"""

import io
import os
import random

import fitz  # PyMuPDF
import numpy as np
import streamlit as st
from PIL import Image

# ── CONSTANTEN ────────────────────────────────────────────────────────────────

IMAGES_DIR       = os.path.join(os.path.dirname(os.path.abspath(__file__)), "afbeeldingen")
MAX_IMAGE_SIZE   = 15
NUM_EXERCISES    = 40
MAX_IMAGE_NUMBER = 117

FONT_CANDIDATES = [
    "/Library/Fonts/Comic Sans MS.ttf",
    "/System/Library/Fonts/Supplemental/Comic Sans MS.ttf",
    "/Library/Fonts/Chalkboard.ttc",
    "C:/Windows/Fonts/comic.ttf",
]

# ── PAGINA CONFIG ─────────────────────────────────────────────────────────────

st.set_page_config(page_title="Rekentekening", page_icon="🎨", layout="centered")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Fredoka+One&family=Nunito:wght@400;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Nunito', sans-serif; }
    h1, h2, h3 { font-family: 'Fredoka One', cursive !important; }

    .stButton > button {
        font-family: 'Fredoka One', cursive;
        font-size: 1.2rem;
        background: linear-gradient(135deg, #f9a825, #e53935);
        color: white;
        border: none;
        border-radius: 16px;
        padding: 0.6rem 2rem;
        width: 100%;
        transition: transform 0.1s;
    }
    .stButton > button:hover { transform: scale(1.03); color: white; }

    .stDownloadButton > button {
        font-family: 'Fredoka One', cursive;
        font-size: 1.1rem;
        background: linear-gradient(135deg, #43a047, #00897b);
        color: white;
        border: none;
        border-radius: 16px;
        padding: 0.6rem 2rem;
        width: 100%;
    }
    .stDownloadButton > button:hover { color: white; }

    .title-block { text-align: center; padding: 1.5rem 0 0.5rem 0; }
    .title-block h1 {
        font-size: 3rem;
        background: linear-gradient(135deg, #f9a825, #e53935);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .subtitle { font-size: 1.1rem; color: #666; text-align: center; margin-top: 0; }
</style>
""", unsafe_allow_html=True)


# ── AFBEELDING VERWERKEN ──────────────────────────────────────────────────────

@st.cache_data
def process_image(image_source, max_size: int = MAX_IMAGE_SIZE):
    """Laad een afbeelding (pad of bestand), reduceer naar max_size en extraheer kleurclusters."""
    image = Image.open(image_source)
    image = image.resize((min(image.width, max_size), min(image.height, max_size)))

    # Transparantie vervangen door witte achtergrond
    if image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info):
        background = Image.new("RGB", image.size, (255, 255, 255))
        rgba = image.convert("RGBA")
        background.paste(rgba, mask=rgba.split()[3])
        image = background
    else:
        image = image.convert("RGB")

    image = image.convert("P", palette=Image.ADAPTIVE, colors=8)
    matrix = np.array(image)
    palette = image.getpalette()

    unique_vals = sorted(set(matrix.flatten()))
    remap = {v: i + 1 for i, v in enumerate(unique_vals)}
    matrix = np.vectorize(remap.get)(matrix)

    cluster_colors = {
        cluster_num: (
            palette[orig_idx * 3]     / 255.0,
            palette[orig_idx * 3 + 1] / 255.0,
            palette[orig_idx * 3 + 2] / 255.0,
        )
        for orig_idx, cluster_num in remap.items()
    }

    return matrix, cluster_colors


# ── OEFENINGEN GENEREREN ──────────────────────────────────────────────────────

def generate_math_exercises(mult_numbers: list, div_numbers: list, num_clusters: int = 8, num_exercises: int = NUM_EXERCISES):
    """Genereer unieke vermenigvuldigings- en deeloefeningen voor de opgegeven tafels."""
    seen = set()
    exercises = []
    answer_to_group = {}
    group_counter = 1

    available_ops = []
    if mult_numbers:
        available_ops.append("multiplication")
    if div_numbers:
        available_ops.append("division")
    if not available_ops:
        return []

    while len(exercises) < num_exercises and len(seen) < num_exercises * 20:
        operation = random.choice(available_ops)
        if operation == "multiplication":
            n1 = random.choice(mult_numbers)
            n2 = random.randint(1, 10)
            answer = n1 * n2
            expr = f"{n1} x {n2}"
        else:
            n2 = random.choice(div_numbers)
            n1 = random.randint(1, n2 * 10)
            while n1 % n2 != 0:
                n1 = random.randint(1, n2 * 10)
            answer = n1 // n2
            expr = f"{n1} : {n2}"

        if expr in seen:
            continue
        seen.add(expr)

        if answer not in answer_to_group:
            answer_to_group[answer] = group_counter
            group_counter += 1

        exercises.append((answer_to_group[answer], answer, f"{expr} = ?"))

    # Antwoordgroepen verdelen over de beschikbare kleurclusters
    unique_groups = sorted(set(g for g, _, _ in exercises))
    cluster_map = {g: (i % num_clusters) + 1 for i, g in enumerate(unique_groups)}
    exercises = [(cluster_map[g], ans, ex) for g, ans, ex in exercises]
    random.shuffle(exercises)

    return exercises


# ── MATRIX INVULLEN MET ANTWOORDEN ────────────────────────────────────────────

def populate_matrix(matrix: np.ndarray, exercises: list, num_clusters: int = 8):
    """Vul elke cel van de matrix met een antwoord van het bijhorende cluster."""
    cluster_answers = {i: [] for i in range(1, num_clusters + 1)}
    for cluster, answer, _ in exercises:
        if answer not in cluster_answers[cluster]:
            cluster_answers[cluster].append(answer)

    def cycling(answers):
        while True:
            shuffled = answers[:]
            random.shuffle(shuffled)
            yield from shuffled

    cycles = {c: cycling(ans) for c, ans in cluster_answers.items() if ans}
    result = np.full(matrix.shape, "?", dtype=object)

    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            cluster = int(matrix[row][col])
            if cluster in cycles:
                result[row][col] = next(cycles[cluster])

    return result


# ── KLEURNAAM BEPALEN ─────────────────────────────────────────────────────────

@st.cache_data
def get_color_name(r: float, g: float, b: float) -> str:
    """Geeft een Nederlandse kleurnaam terug op basis van RGB-waarden (0–1 bereik)."""
    r, g, b = r * 255, g * 255, b * 255
    max_c = max(r, g, b)
    min_c = min(r, g, b)
    sat = (max_c - min_c) / max_c if max_c > 0 else 0

    if max_c > 220 and sat < 0.06:                                      return "Wit"
    if max_c < 60:                                                       return "Zwart"
    if r > g and r > b and (r - b) > 10 and min_c > 100:               return "Lichtroze"
    if sat < 0.15:                                                       return "Grijs"
    if r > 100 and g > 50 and b < 80 and r > g and max_c < 200 and sat > 0.2:
        return "Bruin"

    if max_c == r:
        if g > 200 and b < 80:  return "Geel"
        if g > r * 0.75:        return "Geel"   # geel: groen bijna even hoog als rood
        if g > 80:              return "Oranje"
        if b > r * 0.7:         return "Paars"
        if b > 120:             return "Roze"
        return "Rood"

    if max_c == g:
        if r > 150:             return "Geel"
        if b > g * 0.85:        return "Lichtblauw"
        if r > g * 0.6:         return "Lichtgroen"
        return "Donkergroen"

    # max_c == b
    if r > 150:                 return "Paars"
    if g > 150:                 return "Lichtblauw"
    return "Blauw"


# ── PDF PAGINA TEKENEN ────────────────────────────────────────────────────────

# Cache voor fitz.get_text_length (wordt honderden keren aangeroepen per pagina)
_text_length_cache: dict = {}

def _text_len(text: str, fontname: str, fontsize: float) -> float:
    key = (text, fontname, round(fontsize, 2))
    if key not in _text_length_cache:
        _text_length_cache[key] = fitz.get_text_length(text, fontname=fontname, fontsize=fontsize)
    return _text_length_cache[key]

# Zoek het font eenmalig op bij het laden van de app
_PLAYFUL_FONT_PATH = None
for _path in FONT_CANDIDATES:
    if os.path.exists(_path):
        _PLAYFUL_FONT_PATH = _path
        break


def _load_playful_font(page) -> tuple:
    """Laadt een speels TTF-lettertype indien beschikbaar, anders Helvetica."""
    if _PLAYFUL_FONT_PATH:
        try:
            page.insert_font(fontname="playfont", fontfile=_PLAYFUL_FONT_PATH)
            return "playfont", "playfont"
        except Exception:
            pass
    return "helv", "hebo"


def draw_page(doc, matrix, answer_matrix, exercises, user_numbers, cluster_colors, show_colors=True, show_answers=False):
    """Voeg één pagina toe aan het PDF-document."""
    PAGE_W, PAGE_H   = 595, 842
    MARGIN           = 50
    EX_FONTSIZE      = 11
    ROW_HEIGHT       = 24
    GAP              = 15
    INSTR_FONTSIZE   = 13
    TITLE_FONTSIZE   = INSTR_FONTSIZE + 5
    LINE_H           = 19
    MEASURE_FONT     = "helv"   # fitz.get_text_length vereist ingebouwde fonts
    MEASURE_FONT_B   = "hebo"

    page = doc.new_page(width=PAGE_W, height=PAGE_H)
    font_r, font_br = _load_playful_font(page)

    rows, cols = matrix.shape
    half = (len(exercises) + 1) // 2
    available_h = PAGE_H - 2 * MARGIN
    ex_height = half * ROW_HEIGHT

    cell_size = max(10, min(23, int((available_h - GAP - ex_height) / rows)))
    total_h = rows * cell_size + GAP + ex_height
    grid_y = MARGIN + max(0, (available_h - total_h) // 2)
    grid_x = MARGIN

    # ── Matrix ────────────────────────────────────────────────────────────────
    for r in range(rows):
        for c in range(cols):
            cluster = int(matrix[r][c])
            answer  = str(answer_matrix[r][c])
            x0, y0  = grid_x + c * cell_size, grid_y + r * cell_size
            x1, y1  = x0 + cell_size, y0 + cell_size

            fill = cluster_colors.get(cluster, (1, 1, 1)) if show_colors else (1, 1, 1)
            page.draw_rect(fitz.Rect(x0, y0, x1, y1), color=(0.5, 0.5, 0.5), fill=fill, width=0.5)

            fs = cell_size * 0.55
            tw = _text_len(answer, fontname=MEASURE_FONT_B, fontsize=fs)
            page.insert_text(
                (x0 + (cell_size - tw) / 2, y0 + cell_size / 2 + fs * 0.35),
                answer, fontname=font_r, fontsize=fs, color=(0, 0, 0)
            )

    # ── Instructietekst ───────────────────────────────────────────────────────
    instr_x = grid_x + cols * cell_size + 20

    # user_numbers is een dict {"mult": [...], "div": [...]}
    all_numbers = sorted(set(user_numbers.get("mult", []) + user_numbers.get("div", [])))
    if len(all_numbers) == 1:
        numbers_str = str(all_numbers[0])
    elif len(all_numbers) == 2:
        numbers_str = f"{all_numbers[0]} en {all_numbers[1]}"
    else:
        numbers_str = ", ".join(str(n) for n in all_numbers[:-1]) + f" en {all_numbers[-1]}"

    tafel_word = "tafel" if len(all_numbers) == 1 else "tafels"

    page.insert_text((instr_x, grid_y + 14),
        "Rekentekenen met de", fontname=font_br, fontsize=TITLE_FONTSIZE, color=(0, 0, 0))
    page.insert_text((instr_x, grid_y + 14 + TITLE_FONTSIZE + 4),
        f"{tafel_word} van {numbers_str}!", fontname=font_br, fontsize=TITLE_FONTSIZE, color=(0, 0, 0))

    base_y = grid_y + 14 + (TITLE_FONTSIZE + 4) * 2
    for i, line in enumerate(["", "Los alle oefeningen op en kleur", "daarna de getallen hiernaast", "in de juiste kleur!"]):
        page.insert_text((instr_x, base_y + i * LINE_H), line, fontname=font_r, fontsize=INSTR_FONTSIZE, color=(0, 0, 0))

    # ── Oefeningen in 2 kolommen ──────────────────────────────────────────────
    y = grid_y + rows * cell_size + GAP
    col_width = (PAGE_W - 2 * MARGIN) / 2
    left_col, right_col = exercises[:half], exercises[half:]

    all_names = [get_color_name(*cluster_colors.get(cl, (0, 0, 0))) for cl, _, _ in exercises]
    color_col_w = max(_text_len(n, fontname=MEASURE_FONT_B, fontsize=EX_FONTSIZE) for n in all_names) + 8

    def draw_exercise(ex, x_start, ex_y, col_w):
        cluster, answer, text = ex
        cr, cg, cb = cluster_colors.get(cluster, (0, 0, 0))
        color_name = get_color_name(cr, cg, cb)
        name_w = _text_len(color_name, fontname=MEASURE_FONT_B, fontsize=EX_FONTSIZE)

        page.draw_rect(
            fitz.Rect(x_start, ex_y + 2, x_start + color_col_w, ex_y + ROW_HEIGHT - 2),
            color=None, fill=(cr, cg, cb), width=0
        )
        luminance = 0.299 * cr + 0.587 * cg + 0.114 * cb
        page.insert_text(
            (x_start + (color_col_w - name_w) / 2, ex_y + 16),
            color_name, fontname=font_br, fontsize=EX_FONTSIZE,
            color=(1, 1, 1) if luminance < 0.5 else (0, 0, 0)
        )

        ex_x = x_start + color_col_w + 6
        label = text.replace("= ?", "=")
        page.insert_text((ex_x, ex_y + 16), label, fontname=font_r, fontsize=EX_FONTSIZE, color=(0, 0, 0))
        label_w = _text_len(label, fontname=MEASURE_FONT, fontsize=EX_FONTSIZE)
        if show_answers:
            # Schrijf het antwoord in het zwart na het = teken
            ans_str = str(answer)
            page.insert_text((ex_x + label_w + 4, ex_y + 16), ans_str, fontname=font_br, fontsize=EX_FONTSIZE, color=(0, 0, 0))
        else:
            # Stippellijn als invulruimte
            line_x0 = ex_x + label_w + 4
            page.draw_line((line_x0, ex_y + 15), (x_start + col_w - 8, ex_y + 15),
                           color=(0, 0, 0), width=0.5, dashes="[2 3]")

    for i, ex in enumerate(left_col):
        draw_exercise(ex, MARGIN, y + i * ROW_HEIGHT, col_width)
    for i, ex in enumerate(right_col):
        draw_exercise(ex, MARGIN + col_width + 4, y + i * ROW_HEIGHT, col_width)


# ── PDF GENEREREN ─────────────────────────────────────────────────────────────

def generate_pdf(user_numbers: list, img_choice: int, num_pages: int, show_colors: bool,
                 uploaded_image=None, show_answers: bool = False, num_exercises: int = NUM_EXERCISES) -> bytes:
    """Genereer het volledige PDF-document en geef het terug als bytes."""
    doc = fitz.open()
    used_images = []

    for page_num in range(num_pages):
        # Gebruik geüploade afbeelding voor de eerste pagina
        if uploaded_image is not None and page_num == 0:
            image_source = uploaded_image
        else:
            # Eerste pagina: gebruik specifieke keuze indien opgegeven
            # Volgende pagina's: willekeurig (zonder herhaling)
            if img_choice != 0 and page_num == 0:
                img_num = img_choice
                used_images.append(img_num)
            else:
                available = [i for i in range(1, MAX_IMAGE_NUMBER + 1) if i not in used_images]
                if not available:
                    used_images.clear()
                    available = list(range(1, MAX_IMAGE_NUMBER + 1))
                img_num = random.choice(available)
                used_images.append(img_num)

            image_path = os.path.join(IMAGES_DIR, f"{img_num}.png")
            if not os.path.exists(image_path):
                continue
            image_source = image_path

        matrix, cluster_colors = process_image(image_source)
        exercises     = generate_math_exercises(user_numbers["mult"], user_numbers["div"], num_clusters=len(cluster_colors), num_exercises=num_exercises)
        answer_matrix = populate_matrix(matrix, exercises, num_clusters=len(cluster_colors))

        if show_answers:
            # Pagina 1: zonder kleur en zonder oplossingen (voor de leerling)
            draw_page(doc, matrix, answer_matrix, exercises, user_numbers, cluster_colors,
                      show_colors=False, show_answers=False)
            # Pagina 2: met kleur en met oplossingen (antwoordblad)
            draw_page(doc, matrix, answer_matrix, exercises, user_numbers, cluster_colors,
                      show_colors=True, show_answers=True)
        else:
            draw_page(doc, matrix, answer_matrix, exercises, user_numbers, cluster_colors,
                      show_colors=show_colors, show_answers=False)

    buf = io.BytesIO()
    doc.save(buf)
    doc.close()
    buf.seek(0)
    return buf.read()


# ── STREAMLIT INTERFACE ───────────────────────────────────────────────────────

st.markdown("""
<div class="title-block">
    <h1>🎨 Rekentekening</h1>
    <p class="subtitle">Maak een kleurplaat met rekenoefeningen!</p>
</div>
""", unsafe_allow_html=True)

st.markdown("### 📚 Stap 1 — Welke tafels wil je oefenen?")
col1, col2 = st.columns(2)
with col1:
    mult_selected = st.multiselect(
        "✖️ Maaltafels:",
        options=list(range(1, 11)),
        default=[2, 5],
        format_func=lambda x: f"Tafel van {x}",
    )
with col2:
    div_selected = st.multiselect(
        "➗ Deeltafels:",
        options=list(range(1, 11)),
        default=[2, 5],
        format_func=lambda x: f"Tafel van {x}",
    )
selected = {"mult": mult_selected, "div": div_selected}

st.markdown("### 🖼️ Stap 2 — Kies een afbeelding")
img_mode = st.radio("", ["Willekeurig", "Specifiek nummer", "Eigen afbeelding uploaden"], horizontal=True)
img_choice = 0
uploaded_image = None

if img_mode == "Specifiek nummer":
    img_choice = st.slider("Afbeeldingsnummer:", min_value=1, max_value=MAX_IMAGE_NUMBER, value=1)
    preview_path = os.path.join(IMAGES_DIR, f"{img_choice}.png")
    if os.path.exists(preview_path):
        st.image(preview_path)
    else:
        st.warning(f"Afbeelding {img_choice} niet gevonden.")
elif img_mode == "Eigen afbeelding uploaden":
    uploaded_file = st.file_uploader("Upload een afbeelding (PNG of JPG):", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        uploaded_image = io.BytesIO(uploaded_file.read())
        st.image(uploaded_file, caption="Geüploade afbeelding", width=200)
        st.info("De geüploade afbeelding wordt gebruikt voor de eerste pagina. Extra pagina's gebruiken willekeurige afbeeldingen.")

st.markdown("### 📄 Stap 3 — Hoeveel pagina's?")
num_pages = st.slider("Aantal pagina's:", min_value=1, max_value=40, value=1)

st.markdown("### ⚙️ Opties")
show_answers = st.toggle("Genereer ook een antwoordblad (2 pagina's per afbeelding)", value=False)
if show_answers:
    st.info("📄 Per afbeelding worden 2 pagina's gegenereerd: een werkblad (zonder kleur) en een antwoordblad (met kleur en oplossingen).")
show_colors = False
num_exercises = st.slider("Aantal oefeningen per pagina:", min_value=10, max_value=40, value=40, step=5)

st.markdown("---")

if st.button("✏️ Genereer Rekentekening!"):
    if not selected["mult"] and not selected["div"]:
        st.error("Kies minstens één maaltafel of deeltafel!")
    else:
        with st.spinner("Bezig met genereren..."):
            if img_mode == "Eigen afbeelding uploaden" and uploaded_image is None:
                st.error("Upload eerst een afbeelding!")
                st.stop()
            pdf_bytes = generate_pdf(selected, img_choice, num_pages, show_colors, uploaded_image=uploaded_image, show_answers=show_answers, num_exercises=num_exercises)

        all_selected = sorted(set(selected["mult"] + selected["div"]))
        tafel_str = ", ".join(str(n) for n in all_selected)
        st.success(f"✅ {num_pages} pagina('s) klaar voor de tafels van {tafel_str}!")
        st.download_button(
            label="⬇️ Download Rekentekening.pdf",
            data=pdf_bytes,
            file_name="Rekentekening.pdf",
            mime="application/pdf",
        )
