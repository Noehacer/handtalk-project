"""
Puebla la base de datos con 130+ señas y genera imágenes placeholder.

Uso:
  python seed_dictionary.py

Crea:
  - Registros en la tabla 'signs' de handtalk.db
  - Imágenes PNG de placeholder en backend/assets/
    (coloreadas por categoría, con el nombre de la seña)

Sustituye los placeholders con fotos/GIFs reales cuando los tengas.
"""

import os
from PIL import Image, ImageDraw, ImageFont
from database import init_db, upsert_sign, count_signs

BASE_DIR = os.path.dirname(__file__)
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
os.makedirs(ASSETS_DIR, exist_ok=True)

# ── Paleta por categoría ──────────────────────────────────────────────────────

COLORS = {
    "saludos":     ((41,  182, 246), (13,  71, 161)),
    "cortesias":   ((129, 199, 132), (27, 94,  32)),
    "respuestas":  ((255, 183,  77), (230, 81,   0)),
    "familia":     ((240, 98,  146), (136,  14,  79)),
    "numeros":     ((77,  182, 172), (0,   77,  64)),
    "colores":     ((206, 147, 216), (74,   20, 140)),
    "emociones":   ((255, 138, 101), (191,  54,  12)),
    "lugares":     ((79,  195, 247), (1,   87, 155)),
    "verbos":      ((239,  83,  80), (183,  28,  28)),
    "tiempo":      ((100, 181, 246), (21,  101, 192)),
    "preguntas":   ((255, 213,  79), (245, 127,  23)),
    "cuerpo":      ((174, 213, 129), (51,  105,  30)),
    "emergencias": ((229,  57,  53), (183,  28,  28)),
}
DEFAULT_COLORS = ((158, 158, 158), (66, 66, 66))

# ── Diccionario completo ──────────────────────────────────────────────────────

DICTIONARY = [
    # Saludos
    ("hola",          "saludos"),
    ("adios",         "saludos"),
    ("buenos_dias",   "saludos"),
    ("buenas_tardes", "saludos"),
    ("buenas_noches", "saludos"),
    ("bienvenido",    "saludos"),
    ("hasta_luego",   "saludos"),
    ("hasta_pronto",  "saludos"),
    ("mucho_gusto",   "saludos"),
    ("con_permiso",   "saludos"),
    # Cortesías
    ("por_favor",     "cortesias"),
    ("gracias",       "cortesias"),
    ("de_nada",       "cortesias"),
    ("disculpe",      "cortesias"),
    ("perdon",        "cortesias"),
    ("lo_siento",     "cortesias"),
    # Respuestas
    ("si",            "respuestas"),
    ("no",            "respuestas"),
    ("tal_vez",       "respuestas"),
    ("no_se",         "respuestas"),
    ("claro",         "respuestas"),
    ("entiendo",      "respuestas"),
    ("repite",        "respuestas"),
    ("espera",        "respuestas"),
    ("correcto",      "respuestas"),
    ("incorrecto",    "respuestas"),
    # Familia
    ("mama",          "familia"),
    ("papa",          "familia"),
    ("hermano",       "familia"),
    ("hermana",       "familia"),
    ("abuelo",        "familia"),
    ("abuela",        "familia"),
    ("tio",           "familia"),
    ("tia",           "familia"),
    ("primo",         "familia"),
    ("bebe",          "familia"),
    ("familia",       "familia"),
    ("esposo",        "familia"),
    ("esposa",        "familia"),
    ("hijo",          "familia"),
    ("hija",          "familia"),
    ("amigo",         "familia"),
    # Números
    ("cero",          "numeros"),
    ("uno",           "numeros"),
    ("dos",           "numeros"),
    ("tres",          "numeros"),
    ("cuatro",        "numeros"),
    ("cinco",         "numeros"),
    ("seis",          "numeros"),
    ("siete",         "numeros"),
    ("ocho",          "numeros"),
    ("nueve",         "numeros"),
    ("diez",          "numeros"),
    ("veinte",        "numeros"),
    ("cien",          "numeros"),
    ("mil",           "numeros"),
    # Colores
    ("rojo",          "colores"),
    ("azul",          "colores"),
    ("verde",         "colores"),
    ("amarillo",      "colores"),
    ("blanco",        "colores"),
    ("negro",         "colores"),
    ("naranja",       "colores"),
    ("morado",        "colores"),
    ("rosa",          "colores"),
    ("cafe",          "colores"),
    ("gris",          "colores"),
    # Emociones
    ("feliz",         "emociones"),
    ("triste",        "emociones"),
    ("enojado",       "emociones"),
    ("asustado",      "emociones"),
    ("sorprendido",   "emociones"),
    ("cansado",       "emociones"),
    ("hambre",        "emociones"),
    ("sed",           "emociones"),
    ("dolor",         "emociones"),
    ("bien",          "emociones"),
    ("mal",           "emociones"),
    ("amor",          "emociones"),
    ("aburrido",      "emociones"),
    ("nervioso",      "emociones"),
    # Lugares
    ("casa",          "lugares"),
    ("escuela",       "lugares"),
    ("hospital",      "lugares"),
    ("trabajo",       "lugares"),
    ("tienda",        "lugares"),
    ("calle",         "lugares"),
    ("ciudad",        "lugares"),
    ("mexico",        "lugares"),
    ("bano",          "lugares"),
    ("cuarto",        "lugares"),
    ("cocina",        "lugares"),
    ("parque",        "lugares"),
    ("iglesia",       "lugares"),
    # Verbos
    ("comer",         "verbos"),
    ("beber",         "verbos"),
    ("dormir",        "verbos"),
    ("trabajar",      "verbos"),
    ("estudiar",      "verbos"),
    ("hablar",        "verbos"),
    ("escuchar",      "verbos"),
    ("ver",           "verbos"),
    ("caminar",       "verbos"),
    ("correr",        "verbos"),
    ("ayudar",        "verbos"),
    ("querer",        "verbos"),
    ("saber",         "verbos"),
    ("poder",         "verbos"),
    ("ir",            "verbos"),
    ("venir",         "verbos"),
    ("abrir",         "verbos"),
    ("cerrar",        "verbos"),
    ("leer",          "verbos"),
    ("escribir",      "verbos"),
    ("llamar",        "verbos"),
    ("esperar",       "verbos"),
    # Tiempo
    ("hoy",           "tiempo"),
    ("manana",        "tiempo"),
    ("ayer",          "tiempo"),
    ("hora",          "tiempo"),
    ("dia",           "tiempo"),
    ("semana",        "tiempo"),
    ("mes",           "tiempo"),
    ("ano",           "tiempo"),
    ("tarde",         "tiempo"),
    ("noche",         "tiempo"),
    ("ahora",         "tiempo"),
    ("despues",       "tiempo"),
    ("antes",         "tiempo"),
    # Preguntas
    ("que",           "preguntas"),
    ("quien",         "preguntas"),
    ("donde",         "preguntas"),
    ("cuando",        "preguntas"),
    ("como",          "preguntas"),
    ("cuanto",        "preguntas"),
    ("por_que",       "preguntas"),
    ("cual",          "preguntas"),
    # Cuerpo
    ("cabeza",        "cuerpo"),
    ("mano",          "cuerpo"),
    ("ojo",           "cuerpo"),
    ("oreja",         "cuerpo"),
    ("boca",          "cuerpo"),
    ("nariz",         "cuerpo"),
    ("brazo",         "cuerpo"),
    ("pierna",        "cuerpo"),
    ("pie",           "cuerpo"),
    ("corazon",       "cuerpo"),
    ("espalda",       "cuerpo"),
    ("cara",          "cuerpo"),
    # Emergencias
    ("ayuda",         "emergencias"),
    ("peligro",       "emergencias"),
    ("emergencia",    "emergencias"),
    ("policia",       "emergencias"),
    ("ambulancia",    "emergencias"),
    ("fuego",         "emergencias"),
    ("accidente",     "emergencias"),
]


# ── Generador de imágenes placeholder ────────────────────────────────────────

_CATEGORY_COLORS = {
    "saludos":    (52, 152, 219),
    "cortesias":  (46, 204, 113),
    "familia":    (155, 89, 182),
    "numeros":    (230, 126, 34),
    "colores":    (231, 76, 60),
    "abecedario": (52, 73, 94),
    "preguntas":  (241, 196, 15),
    "tiempo":     (26, 188, 156),
    "verbos":     (211, 84, 0),
    "default":    (127, 140, 141),
}


def create_placeholder(word: str, filename: str, category: str = "default"):
    size  = 200
    color = _CATEGORY_COLORS.get(category, _CATEGORY_COLORS["default"])
    img   = Image.new("RGB", (size, size), color)
    draw  = ImageDraw.Draw(img)

    # Gradient band (lighter at top)
    for i in range(size // 2):
        alpha = int(255 * (i / (size // 2)) * 0.3)
        band  = tuple(min(255, c + alpha) for c in color)
        draw.rectangle([0, i, size, i + 1], fill=band)

    # Simple hand icon: palm ellipse + finger rectangles
    cx, cy = size // 2, size // 2 - 20
    draw.ellipse([cx - 25, cy - 30, cx + 25, cy + 10], fill="white")
    for dx in [-20, -8, 4, 16]:
        draw.rectangle([cx + dx, cy - 50, cx + dx + 8, cy - 15], fill="white")
    draw.rectangle([cx - 28, cy - 35, cx - 14, cy - 10], fill="white")

    # Sign name
    word_display = word.replace("_", " ").upper()
    font_size    = max(14, 28 - max(0, len(word_display) - 6) * 2)
    try:
        font  = ImageFont.truetype("arial.ttf", font_size)
        small = ImageFont.truetype("arial.ttf", 11)
    except Exception:
        font  = ImageFont.load_default()
        small = font

    bbox = draw.textbbox((0, 0), word_display, font=font)
    tw   = bbox[2] - bbox[0]
    draw.text(((size - tw) // 2, size - 55), word_display, font=font, fill="white")

    cat_text = f"#{category}"
    bbox2    = draw.textbbox((0, 0), cat_text, font=small)
    tw2      = bbox2[2] - bbox2[0]
    draw.text(((size - tw2) // 2, size - 25), cat_text, font=small, fill=(220, 220, 220))

    dest = os.path.join(ASSETS_DIR, filename)
    img.save(dest)


# ── Seeder principal ──────────────────────────────────────────────────────────

def seed():
    init_db()
    before = count_signs()

    print(f"Generando {len(DICTIONARY)} señas...\n")
    created = 0
    skipped = 0

    for word, category in DICTIONARY:
        filename = f"{word}.png"
        full_path = os.path.join(ASSETS_DIR, filename)

        # Generar placeholder solo si no existe imagen real
        if not os.path.exists(full_path):
            create_placeholder(word, filename, category)
            created += 1
        else:
            skipped += 1

        upsert_sign(word=word, filename=filename, media_type="image/png", category=category)

    after = count_signs()
    print(f"  Imágenes creadas : {created}")
    print(f"  Imágenes existentes (sin tocar): {skipped}")
    print(f"  Señas en BD      : {before} → {after}")
    print(f"\nAssets guardados en: {ASSETS_DIR}")
    print("Listo. Sustituye los PNGs con fotos/GIFs reales cuando los tengas.")


if __name__ == "__main__":
    seed()
