STATIC_LABELS = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I',
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S',
    'T', 'U', 'V', 'W', 'X', 'Y',
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'hola', 'gracias', 'si', 'no', 'por_favor', 'ayuda',
]

# J y Z requieren movimiento, se detectan con el modelo de secuencias
DYNAMIC_LABELS = [
    # Letras con movimiento en LSM
    'J', 'Z',
    # Expresiones y respuestas comunes
    'adios', 'bien', 'mal',
    # Verbos y palabras de uso frecuente
    'quiero', 'comer', 'beber',
    'ir', 'venir',
    'mucho', 'poco',
    'nombre', 'llamar',
]

SEQUENCE_LENGTH = 30  # frames por seña dinámica
