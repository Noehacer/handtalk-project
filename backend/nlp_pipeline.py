import config
from logger import get_logger

log = get_logger(__name__)

_INVALID = {
    "NO RECONOCIDO", "NO DETECTADO", "CAPTURANDO...",
    "MODELO NO DISPONIBLE", "ERROR",
}

_SYSTEM_PROMPT = (
    "Eres un intérprete especializado en Lengua de Señas Mexicana (LSM). "
    "Recibirás una secuencia de señas detectadas en orden LSM (SOV — Sujeto-Objeto-Verbo). "
    "Tradúcelas a una oración natural en español considerando:\n"
    "- LSM usa orden SOV; español usa SVO — reordena adecuadamente\n"
    "- Las negaciones se expresan al final en LSM\n"
    "- Omite artículos y preposiciones que no existen en LSM\n"
    "- Infiere tiempos verbales del contexto si no hay marcadores temporales\n"
    "- Si la secuencia es 1-2 señas, retorna la traducción directa\n"
    "Responde SOLO con la oración traducida. Sin explicaciones."
)


def normalize_signs(signs: list[str]) -> list[str]:
    """Removes invalid/duplicate consecutive signs. Preserves original casing."""
    result, last_upper = [], ""
    for s in signs:
        upper = s.upper()
        if upper in _INVALID:
            continue
        if upper != last_upper:
            result.append(s)
            last_upper = upper
    return result


async def translate_to_spanish(signs: list[str]) -> str:
    """Translates a sequence of LSM signs to natural Spanish using Claude API.
    Falls back to space-joined signs when API key is not configured.
    """
    normalized = normalize_signs(signs)
    if not normalized:
        return ""
    if len(normalized) == 1:
        return normalized[0].capitalize()
    if not config.ANTHROPIC_API_KEY:
        return " ".join(normalized)
    try:
        import anthropic
        client   = anthropic.AsyncAnthropic(api_key=config.ANTHROPIC_API_KEY)
        sequence = " → ".join(normalized)
        msg      = await client.messages.create(
            model=config.NLP_MODEL,
            max_tokens=256,
            system=[{
                "type": "text",
                "text": _SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }],
            messages=[{"role": "user", "content": f"Señas LSM: {sequence}"}],
        )
        return msg.content[0].text.strip()
    except Exception as exc:
        log.error(f"Error Claude API: {exc}")
        return " ".join(normalized)
