"""
Einfache API-Key-Authentifizierung über den Header "X-API-Key".

Bewusst simpel gehalten - ein einzelner, geteilter Key statt Pro-User-
Tokens/OAuth2. Passend für einen einzelnen Client oder eine kleine,
vertrauenswürdige Gruppe, NICHT für ein Multi-User-Produkt mit
unterschiedlichen Rechten pro Nutzer (dafür bräuchte es echte
Nutzerkonten + Tokens statt eines geteilten Geheimnisses).

Design-Entscheidung "fail open" statt "fail closed": Ist config.API_KEY
nicht gesetzt, bleibt der Endpoint bewusst offen (mit deutlicher Warnung
beim Start), statt den Server-Start zu verweigern. Das verhindert, dass
lokale Entwicklung ohne .env kaputtgeht. Vor jedem Einsatz ausserhalb von
localhost MUSS API_KEY in der .env gesetzt werden.
"""

import logging
import secrets

from fastapi import Header, HTTPException, Security
from fastapi.security import APIKeyHeader

from . import config

logger = logging.getLogger(__name__)

if not config.API_KEY:
    logger.warning(
        "Kein API_KEY gesetzt - POST /query ist OHNE Zugriffsschutz "
        "erreichbar! API_KEY in der .env setzen, bevor die API ausserhalb "
        "von localhost erreichbar ist."
    )

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def require_api_key(x_api_key: str = Security(api_key_header)) -> None:
    """FastAPI-Dependency, die (falls konfiguriert) einen gültigen
    X-API-Key-Header verlangt."""
    if not config.API_KEY:
        # Kein Key konfiguriert -> bewusst offen (Fail-Open)
        return

    if not x_api_key or not secrets.compare_digest(x_api_key, config.API_KEY):
        raise HTTPException(status_code=401, detail="Ungültiger oder fehlender API-Key.")