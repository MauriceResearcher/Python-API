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

from fastapi import Header, HTTPException

from . import config

logger = logging.getLogger(__name__)

if not config.API_KEY:
    logger.warning(
        "Kein API_KEY gesetzt - POST /query ist OHNE Zugriffsschutz "
        "erreichbar! API_KEY in der .env setzen, bevor die API ausserhalb "
        "von localhost erreichbar ist."
    )


async def require_api_key(x_api_key: str = Header(default=None)) -> None:
    """FastAPI-Dependency, die (falls konfiguriert) einen gültigen
    X-API-Key-Header verlangt. Als dependencies=[Depends(...)] auf einer
    Route eingesetzt, statt als Funktionsparameter, weil der Rückgabewert
    (None) nirgends gebraucht wird - es geht nur ums "durchlassen oder
    nicht"."""
    if not config.API_KEY:
        # Kein Key konfiguriert -> bewusst offen, siehe Modul-Docstring.
        return

    # secrets.compare_digest() statt "==": ein normaler String-Vergleich
    # bricht bei der ersten abweichenden Stelle ab, wodurch die
    # Vergleichsdauer indirekt verrät, wie viele Zeichen am Anfang bereits
    # korrekt waren (Timing-Angriff). compare_digest() vergleicht immer in
    # konstanter Zeit.
    if not x_api_key or not secrets.compare_digest(x_api_key, config.API_KEY):
        raise HTTPException(status_code=401, detail="Ungültiger oder fehlender API-Key.")