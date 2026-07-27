"""
RAG pipeline -> Docs lesen, metadaten auslesen, zweifaches chunking

"""


import re
import requests
import html2text
from bs4 import BeautifulSoup
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter


# test_script = "https://docs.python.org/3/tutorial/controlflow.html"

def load_python_docs(url):


    # timeout=(connect, read): 5s um die Verbindung aufzubauen, 30s um die
    # Antwort zu lesen. Ohne Timeout würde ein nicht antwortender Server
    # den kompletten App-Start (build() beim ersten Hochfahren) für immer
    # blockieren.
    response = requests.get(url, timeout=(5, 30))
    response.raise_for_status()

    soup = BeautifulSoup(response.content, "html.parser")

    # 1. Den echten Seitentitel aus dem <h1>-Tag des HTMLs lesen
    page_title_tag = soup.find("h1")
    page_title = page_title_tag.get_text(strip=True) if page_title_tag else "Python Docs"

    # 2. Permalinks (¶) aus dem HTML löschen
    for link in soup.find_all("a", class_="headerlink"):
        link.decompose()

    # 3. Hauptinhalt isolieren
    main_content = soup.find("div", {"role": "main"}) or soup.find("div", class_="body")

    # 4. In Markdown umwandeln
    h = html2text.HTML2Text()
    h.ignore_links = True
    h.ignore_images = True
    h.body_width = 0
    markdown_text = h.handle(str(main_content))

    # 5. Cleanup
    markdown_text = re.sub(r'\\([._\-])', r'\1', markdown_text)

    # Wir geben sowohl das Markdown als auch den ermittelten Titel zurück
    return markdown_text, page_title


# Aufruf der Funktion
# test_markdown, page_title = load_python_docs(test_script)

def chunk_splitter(markdown_text, page_title):
    # 6. Splitten ab ## (H2), ### (H3) und #### (H4)
    headers_to_split_on = [
        ("##", "header_2"),
        ("###", "header_3"),
        ("####", "header_4"),
    ]

    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False
    )
    chunks = markdown_splitter.split_text(markdown_text)

    # 7. Den verlässlichen Seitentitel manuell in alle Chunk-Metadaten schreiben
    for chunk in chunks:
        chunk.metadata["header_1"] = page_title



        # nochmal character split
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )

    chunks = text_splitter.split_documents(chunks)

    return chunks


"""

Alter Testcode
# Ergebnis prüfen
print(f"Anzahl finaler Chunks: {len(chunks)}")
print("\n--- BEISPIEL CHUNK METADATEN ---")
print(chunks[16].metadata)
print("\n--- BEISPIEL CHUNK INHALT ---")
print(chunks[16].page_content[:300])
"""