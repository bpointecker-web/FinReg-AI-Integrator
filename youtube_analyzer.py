"""YouTube-Video-Analyzer: Transkript abrufen und via Claude API zusammenfassen."""

import argparse
import os
import re
import sys
from urllib.parse import parse_qs, urlparse

import anthropic
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    NoTranscriptFound,
    TranscriptsDisabled,
    VideoUnavailable,
)

SYSTEM_PROMPT = """Du bist ein Analyst, der YouTube-Video-Transkripte prägnant zusammenfasst.
Erstelle eine strukturierte Zusammenfassung mit:

1. **Kernthema** (1-2 Sätze)
2. **Hauptpunkte** (3-6 Bullet Points)
3. **Erkenntnisse & Takeaways** (2-4 Bullet Points)
4. **Zielgruppe** (1 Satz)

Antworte auf Deutsch, sachlich und ohne Füllwörter."""


def extract_video_id(url_or_id: str) -> str:
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", url_or_id):
        return url_or_id

    parsed = urlparse(url_or_id)
    if parsed.hostname in ("youtu.be",):
        return parsed.path.lstrip("/")
    if parsed.hostname and "youtube.com" in parsed.hostname:
        if parsed.path == "/watch":
            return parse_qs(parsed.query).get("v", [""])[0]
        match = re.match(r"^/(embed|shorts|v)/([A-Za-z0-9_-]{11})", parsed.path)
        if match:
            return match.group(2)

    raise ValueError(f"Konnte keine Video-ID aus '{url_or_id}' extrahieren.")


def fetch_transcript(video_id: str, languages: list[str]) -> tuple[str, str]:
    api = YouTubeTranscriptApi()
    transcript_list = api.list(video_id)

    try:
        transcript = transcript_list.find_transcript(languages)
    except NoTranscriptFound:
        transcript = next(iter(transcript_list))

    fetched = transcript.fetch()
    text = " ".join(snippet.text.strip() for snippet in fetched if snippet.text.strip())
    return text, transcript.language_code


def summarize(transcript: str, video_id: str, language_code: str) -> None:
    client = anthropic.Anthropic()

    user_message = (
        f"Video-ID: {video_id}\n"
        f"Transkript-Sprache: {language_code}\n\n"
        f"Transkript:\n{transcript}"
    )

    with client.messages.stream(
        model="claude-opus-4-7",
        max_tokens=64000,
        thinking={"type": "adaptive"},
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    ) as stream:
        for text in stream.text_stream:
            print(text, end="", flush=True)

        final = stream.get_final_message()

    print(
        f"\n\n---\nTokens: input={final.usage.input_tokens}, "
        f"output={final.usage.output_tokens}",
        file=sys.stderr,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="YouTube-Transkript abrufen und via Claude API zusammenfassen."
    )
    parser.add_argument("url", help="YouTube-URL oder Video-ID")
    parser.add_argument(
        "--lang",
        default="de,en",
        help="Bevorzugte Transkript-Sprachen (komma-getrennt, Standard: de,en)",
    )
    args = parser.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print(
            "Fehler: Umgebungsvariable ANTHROPIC_API_KEY ist nicht gesetzt.",
            file=sys.stderr,
        )
        return 1

    try:
        video_id = extract_video_id(args.url)
    except ValueError as e:
        print(f"Fehler: {e}", file=sys.stderr)
        return 1

    languages = [lang.strip() for lang in args.lang.split(",") if lang.strip()]

    print(f"Lade Transkript für Video {video_id}...", file=sys.stderr)
    try:
        transcript, lang_code = fetch_transcript(video_id, languages)
    except (TranscriptsDisabled, NoTranscriptFound):
        print(
            f"Fehler: Für Video {video_id} ist kein Transkript verfügbar.",
            file=sys.stderr,
        )
        return 1
    except VideoUnavailable:
        print(f"Fehler: Video {video_id} ist nicht verfügbar.", file=sys.stderr)
        return 1

    print(
        f"Transkript geladen ({lang_code}, {len(transcript)} Zeichen). "
        f"Erstelle Zusammenfassung...\n",
        file=sys.stderr,
    )

    summarize(transcript, video_id, lang_code)
    return 0


if __name__ == "__main__":
    sys.exit(main())
