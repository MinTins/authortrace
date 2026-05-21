"""
Модуль перекладу неангломовних текстів на англійську.

Призначення
-----------
Модель AuthorTrace навчена на англомовному корпусі HC3, тому її поведінка
для текстів іншими мовами (зокрема українською) непередбачувана. Цей модуль
дозволяє «вирівняти» вхідні дані: тексти-не-англійською автоматично
перекладаються англійською, після чого детектор працює у звичному режимі.

Стратегія розпізнавання мови — навмисно проста: рахуємо співвідношення
кириличних і латинських літер. Цього достатньо для основного сценарію
використання (українська/російська → англійська) і не вимагає
додаткової мовної моделі.

Переклад виконується через `deep-translator` (безкоштовний публічний
ендпойнт Google Translate, без ключа API). Запити чанкуються, бо
ендпойнт обмежує довжину одного фрагмента (~5000 символів).

Кеш перекладів убезпечує від повторних запитів для тих самих текстів
у межах сесії.
"""

import re
from typing import List, Optional

# Кириличний діапазон, що покриває укр./рос. і близькоспоріднені мови.
_CYRILLIC_RE = re.compile(r"[А-Яа-яЁёІіЇїЄєҐґЎў]")
# Максимальна довжина одного запиту до перекладача (з запасом).
_MAX_CHUNK_CHARS = 4500


def detect_language(text: str) -> str:
    """
    Грубе визначення мови за символьним складом.

    :return: ISO-код мови — наразі підтримується 'uk' (кирилиця) та 'en'
             (усе інше). Для коротких / порожніх текстів повертає 'en'.
    """
    if not text or not text.strip():
        return "en"

    cyr = len(_CYRILLIC_RE.findall(text))
    latin = sum(1 for ch in text if "a" <= ch.lower() <= "z")
    total = cyr + latin
    if total == 0:
        return "en"

    return "uk" if (cyr / total) > 0.30 else "en"


def _split_for_translation(text: str, max_len: int = _MAX_CHUNK_CHARS) -> List[str]:
    """
    Поділ тексту на фрагменти, придатні для надсилання у перекладач.

    Спочатку пробуємо різати по порожніх рядках (абзаци). Якщо абзац сам
    надто довгий — додатково ріжемо по реченнях. Якщо й речення занадто
    довге (рідкісний випадок) — ріжемо за словами.
    """
    if len(text) <= max_len:
        return [text]

    def by_sentences(block: str) -> List[str]:
        # Ріжемо за межами речень, але зберігаємо самі знаки.
        sentences = re.split(r"(?<=[.!?])\s+", block)
        out: List[str] = []
        buf = ""
        for s in sentences:
            if not s:
                continue
            if len(s) > max_len:
                # Розбити дуже довге «речення» за словами.
                words = s.split()
                cur = ""
                for w in words:
                    candidate = (cur + " " + w).strip() if cur else w
                    if len(candidate) > max_len:
                        if cur:
                            out.append(cur)
                        cur = w
                    else:
                        cur = candidate
                if cur:
                    out.append(cur)
                continue
            candidate = (buf + " " + s).strip() if buf else s
            if len(candidate) > max_len:
                if buf:
                    out.append(buf)
                buf = s
            else:
                buf = candidate
        if buf:
            out.append(buf)
        return out

    chunks: List[str] = []
    buf = ""
    for paragraph in text.split("\n\n"):
        if not paragraph.strip():
            continue
        candidate = (buf + "\n\n" + paragraph) if buf else paragraph
        if len(candidate) <= max_len:
            buf = candidate
            continue
        # Не вміщається — спершу скидаємо буфер.
        if buf:
            chunks.append(buf)
            buf = ""
        if len(paragraph) <= max_len:
            buf = paragraph
        else:
            chunks.extend(by_sentences(paragraph))
    if buf:
        chunks.append(buf)
    return chunks


class Translator:
    """
    Безкоштовний перекладач на основі `deep-translator` (Google Translate).

    Усі публічні методи безпечні до викликів для англомовних текстів —
    у такому разі переклад не виконується. Для довгих текстів запити
    розбиваються на чанки автоматично.
    """

    def __init__(self, target: str = "en"):
        self.target = target
        self._impl = None
        self._cache: dict = {}

    # --- Внутрішні методи -------------------------------------------------

    def _get_impl(self):
        """Лінива ініціалізація бекенду (щоб імпорт пакета був опціональним)."""
        if self._impl is None:
            try:
                from deep_translator import GoogleTranslator
            except ImportError as exc:
                raise RuntimeError(
                    "Для перекладу потрібен пакет deep-translator. "
                    "Встановіть: pip install deep-translator"
                ) from exc
            # source='auto' — Google визначить мову сам; це надійніше,
            # ніж покладатись лише на нашу евристику.
            self._impl = GoogleTranslator(source="auto", target=self.target)
        return self._impl

    # --- Публічний інтерфейс ---------------------------------------------

    def needs_translation(self, text: str, source_lang: Optional[str] = None) -> bool:
        """
        Перевіряє, чи потрібен переклад. Якщо мова явно вказана —
        порівнюємо її з цільовою; інакше визначаємо автоматично.
        """
        if not text or not text.strip():
            return False
        lang = source_lang or detect_language(text)
        return lang != self.target

    def translate(self, text: str) -> str:
        """
        Перекладає текст на `self.target`. Англомовний текст повертає
        без змін. Довгі тексти розбиваються на фрагменти автоматично.
        """
        if not text or not text.strip():
            return text
        if not self.needs_translation(text):
            return text

        if text in self._cache:
            return self._cache[text]

        impl = self._get_impl()
        chunks = _split_for_translation(text)
        translated_chunks: List[str] = []
        for chunk in chunks:
            if not chunk.strip():
                continue
            try:
                translated_chunks.append(impl.translate(chunk) or "")
            except Exception:
                # Перекладач помилився на конкретному фрагменті —
                # повертаємо оригінал цього шматка, щоб не втратити
                # сенс усього тексту.
                translated_chunks.append(chunk)
        result = "\n\n".join(c for c in translated_chunks if c)
        self._cache[text] = result
        return result
