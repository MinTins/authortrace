"""
Модуль перекладу неангломовних текстів на англійську.

Базова модель AuthorTrace навчена на англомовному корпусі HC3, тому
для текстів іншими мовами передбачено автоматичний переклад на цільову
мову моделі — англійську. Це дозволяє коректно обробляти україномовні
(а також інші неангломовні) тексти в єдиному конвеєрі.

Стратегія розпізнавання мови — проста й прозора: обчислюється
співвідношення кириличних і латинських літер. Цього достатньо для
основного сценарію використання (українська/російська → англійська)
і не вимагає окремої мовної моделі.

Переклад виконується через `deep-translator` (публічний ендпойнт
Google Translate, не потребує ключа API). Довгі тексти автоматично
розбиваються на фрагменти. На випадок мережевих збоїв реалізовано
кілька рівнів захисту: повтори запитів, рекурсивний поділ невдалих
фрагментів на коротші, та явна помилка у разі тотальної відмови
бекенду. Кеш перекладів запобігає повторним запитам для однакових
текстів у межах сесії.
"""

import re
import time
from typing import Callable, List, Optional

_CYRILLIC_RE = re.compile(r"[А-Яа-яЁёІіЇїЄєҐґЎў]")
# Максимальна довжина одного запиту до перекладача. Менше за номінальні
# 5000 символів — щоб мати запас на службові символи та зменшити частоту
# таймаутів на нестабільних з'єднаннях.
_MAX_CHUNK_CHARS = 3500
# Мінімальний розмір при рекурсивному поділі — нижче нього подальше
# дроблення безглузде.
_MIN_CHUNK_CHARS = 200
# Кількість спроб для одного запиту, з прогресивною паузою між ними.
_MAX_RETRIES = 3


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

    Спочатку пробуємо різати по порожніх рядках (абзаци). Якщо абзац
    сам надто довгий — додатково ріжемо по одинарних переносах рядка,
    далі — по реченнях. Якщо й речення задовге — за словами.
    """
    if len(text) <= max_len:
        return [text]

    def by_words(block: str) -> List[str]:
        words = block.split()
        out: List[str] = []
        cur = ""
        for w in words:
            candidate = (cur + " " + w).strip() if cur else w
            if len(candidate) > max_len and cur:
                out.append(cur)
                cur = w
            else:
                cur = candidate
        if cur:
            out.append(cur)
        return out

    def by_sentences(block: str) -> List[str]:
        sentences = re.split(r"(?<=[.!?])\s+", block)
        out: List[str] = []
        buf = ""
        for s in sentences:
            if not s:
                continue
            if len(s) > max_len:
                if buf:
                    out.append(buf)
                    buf = ""
                out.extend(by_words(s))
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

    def by_lines(block: str) -> List[str]:
        """Поділ за одинарними переносами, якщо немає \\n\\n-розділювачів."""
        lines = block.split("\n")
        out: List[str] = []
        buf = ""
        for line in lines:
            if not line.strip():
                continue
            if len(line) > max_len:
                if buf:
                    out.append(buf)
                    buf = ""
                out.extend(by_sentences(line))
                continue
            candidate = (buf + "\n" + line) if buf else line
            if len(candidate) > max_len:
                if buf:
                    out.append(buf)
                buf = line
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
        if buf:
            chunks.append(buf)
            buf = ""
        if len(paragraph) <= max_len:
            buf = paragraph
        else:
            chunks.extend(by_lines(paragraph))
    if buf:
        chunks.append(buf)
    return chunks


class TranslationError(RuntimeError):
    """Помилка перекладу, що вказує на неможливість завершити операцію."""


class Translator:
    """
    Перекладач на базі `deep-translator` (Google Translate).

    Англомовний текст повертає без виклику бекенду. Для довгих текстів
    виконує автоматичне розбиття на фрагменти, повтори запитів та
    рекурсивний поділ при невдачах.
    """

    def __init__(self, target: str = "en"):
        self.target = target
        self._impl = None
        self._cache: dict = {}

    # --- Внутрішні методи -------------------------------------------------

    def _get_impl(self):
        if self._impl is None:
            try:
                from deep_translator import GoogleTranslator
            except ImportError as exc:
                raise TranslationError(
                    "Для перекладу потрібен пакет deep-translator. "
                    "Встановіть: pip install deep-translator"
                ) from exc
            self._impl = GoogleTranslator(source="auto", target=self.target)
        return self._impl

    def _translate_chunk(self, chunk: str) -> Optional[str]:
        """
        Перекладає один фрагмент із повторами. Повертає рядок при успіху
        або None, якщо всі спроби закінчились невдачею.
        """
        impl = self._get_impl()
        last_error: Optional[Exception] = None
        for attempt in range(_MAX_RETRIES):
            try:
                result = impl.translate(chunk)
                if result and result.strip():
                    return result
            except Exception as exc:
                last_error = exc
            # Прогресивна пауза між повторами.
            time.sleep(0.5 * (attempt + 1))
        # Усі спроби невдалі — повертаємо None, щоб виклична сторона
        # могла спробувати рекурсивний поділ.
        if last_error is not None:
            # Зберігаємо інформацію про останню помилку у власному
            # атрибуті, щоб у разі тотального провалу побудувати
            # осмислене повідомлення.
            self._last_error = last_error
        return None

    def _translate_with_split(self, chunk: str) -> Optional[str]:
        """
        Рекурсивно ділить фрагмент навпіл при невдачі та повторює спроби.
        Це рятує ситуацію, коли довгий фрагмент стабільно фейлить (зокрема
        через ліміти ендпойнта), а коротший за нього — успішно
        перекладається.
        """
        result = self._translate_chunk(chunk)
        if result is not None:
            return result
        if len(chunk) <= _MIN_CHUNK_CHARS:
            return None

        # Намагаємось поділити фрагмент далі.
        target_len = max(_MIN_CHUNK_CHARS, len(chunk) // 2)
        sub_chunks = _split_for_translation(chunk, max_len=target_len)
        if len(sub_chunks) <= 1:
            # Поділ не зменшив розміру — припиняємо рекурсію.
            return None

        translated_parts: List[str] = []
        for sub in sub_chunks:
            sub_translated = self._translate_with_split(sub)
            if sub_translated is None:
                return None
            translated_parts.append(sub_translated)
        return " ".join(translated_parts)

    # --- Публічний інтерфейс ---------------------------------------------

    def needs_translation(self, text: str, source_lang: Optional[str] = None) -> bool:
        if not text or not text.strip():
            return False
        lang = source_lang or detect_language(text)
        return lang != self.target

    def translate(
        self,
        text: str,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> str:
        """
        Перекладає текст на цільову мову. Англомовний — повертає без змін.

        :param progress_callback: опційний callback `(done, total)` для
                                  відображення прогресу у користувацькому
                                  інтерфейсі під час перекладу довгих
                                  текстів.
        :raises TranslationError: якщо переклад не вдався після всіх
                                  спроб і рекурсивного поділу.
        """
        if not text or not text.strip():
            return text
        if not self.needs_translation(text):
            return text
        if text in self._cache:
            return self._cache[text]

        chunks = _split_for_translation(text)
        total = len(chunks)
        translated_chunks: List[str] = []
        failed: List[int] = []

        for i, chunk in enumerate(chunks):
            if not chunk.strip():
                continue
            translated = self._translate_with_split(chunk)
            if translated is None:
                failed.append(i)
                # Не вставляємо оригінал-кирилицю у результат — він
                # зіпсував би подальший аналіз. Замість цього просто
                # пропускаємо невдалий шматок і відмічаємо його.
            else:
                translated_chunks.append(translated)
            if progress_callback is not None:
                progress_callback(i + 1, total)

        # Якщо взагалі нічого не переклалось — кидаємо помилку.
        if not translated_chunks:
            last_err = getattr(self, "_last_error", None)
            detail = f" ({last_err})" if last_err else ""
            raise TranslationError(
                f"Не вдалося перекласти жодного фрагмента тексту"
                f" з {total}.{detail}"
            )

        # Якщо частина чанків випала — попереджаємо в результаті,
        # але повертаємо те, що змогли отримати. Викликаюча сторона
        # сама вирішить, чи цього достатньо.
        result = "\n\n".join(translated_chunks)
        self._cache[text] = result
        return result
