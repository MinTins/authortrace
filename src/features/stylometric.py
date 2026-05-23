"""
Модуль вилучення лінгвостатистичних та стилометричних ознак тексту.

Стилометричні ознаки — інтерпретована група сигналів, що обчислюється
без використання нейромережі. Гіпотеза: штучно згенерований текст має
характерні поверхневі властивості (рівномірність структури, специфічні
шаблони пунктуації, переважання абстрактного словника), які відрізняють
його від людського письма.

Модуль постачає 25 ознак, об'єднаних у п'ять змістових груп:

  Лексика та різноманітність:
    • type_token_ratio        — лексична різноманітність
    • hapax_ratio             — частка слів, що трапились один раз
    • bigram_uniqueness       — частка унікальних біграм
    • lexical_uniformity      — стабільність TTR уздовж тексту

  Структура речень і абзаців:
    • avg_word_length         — середня довжина слова
    • avg_sentence_length     — середня довжина речення
    • std_sentence_length     — розкид довжин речень
    • sentence_length_entropy — ентропія розподілу довжин
    • sentence_length_cv      — нормований коефіцієнт варіації
    • paragraph_balance       — варіація довжин абзаців
    • parallel_structure      — індекс синтаксичного паралелізму
    • long_word_ratio         — частка довгих слів
    • flesch_reading_ease     — індекс легкості читання

  Пунктуація та типографіка:
    • comma_ratio             — щільність ком
    • punctuation_ratio       — щільність розділових знаків
    • em_dash_ratio           — щільність типографських тире
    • comma_run_score         — частота тріплет-перерахувань «X, Y, Z»
    • digit_ratio             — частка цифрових символів
    • uppercase_ratio         — частка великих літер

  Дискурс і регістр:
    • function_word_ratio     — частка службових слів
    • discourse_marker_ratio  — частка дискурсивних маркерів
    • connector_entropy       — ентропія типів конекторів
    • hedge_density           — щільність «виваженої» мови
    • nominalization_ratio    — частка віддієслівних іменників
    • formal_register         — індекс формального регістру

ДВОМОВНА ПІДТРИМКА
------------------
Усі словники (конектори, hedges, formal register, nominalization-суфікси)
двомовні — українська та англійська. Це дозволяє обчислювати ознаки
безпосередньо на оригіналі тексту, не покладаючись на машинний переклад,
який вирівнює стилістичні відмінності.
"""

import re
import math
from collections import Counter

import numpy as np

# --- Базові словники (порядок впливає на масштабатор — не змінювати) -------

FUNCTION_WORDS = {
    "the", "a", "an", "and", "or", "but", "if", "of", "to", "in", "on", "at",
    "for", "with", "as", "by", "from", "that", "this", "these", "those", "it",
    "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
    "do", "does", "did", "not", "no", "so", "than", "then", "there", "here",
    "we", "you", "they", "he", "she", "i", "my", "your", "their", "its",
}

DISCOURSE_MARKERS = {
    "however", "therefore", "moreover", "furthermore", "additionally",
    "overall", "consequently", "thus", "hence", "indeed", "ultimately",
    "importantly", "notably", "specifically", "generally",
}

# --- Двомовні словники нових ознак ------------------------------------------

# Дискурсивні конектори (англ. + укр.) — використовуються для оцінки
# розмаїття типів конекторів. Один конектор у тексті — нейтрально,
# рівномірне використання багатьох типів — сигнал стилю LLM.
CONNECTORS_EN = {
    "however": "contrast", "but": "contrast", "yet": "contrast",
    "although": "contrast", "though": "contrast", "whereas": "contrast",
    "moreover": "addition", "furthermore": "addition",
    "additionally": "addition", "also": "addition", "besides": "addition",
    "therefore": "consequence", "thus": "consequence", "hence": "consequence",
    "consequently": "consequence", "accordingly": "consequence",
    "specifically": "example", "particularly": "example",
    "notably": "example", "namely": "example",
    "finally": "summary", "overall": "summary", "ultimately": "summary",
    "essentially": "summary", "generally": "summary",
}

CONNECTORS_UK = {
    "однак": "contrast", "проте": "contrast", "втім": "contrast",
    "натомість": "contrast", "хоча": "contrast", "тоді як": "contrast",
    "крім того": "addition", "більше того": "addition",
    "до того ж": "addition", "також": "addition",
    "отже": "consequence", "таким чином": "consequence",
    "тому": "consequence", "як наслідок": "consequence",
    "відтак": "consequence", "внаслідок": "consequence",
    "зокрема": "example", "наприклад": "example", "приміром": "example",
    "а саме": "example", "власне": "example",
    "загалом": "summary", "врешті-решт": "summary", "підсумовуючи": "summary",
    "узагальнюючи": "summary", "у підсумку": "summary",
}

# «Виважена» мова (hedging) — формулювання, що пом'якшують категоричність.
# Claude і GPT-4+ дуже часто їх використовують для уникнення безапеляційних
# тверджень. Людина в академічному тексті теж їх вживає, але рідше і
# нерівномірніше.
HEDGES_EN = {
    "may", "might", "could", "would", "should",
    "perhaps", "possibly", "likely", "presumably",
    "typically", "generally", "usually", "often", "frequently",
    "somewhat", "relatively", "rather", "fairly",
    "tend", "tends", "tendency", "suggests", "indicates", "appears",
    "seems", "considered", "approximately",
}

HEDGES_UK = {
    "може", "можуть", "ймовірно", "ймовірний", "можливо",
    "переважно", "зазвичай", "як правило", "типово",
    "відносно", "доволі", "досить", "достатньо",
    "схоже", "видається", "вважається", "припускається",
    "приблизно", "близько", "загалом", "в цілому",
    "тенденція", "тенденції", "тяжіє", "схильний",
}

# Слова-маркери формального регістру (двомовно).
FORMAL_REGISTER_EN = {
    "furthermore", "moreover", "consequently", "nonetheless",
    "notwithstanding", "henceforth", "thereby", "wherein",
    "utilize", "facilitate", "implement", "demonstrate",
    "constitute", "comprise", "encompass", "establish",
}

FORMAL_REGISTER_UK = {
    "здійснюється", "реалізується", "забезпечується", "виконується",
    "характеризується", "визначається", "відзначається",
    "становить", "являє", "охоплює", "передбачає",
    "застосовується", "використовується", "впроваджується",
    "ґрунтується", "базується", "полягає",
}

# Об'єднані словники — використовуються в підрахунку без розділення мов.
ALL_CONNECTORS = {**CONNECTORS_EN, **CONNECTORS_UK}
ALL_HEDGES = HEDGES_EN | HEDGES_UK
ALL_FORMAL = FORMAL_REGISTER_EN | FORMAL_REGISTER_UK

# --- Регулярні вирази -------------------------------------------------------

_WORD_RE = re.compile(r"[A-Za-zА-Яа-яЁёІіЇїЄєҐґ']+")
_SENT_RE = re.compile(r"[.!?]+")
_PARAGRAPH_RE = re.compile(r"\n\s*\n")
_VOWELS = "aeiouyаеєиіїоуюя"

# Тире — як типографське «—», так і «–»; звичайний дефіс не рахуємо,
# щоб не плутати з з'єднаннями типу «client-server».
_EM_DASH_RE = re.compile(r"[—–]")

# Шаблон «X, Y, Z» — послідовність кома-слово принаймні тричі поспіль.
# Свідчить про схильність до «тріплет-перерахувань», типову для LLM.
_COMMA_TRIPLET_RE = re.compile(
    r"[A-Za-zА-Яа-яЁёІіЇїЄєҐґ]+\s*,\s*"
    r"[A-Za-zА-Яа-яЁёІіЇїЄєҐґ]+\s*,\s*"
    r"[A-Za-zА-Яа-яЁёІіЇїЄєҐґ]+"
)

# Віддієслівні іменники — суфікси, що позначають дію/процес. Покривають
# обидві мови; повний словник тут не потрібен, бо ознака використовує
# щільність по суфіксах, що добре корелює з абстрактним академічним
# регістром, типовим для LLM.
_NOMINALIZATION_RE = re.compile(
    r"\b\w+("
    r"tion|sion|ment|ness|ity|ance|ence|ism|"
    r"ння|ття|ція|сія|ість|ення|ання|изм|ізм"
    r")\b",
    re.IGNORECASE,
)

# --- Назви ознак (порядок критичний!) ---------------------------------------

# --- Назви ознак (порядок критичний — він прошитий у scaler.json та
# у вхідному вимірі базової FusionMLP) --------------------------------------

# Базові ознаки — підмножина, на якій навчена фузійна нейромережа.
# Зміна порядку або складу зламає сумісність з збереженим масштабатором.
_BASE_STYLOMETRIC_NAMES = [
    "type_token_ratio",
    "hapax_ratio",
    "avg_word_length",
    "avg_sentence_length",
    "std_sentence_length",
    "sentence_length_entropy",
    "comma_ratio",
    "punctuation_ratio",
    "function_word_ratio",
    "discourse_marker_ratio",
    "bigram_uniqueness",
    "digit_ratio",
    "uppercase_ratio",
    "flesch_reading_ease",
    "long_word_ratio",
]

# Розширені ознаки — вхід калібратора. Не подаються до базової мережі.
EXTENDED_STYLOMETRIC_NAMES = [
    "sentence_length_cv",     # CV довжин речень (нормоване std/mean)
    "paragraph_balance",      # рівномірність довжин абзаців
    "connector_entropy",      # ентропія типів конекторів
    "hedge_density",          # щільність «виваженої» мови
    "nominalization_ratio",   # частка віддієслівних іменників
    "parallel_structure",     # індекс синтаксичного паралелізму
    "em_dash_ratio",          # щільність типографських тире
    "comma_run_score",        # частота тріплет-перерахувань
    "lexical_uniformity",     # рівномірність TTR за вікнами
    "formal_register",        # індекс формального регістру
]

STYLOMETRIC_NAMES = _BASE_STYLOMETRIC_NAMES + EXTENDED_STYLOMETRIC_NAMES
N_STYLOMETRIC_BASE = len(_BASE_STYLOMETRIC_NAMES)
N_STYLOMETRIC_EXTENDED = len(EXTENDED_STYLOMETRIC_NAMES)
N_STYLOMETRIC = len(STYLOMETRIC_NAMES)


# --- Допоміжні функції ------------------------------------------------------

def _safe_div(a, b):
    return a / b if b else 0.0


def _entropy(values):
    """Ентропія Шеннона для дискретного розподілу значень."""
    if not values:
        return 0.0
    counts = Counter(values)
    total = sum(counts.values())
    ent = 0.0
    for c in counts.values():
        p = c / total
        ent -= p * math.log(p + 1e-12, 2)
    return ent


def _count_syllables(word):
    """Грубе оцінювання кількості складів за групами голосних."""
    word = word.lower()
    groups = re.findall(r"[%s]+" % _VOWELS, word)
    return max(1, len(groups))


# --- Обчислення нових ознак -------------------------------------------------

def _sentence_length_cv(sent_lengths):
    """Коефіцієнт варіації довжин речень: std/mean.

    На відміну від абсолютного std (`std_sentence_length`), CV нормований
    за середньою довжиною і відображає відносний розкид. У LLM-текстах
    CV систематично нижчий: моделі схильні писати речення приблизно
    однакової довжини.
    """
    if len(sent_lengths) < 2:
        return 0.0
    mean = float(np.mean(sent_lengths))
    if mean <= 0:
        return 0.0
    std = float(np.std(sent_lengths))
    return std / mean


def _paragraph_balance(text):
    """Варіація довжин абзаців у словах.

    LLM-моделі тяжіють до абзаців приблизно однакової довжини, тоді як
    люди вільніше варіюють розмір. Якщо абзаців менше двох — повертаємо
    0.0 (нейтрально). Якщо текст не розбито на абзаци — ознака нейтральна
    і не штрафує жоден клас.
    """
    paragraphs = [p for p in _PARAGRAPH_RE.split(text) if p.strip()]
    if len(paragraphs) < 2:
        return 0.0
    lengths = [len(_WORD_RE.findall(p)) for p in paragraphs]
    lengths = [l for l in lengths if l > 0]
    if len(lengths) < 2:
        return 0.0
    mean = float(np.mean(lengths))
    if mean <= 0:
        return 0.0
    return float(np.std(lengths)) / mean


def _connector_entropy(words_lower):
    """Ентропія розподілу ТИПІВ конекторів (contrast/addition/...).

    Висока ентропія = рівномірне використання багатьох категорій конекторів
    у короткому тексті, що є маркером LLM-стилю. Люди зазвичай мають
    «улюблені» категорії і не балансують їх рівномірно.

    Якщо конекторів менше двох — повертаємо 0 (недостатньо сигналу).
    """
    types_used = []
    # Перевіряємо як окремі слова, так і двослівні конектори.
    text_str = " ".join(words_lower)
    for connector, ctype in ALL_CONNECTORS.items():
        if " " in connector:
            # Двослівні — шукаємо як підрядок з межами слова.
            pattern = r"\b" + re.escape(connector) + r"\b"
            count = len(re.findall(pattern, text_str))
            types_used.extend([ctype] * count)
        else:
            count = sum(1 for w in words_lower if w == connector)
            types_used.extend([ctype] * count)

    if len(types_used) < 2:
        return 0.0
    return _entropy(types_used)


def _hedge_density(words_lower, n_words):
    """Щільність «виваженої» мови — часток слів-хеджів від загального числа."""
    if n_words == 0:
        return 0.0
    count = sum(1 for w in words_lower if w in ALL_HEDGES)
    # Двослівні хеджі обробляємо окремо (наразі їх мало, але для
    # точності перевіряємо.
    # «як правило», «в цілому»…
    text_str = " ".join(words_lower)
    for hedge in ALL_HEDGES:
        if " " in hedge:
            count += len(re.findall(r"\b" + re.escape(hedge) + r"\b", text_str))
    return count / n_words


def _nominalization_ratio(text, n_words):
    """Частка віддієслівних іменників (за суфіксами).

    Висока щільність характерна для академічного/бюрократичного
    регістру, який LLM-моделі схильні застосовувати навіть тоді, коли
    задача цього не вимагає. Людина в неформальному тексті матиме дуже
    низьке значення; у формальному — помірне; LLM на тій самій темі —
    стабільно високе.
    """
    if n_words == 0:
        return 0.0
    matches = _NOMINALIZATION_RE.findall(text)
    return len(matches) / n_words


def _parallel_structure(sentences):
    """Індекс синтаксичного паралелізму між сусідніми реченнями.

    Спрощений детектор: для кожної пари сусідніх речень рахуємо частку
    спільних довжин слів на однакових позиціях (структурна схожість).
    LLM-моделі частіше будують речення-«близнюки» в межах абзацу;
    люди структурно різноманітніші.
    """
    if len(sentences) < 2:
        return 0.0

    scores = []
    for i in range(len(sentences) - 1):
        words_a = _WORD_RE.findall(sentences[i])
        words_b = _WORD_RE.findall(sentences[i + 1])
        if len(words_a) < 3 or len(words_b) < 3:
            continue
        # Порівнюємо довжини слів на спільних позиціях.
        n = min(len(words_a), len(words_b))
        matches = sum(
            1 for k in range(n) if len(words_a[k]) == len(words_b[k])
        )
        scores.append(matches / n)

    if not scores:
        return 0.0
    return float(np.mean(scores))


def _em_dash_ratio(text, n_words):
    """Щільність типографських тире (— і –) на слово.

    Claude і сучасні LLM активно використовують довге тире як
    стилістичний прийом; у природному людському тексті українською/
    англійською воно зазвичай зустрічається рідше і нерівномірніше.
    """
    if n_words == 0:
        return 0.0
    return len(_EM_DASH_RE.findall(text)) / n_words


def _comma_run_score(text, n_words):
    """Частота тріплет-перерахувань «X, Y, Z» на слово.

    «Тріплет-структура» (три однорідні елементи через кому) — улюблений
    риторичний прийом LLM. Норма для людського тексту в академічному
    стилі — 0.005–0.015; у LLM-тексті часто 0.025+.
    """
    if n_words == 0:
        return 0.0
    matches = _COMMA_TRIPLET_RE.findall(text)
    return len(matches) / n_words


def _lexical_uniformity(words_lower, window=50):
    """Рівномірність TTR за послідовними вікнами тексту.

    Розраховуємо TTR (type-token ratio) у непересічних вікнах фіксованого
    розміру і повертаємо інвертовану дисперсію: чим стабільніший TTR
    вздовж тексту, тим вищий показник. LLM зазвичай мають стабільну
    лексичну різноманітність, тоді як люди мають «вибухи» нової лексики.

    Значення в діапазоні [0, 1].
    """
    if len(words_lower) < 2 * window:
        return 0.0

    ttrs = []
    for i in range(0, len(words_lower) - window + 1, window):
        chunk = words_lower[i:i + window]
        ttrs.append(len(set(chunk)) / len(chunk))

    if len(ttrs) < 2:
        return 0.0

    # Інверсія дисперсії: 1 / (1 + var * 100). Множник 100 — щоб
    # значення не злипались біля 1; підібрано емпірично.
    var = float(np.var(ttrs))
    return 1.0 / (1.0 + var * 100.0)


def _formal_register(words_lower, n_words):
    """Індекс формального регістру — частка слів зі словника formal."""
    if n_words == 0:
        return 0.0
    count = sum(1 for w in words_lower if w in ALL_FORMAL)
    return count / n_words


# --- Основна функція вилучення ----------------------------------------------

def extract_stylometric(text):
    """Повертає numpy-вектор з 25 стилометричних ознак для одного тексту.

    Порядок ознак фіксований і визначається `STYLOMETRIC_NAMES`. Перші
    `N_STYLOMETRIC_BASE` (=15) ознак подаються до базової фузійної
    нейромережі, решта `N_STYLOMETRIC_EXTENDED` (=10) — до калібратора
    пост-обробки.

    :param text: вхідний рядок (укр./англ. або змішаний)
    :return: np.ndarray розмірності N_STYLOMETRIC (=25)
    """
    words = _WORD_RE.findall(text)
    words_lower = [w.lower() for w in words]
    n_words = len(words)
    n_chars = max(1, len(text))

    # Поділ на речення.
    sentences = [s.strip() for s in _SENT_RE.split(text) if s.strip()]
    sent_lengths = [len(_WORD_RE.findall(s)) for s in sentences]
    sent_lengths = [s for s in sent_lengths if s > 0]

    if n_words == 0:
        return np.zeros(N_STYLOMETRIC, dtype=np.float32)

    # === Базові ознаки (вхід фузійної мережі) ============================
    counts = Counter(words_lower)
    ttr = _safe_div(len(counts), n_words)
    hapax = _safe_div(sum(1 for c in counts.values() if c == 1), n_words)

    avg_word_len = _safe_div(sum(len(w) for w in words), n_words)
    avg_sent_len = float(np.mean(sent_lengths)) if sent_lengths else 0.0
    std_sent_len = float(np.std(sent_lengths)) if len(sent_lengths) > 1 else 0.0
    sent_entropy = _entropy(sent_lengths)

    comma_ratio = _safe_div(text.count(","), n_words)
    punct_ratio = _safe_div(sum(text.count(c) for c in ".,;:!?-"), n_words)

    fw_ratio = _safe_div(
        sum(1 for w in words_lower if w in FUNCTION_WORDS), n_words
    )
    dm_ratio = _safe_div(
        sum(1 for w in words_lower if w in DISCOURSE_MARKERS), n_words
    )

    bigrams = list(zip(words_lower, words_lower[1:]))
    bigram_uniq = (
        _safe_div(len(set(bigrams)), len(bigrams)) if bigrams else 0.0
    )

    digit_ratio = _safe_div(sum(c.isdigit() for c in text), n_chars)
    upper_ratio = _safe_div(sum(c.isupper() for c in text), n_chars)
    long_word_ratio = _safe_div(sum(1 for w in words if len(w) > 6), n_words)

    n_sent = max(1, len(sent_lengths))
    n_syll = sum(_count_syllables(w) for w in words)
    flesch = (
        206.835 - 1.015 * (n_words / n_sent) - 84.6 * (n_syll / n_words)
    )
    flesch = max(-50.0, min(120.0, flesch)) / 100.0

    # === Розширені ознаки (вхід калібратора) =============================
    sl_cv = _sentence_length_cv(sent_lengths)
    par_bal = _paragraph_balance(text)
    conn_ent = _connector_entropy(words_lower)
    hedge_dens = _hedge_density(words_lower, n_words)
    nomin = _nominalization_ratio(text, n_words)
    parallel = _parallel_structure(sentences)
    em_dash = _em_dash_ratio(text, n_words)
    comma_run = _comma_run_score(text, n_words)
    lex_unif = _lexical_uniformity(words_lower)
    formal = _formal_register(words_lower, n_words)

    return np.array([
        # Базові ознаки (індекси 0..14)
        ttr, hapax, avg_word_len, avg_sent_len, std_sent_len, sent_entropy,
        comma_ratio, punct_ratio, fw_ratio, dm_ratio, bigram_uniq,
        digit_ratio, upper_ratio, flesch, long_word_ratio,
        # Розширені ознаки (індекси 15..24)
        sl_cv, par_bal, conn_ent, hedge_dens, nomin,
        parallel, em_dash, comma_run, lex_unif, formal,
    ], dtype=np.float32)
