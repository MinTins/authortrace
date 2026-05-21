"""
Інтерактивний веб-інтерфейс системи AuthorTrace.

Можливості:
  * галерея курованих прикладів для швидкого тестування (категорії
    «Людина», «Штучний текст», крайові випадки);
  * аналіз довільного тексту з обчисленням імовірності штучного
    походження;
  * автоматичне визначення мови та переклад україномовних текстів
    на англійську (модель навчалася на англомовному корпусі);
  * візуалізація внеску кожної з трьох груп ознак у вердикт;
  * посегментний аналіз з інлайн-виділенням підозрілих фрагментів у тексті
    та кольоровим градієнтом ймовірності;
  * налаштування порогу класифікації в бічній панелі.

Запуск з кореня репозиторію:
    streamlit run app/streamlit_app.py
"""

import html
import json
import os
import sys

import yaml
import streamlit as st

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.detector import AuthorTraceDetector
from src.translate import detect_language

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EXAMPLES_PATH = os.path.join(os.path.dirname(__file__), "examples.json")

CATEGORY_LABELS = {
    "human": "Людина",
    "ai": "Штучний текст",
    "edge": "Крайові випадки",
}

LANG_NAMES = {"uk": "українська", "en": "англійська"}


# --- Завантаження ресурсів -------------------------------------------------

@st.cache_resource
def load_detector():
    cfg = yaml.safe_load(open(os.path.join(ROOT, "config.yaml"), encoding="utf-8"))
    return AuthorTraceDetector(
        model_path=os.path.join(ROOT, cfg["paths"]["model"]),
        scaler_path=os.path.join(ROOT, cfg["paths"]["scaler"]),
        lm_name=cfg["language_model"]["name"],
        max_tokens=cfg["language_model"]["max_tokens"],
        window_size=cfg["language_model"]["window_size"],
        top_k=cfg["language_model"]["top_k"],
        mcfg=cfg["model"],
    ), cfg


@st.cache_data
def load_examples():
    return json.load(open(EXAMPLES_PATH, encoding="utf-8"))


@st.cache_data
def load_metrics():
    path = os.path.join(ROOT, "results", "metrics.json")
    if os.path.exists(path):
        return json.load(open(path, encoding="utf-8"))
    return None


# --- Інтерфейсні допоміжні функції -----------------------------------------

def render_sidebar(metrics, threshold, lang_mode):
    with st.sidebar:
        st.markdown("### Налаштування")
        threshold = st.slider(
            "Поріг класифікації",
            min_value=0.10, max_value=0.90, value=threshold, step=0.05,
            help="Імовірність штучного походження, починаючи з якої текст "
                 "позначається як 'AI'. Стандартне значення — 0.5.",
        )

        st.markdown("#### Мова вхідного тексту")
        lang_mode = st.radio(
            "Як обробляти текст",
            options=["auto", "en", "translate"],
            format_func=lambda k: {
                "auto": "Авто (визначити мову)",
                "en": "Англійська (без перекладу)",
                "translate": "Українська → переклад → аналіз",
            }[k],
            index=["auto", "en", "translate"].index(lang_mode),
            help=(
                "Модель навчалася на англомовних текстах. У режимі «Авто» "
                "система визначає мову автоматично та перекладає "
                "україномовні тексти. Переклад виконується безкоштовно "
                "(deep-translator + Google Translate)."
            ),
        )

        st.markdown("---")
        st.markdown("### Про модель")
        st.markdown(
            "**AuthorTrace** — гібридна нейромережа, що поєднує три групи "
            "ознак:\n"
            "* стилометричні (15 ознак);\n"
            "* перплексійні (8 ознак, `distilgpt2`);\n"
            "* семантичні (768, прихований стан `distilgpt2`)."
        )

        if metrics is not None:
            st.markdown("### Якість на тесті")
            full = metrics["full_model"]
            c1, c2 = st.columns(2)
            c1.metric("Точність", f"{full['accuracy']:.3f}")
            c2.metric("F1", f"{full['f1']:.3f}")
            c1.metric("ROC AUC", f"{full['roc_auc']:.3f}")
            c2.metric("Розмір тесту", f"{metrics['n_test']}")
            st.caption(f"Корпус: {metrics['dataset_source']} "
                       f"({metrics['n_train']} тренувальних текстів)")

        st.markdown("---")
        st.caption("Модель навчалася на англомовному корпусі HC3. "
                   "Україномовні тексти автоматично перекладаються "
                   "англійською — результати лишаються інтерпретованими, "
                   "але можуть бути менш точними, ніж для оригінально "
                   "англомовних текстів.")
    return threshold, lang_mode


def render_example_gallery(examples, active_filter):
    """Виводить картки прикладів з кнопками завантаження."""
    filtered = [
        e for e in examples
        if active_filter == "all" or e["category"] == active_filter
    ]
    cols = st.columns(2)
    for i, ex in enumerate(filtered):
        with cols[i % 2]:
            with st.container(border=True):
                badge = CATEGORY_LABELS[ex["category"]]
                st.markdown(f"**{ex['title']}**")
                st.caption(f"Категорія: *{badge}* &nbsp;·&nbsp; "
                           f"Очікувано: *{ex['expected']}*")
                preview = ex["text"][:170].replace("\n", " ")
                st.markdown(
                    f"<div style='font-size:0.86em; color:#444;'>"
                    f"{preview}{'...' if len(ex['text']) > 170 else ''}"
                    f"</div>", unsafe_allow_html=True)
                with st.expander("Чому цей приклад?"):
                    st.markdown(ex["description"])
                if st.button("Завантажити цей приклад",
                             key=f"load_{ex['id']}", use_container_width=True):
                    st.session_state["input_text"] = ex["text"]
                    st.session_state["loaded_id"] = ex["id"]
                    st.rerun()


def render_translation_note(trans_info):
    """Показує плашку про виконаний переклад / визначену мову."""
    src = trans_info.get("source_language", "en")
    if trans_info.get("translated"):
        st.info(
            f"Виявлено мову: **{LANG_NAMES.get(src, src)}**. "
            f"Текст перекладено англійською для аналізу. "
            f"Усі ймовірності та сегменти нижче стосуються перекладу."
        )
        with st.expander("Показати переклад, який було проаналізовано"):
            st.code(trans_info["translated_text"], language=None)
    elif src != "en":
        st.warning(
            f"Виявлено мову **{LANG_NAMES.get(src, src)}**, але переклад "
            f"вимкнено. Результат може бути ненадійним — модель навчалася "
            f"на англомовних текстах. Увімкніть «Українська → переклад» "
            f"у бічній панелі."
        )


def render_results(result, threshold):
    """Виводить результат аналізу з візуалізацією внесків і сегментів."""
    prob = result["ai_probability"]
    is_ai = prob >= threshold

    # Великий блок вердикту.
    verdict_label = "Штучно згенерований текст" if is_ai else "Текст, написаний людиною"
    st.markdown("### Результат аналізу")

    render_translation_note(result.get("translation") or {})

    box_color = "#fbe9e7" if is_ai else "#e8f5e9"
    border_color = "#c62828" if is_ai else "#2e7d32"
    st.markdown(
        f"<div style='background:{box_color}; border-left:6px solid "
        f"{border_color}; padding:14px 18px; border-radius:6px;'>"
        f"<div style='font-size:0.9em; color:#555;'>Вердикт моделі</div>"
        f"<div style='font-size:1.5em; font-weight:600; color:{border_color};'>"
        f"{verdict_label}</div></div>",
        unsafe_allow_html=True,
    )
    st.write("")

    # Метрики ймовірності.
    c1, c2, c3 = st.columns(3)
    c1.metric("Імовірність ШІ", f"{prob * 100:.1f}%")
    c2.metric("Впевненість",
              f"{result['confidence'] * 100:.1f}%",
              help="Близькість імовірності до однієї з крайніх точок (0 або 1).")
    c3.metric("Поріг", f"{threshold:.2f}")
    st.progress(prob)

    # Внесок груп ознак.
    st.markdown("#### Внесок груп ознак у вердикт")
    st.caption("Обчислено методом абляції гілок мережі: показує, наскільки "
               "кожна група ознак вплинула на остаточне рішення.")
    names = {"stylometric": "Стилометрія",
             "perplexity": "Перплексія",
             "semantic": "Семантика"}
    contribs = sorted(result["feature_contributions"].items(),
                      key=lambda x: -x[1])
    for grp, share in contribs:
        col_a, col_b = st.columns([1, 4])
        col_a.markdown(f"**{names[grp]}**")
        col_b.progress(share / 100.0, text=f"{share:.1f}%")

    # Посегментний аналіз: підсвічування підозрілих фрагментів у тексті.
    st.markdown("#### Посегментний аналіз")
    st.caption("Кожен фрагмент тексту класифіковано окремо. Колір та "
               "числовий бейдж показують ймовірність штучного походження "
               "саме цього фрагмента. Для дуже довгих текстів сегменти "
               "укрупнюються адаптивно.")
    render_segment_highlights(result["segments"], threshold)


def _segment_palette(prob, threshold):
    """
    Повертає кольори для виділення сегмента залежно від його ймовірності.

    Гладкий HSL-градієнт зелений → жовтий → червоний від 0 до 1; межа
    переходу прив'язана до порогу класифікації, щоб візуальний акцент
    збігався з логікою вердикту.
    """
    # Прив'язуємо «нейтральну точку» (жовтий) до порогу — нижче порогу
    # перевага зелених відтінків, вище — червоних.
    if prob <= threshold:
        # Лінійна інтерполяція 120° (зелений) -> 60° (жовтий)
        ratio = prob / threshold if threshold > 0 else 0.0
        hue = 120 - ratio * 60
    else:
        # Лінійна інтерполяція 60° (жовтий) -> 0° (червоний)
        ratio = (prob - threshold) / max(1.0 - threshold, 1e-6)
        hue = 60 - ratio * 60

    # Світлий фон для читабельності, насиченіший бейдж.
    bg = f"hsl({hue:.0f}, 70%, 88%)"
    badge_bg = f"hsl({hue:.0f}, 60%, 38%)"
    border = f"hsl({hue:.0f}, 55%, 55%)"
    return bg, badge_bg, border


def render_segment_highlights(segments, threshold):
    """Виводить текст з кольоровим виділенням сегментів та бейджами P(ШІ)."""
    if not segments:
        st.info("Текст занадто короткий для посегментного аналізу.")
        return

    # Легенда градієнта.
    legend_steps = [0.0, 0.25, 0.5, 0.75, 1.0]
    legend_html = "".join(
        f"<span style='background:{_segment_palette(p, threshold)[0]}; "
        f"padding:3px 10px; margin:0 1px; font-size:0.78em; color:#333;'>"
        f"{p:.2f}</span>"
        for p in legend_steps
    )
    st.markdown(
        "<div style='font-size:0.82em; color:#555; margin:6px 0 4px;'>"
        "Шкала ймовірності штучного походження сегмента:"
        f"</div><div style='margin-bottom:10px;'>{legend_html} "
        f"<span style='font-size:0.78em; color:#777; margin-left:8px;'>"
        f"(поріг: {threshold:.2f})</span></div>",
        unsafe_allow_html=True,
    )

    # Текст з виділеними сегментами.
    pieces = []
    n_suspicious = 0
    for i, seg in enumerate(segments, 1):
        prob = seg["ai_probability"]
        bg, badge_bg, border = _segment_palette(prob, threshold)
        is_suspicious = prob >= threshold
        if is_suspicious:
            n_suspicious += 1
        text_safe = html.escape(seg["text"])
        badge = (
            f"<span style='background:{badge_bg}; color:white; "
            f"padding:1px 6px; border-radius:3px; margin-right:6px; "
            f"font-size:0.74em; font-weight:600; vertical-align:middle; "
            f"white-space:nowrap;' "
            f"title='Сегмент {i}: P(ШІ) = {prob:.3f}'>"
            f"{prob:.2f}</span>"
        )
        piece = (
            f"<span style='background:{bg}; padding:2px 4px; margin:2px 1px; "
            f"border-radius:3px; box-decoration-break:clone; "
            f"-webkit-box-decoration-break:clone;'>"
            f"{badge}{text_safe}</span>"
        )
        pieces.append(piece)

    container = (
        "<div style='line-height:2.0; padding:14px 16px; background:#fcfcfc; "
        "border:1px solid #e5e5e5; border-radius:6px; font-size:0.95em;'>"
        + " ".join(pieces) +
        "</div>"
    )
    st.markdown(container, unsafe_allow_html=True)

    # Зведення.
    n_total = len(segments)
    st.caption(
        f"Усього сегментів: **{n_total}**, "
        f"підозрілих (P ≥ {threshold:.2f}): **{n_suspicious}** "
        f"({n_suspicious / n_total * 100:.0f}%)."
    )

    # Числові деталі — за бажанням, у згорнутому блоці.
    with st.expander("Показати числові деталі по сегментах"):
        for i, seg in enumerate(segments, 1):
            mark = "ШІ" if seg["ai_probability"] >= threshold else "Людина"
            preview = seg["text"][:120] + ("..." if len(seg["text"]) > 120 else "")
            st.markdown(
                f"**[{i}]** `P(ШІ)={seg['ai_probability']:.3f}` — *{mark}*  \n"
                f"<span style='color:#555; font-size:0.9em;'>{html.escape(preview)}</span>",
                unsafe_allow_html=True,
            )


def _resolve_translate_flag(text, lang_mode):
    """
    Перетворює налаштування мови з бічної панелі на булевий прапор `translate`,
    що очікує детектор. Повертає (translate_flag, detected_language).
    """
    detected = detect_language(text) if text else "en"
    if lang_mode == "translate":
        return True, detected
    if lang_mode == "en":
        return False, detected
    # auto
    return (detected != "en"), detected


# --- Головна сторінка ------------------------------------------------------

def main():
    st.set_page_config(
        page_title="AuthorTrace — детектор штучних текстів",
        layout="wide", initial_sidebar_state="expanded",
    )

    if "input_text" not in st.session_state:
        st.session_state["input_text"] = ""
    if "loaded_id" not in st.session_state:
        st.session_state["loaded_id"] = None
    if "threshold" not in st.session_state:
        st.session_state["threshold"] = 0.5
    if "lang_mode" not in st.session_state:
        st.session_state["lang_mode"] = "auto"

    metrics = load_metrics()
    threshold, lang_mode = render_sidebar(
        metrics, st.session_state["threshold"], st.session_state["lang_mode"]
    )
    st.session_state["threshold"] = threshold
    st.session_state["lang_mode"] = lang_mode

    st.title("AuthorTrace")
    st.markdown(
        "Гібридна нейромережева система детекції штучно згенерованих текстів. "
        "Поєднує стилометричні, перплексійні та семантичні ознаки в одній "
        "моделі та пояснює власне рішення. Підтримує україномовні тексти "
        "через автоматичний переклад."
    )

    examples = load_examples()["examples"]

    tab_examples, tab_custom = st.tabs(
        ["📚 Тестові приклади", "✍️ Власний текст"]
    )

    with tab_examples:
        st.markdown(
            "Виберіть готовий приклад зі шкільної бібліотеки тестових "
            "текстів. Кожен супроводжується поясненням, чому він "
            "представляє той чи інший клас."
        )
        filter_labels = {
            "all": "Усі",
            "human": "Тільки людські",
            "ai": "Тільки штучні",
            "edge": "Крайові випадки",
        }
        filter_choice = st.radio(
            "Фільтр за категорією",
            options=list(filter_labels.keys()),
            format_func=lambda k: filter_labels[k],
            horizontal=True, index=0,
        )
        render_example_gallery(examples, filter_choice)

    with tab_custom:
        st.markdown(
            "Вставте власний текст. Англійську модель обробляє безпосередньо; "
            "україномовний текст буде перекладено англійською (налаштовується "
            "в бічній панелі). Для надійної класифікації бажано не менше "
            "50 слів."
        )
        st.session_state["input_text"] = st.text_area(
            "Текст для аналізу",
            value=st.session_state["input_text"],
            height=240,
            placeholder="Вставте текст сюди...",
            label_visibility="collapsed",
        )

    st.markdown("---")

    # Спільний блок аналізу.
    if st.session_state.get("loaded_id"):
        loaded_ex = next(e for e in examples
                         if e["id"] == st.session_state["loaded_id"])
        st.info(f"Завантажено приклад: **{loaded_ex['title']}**  "
                f"(очікуваний клас: *{loaded_ex['expected']}*)")

    col_a, col_b, col_c = st.columns([2, 1, 1])
    do_analyze = col_a.button("🔍 Проаналізувати текст",
                              type="primary", use_container_width=True)
    if col_b.button("🗑️ Очистити", use_container_width=True):
        st.session_state["input_text"] = ""
        st.session_state["loaded_id"] = None
        st.rerun()

    text = st.session_state["input_text"].strip()

    if do_analyze:
        if not text:
            st.warning("Введіть або завантажте текст для аналізу.")
        elif len(text.split()) < 10:
            st.warning(
                "Текст занадто короткий для надійної класифікації "
                "(потрібно щонайменше 10 слів)."
            )
        else:
            translate_flag, detected = _resolve_translate_flag(text, lang_mode)
            spinner_msg = "Виконується аналіз..."
            if translate_flag:
                spinner_msg = ("Перекладаю текст англійською та виконую "
                               "аналіз... (для довгих текстів може зайняти "
                               "до хвилини)")
            with st.spinner(spinner_msg):
                detector, _ = load_detector()
                try:
                    result = detector.analyze(
                        text, threshold=threshold, translate=translate_flag
                    )
                except RuntimeError as e:
                    # Найімовірніше — не встановлено deep-translator
                    # або немає інтернету.
                    st.error(
                        f"Не вдалося виконати переклад: {e}\n\n"
                        f"Спробуйте перемкнути режим мови на «Англійська» "
                        f"в бічній панелі та проаналізувати оригінальний "
                        f"текст без перекладу."
                    )
                    return
            render_results(result, threshold)


if __name__ == "__main__":
    main()
