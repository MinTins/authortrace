"""
Інтерактивний веб-інтерфейс системи AuthorTrace.

Можливості:
  * галерея курованих прикладів для швидкого тестування (категорії
    «Людина», «Штучний текст», крайові випадки);
  * аналіз довільного тексту з обчисленням імовірності штучного
    походження;
  * візуалізація внеску кожної з трьох груп ознак у вердикт;
  * посегментний аналіз з підсвічуванням підозрілих фрагментів;
  * налаштування порогу класифікації в бічній панелі.

Запуск з кореня репозиторію:
    streamlit run app/streamlit_app.py
"""

import json
import os
import sys

import yaml
import streamlit as st

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.detector import AuthorTraceDetector

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EXAMPLES_PATH = os.path.join(os.path.dirname(__file__), "examples.json")

CATEGORY_LABELS = {
    "human": "Людина",
    "ai": "Штучний текст",
    "edge": "Крайові випадки",
}


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

def render_sidebar(metrics, threshold):
    with st.sidebar:
        st.markdown("### Налаштування")
        threshold = st.slider(
            "Поріг класифікації",
            min_value=0.10, max_value=0.90, value=threshold, step=0.05,
            help="Імовірність штучного походження, починаючи з якої текст "
                 "позначається як 'AI'. Стандартне значення — 0.5.",
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
                   "Тексти іншими мовами оброблятимуться, але якість "
                   "може бути нижчою.")
    return threshold


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


def render_results(result, threshold):
    """Виводить результат аналізу з візуалізацією внесків і сегментів."""
    prob = result["ai_probability"]
    is_ai = prob >= threshold

    # Великий блок вердикту.
    verdict_label = "Штучно згенерований текст" if is_ai else "Текст, написаний людиною"
    st.markdown("### Результат аналізу")
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

    # Посегментний аналіз.
    st.markdown("#### Посегментний аналіз")
    st.caption("Кожен фрагмент тексту класифіковано окремо. Це дозволяє "
               "виявити частково згенеровані тексти.")
    if not result["segments"]:
        st.info("Текст занадто короткий для посегментного аналізу.")
    for i, seg in enumerate(result["segments"], 1):
        seg_prob = seg["ai_probability"]
        seg_is_ai = seg_prob >= threshold
        tag_color = "#c62828" if seg_is_ai else "#2e7d32"
        tag_text = "ШІ" if seg_is_ai else "Людина"
        st.markdown(
            f"<div style='border-left:3px solid {tag_color}; padding:8px 12px; "
            f"margin:6px 0; background:#fafafa;'>"
            f"<span style='display:inline-block; min-width:60px; "
            f"font-weight:600; color:{tag_color};'>[{i}] {tag_text}</span>"
            f"<span style='color:#666;'>P(ШІ) = {seg_prob:.3f}</span><br>"
            f"<span style='font-size:0.92em;'>{seg['text']}</span>"
            f"</div>", unsafe_allow_html=True)


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

    metrics = load_metrics()
    threshold = render_sidebar(metrics, st.session_state["threshold"])
    st.session_state["threshold"] = threshold

    st.title("AuthorTrace")
    st.markdown(
        "Гібридна нейромережева система детекції штучно згенерованих текстів. "
        "Поєднує стилометричні, перплексійні та семантичні ознаки в одній "
        "моделі та пояснює власне рішення."
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
            "Вставте власний текст англійською мовою. Для надійної "
            "класифікації бажано не менше 50 слів."
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
            with st.spinner("Виконується аналіз... "
                            "(перший запуск може зайняти 10-15 секунд "
                            "через ініціалізацію мовної моделі)"):
                detector, _ = load_detector()
                result = detector.analyze(text, threshold=threshold)
            render_results(result, threshold)


if __name__ == "__main__":
    main()
