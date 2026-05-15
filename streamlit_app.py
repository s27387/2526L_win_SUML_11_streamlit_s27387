import streamlit as st

st.set_page_config(
    page_title="Właściwy tytuł strony",
    layout="centered",
)
st.title("Nagłówek aplikacji")
st.image(
    "https://www.webopedia.com/wp-content/uploads/2010/07/what-is-a-computer-cover-scaled.webp",
    use_container_width=True
)
with st.expander("O aplikacji – kliknij, aby rozwinąć", expanded=True):
    st.markdown(
        """
        Funkcje:

        1.  Wydźwięk emocjonalny: Analiza sentymentu tekstu (ang.) pozytywny / negatywny.
        2.  Tłumaczenie EN → DE: Tłumaczenie tekstu z angielskiego na niemiecki.
 
        Instrukcja obsługi:
        1. Wybierz opcję z listy poniżej.
        2. Wpisz tekst w podanym języku.
        3. Kliknij przycisk i poczekaj na wynik.
        """
    )

st.header('Przetwarzanie języka naturalnego')

import streamlit as st
from transformers import MarianMTModel, MarianTokenizer, pipeline

option = st.selectbox(
    "Opcje",
    [
        "Wydźwięk emocjonalny tekstu (eng)",
        "Tłumaczenie tekstu: angielski → niemiecki",
    ],
)

if option == "Wydźwięk emocjonalny tekstu (eng)":
    text = st.text_area(label="Wpisz tekst")
    if text:
        classifier = pipeline("sentiment-analysis")
        answer = classifier(text)
        st.write(answer)
elif option == "Tłumaczenie tekstu: angielski → niemiecki":
    text = st.text_area(label="Wpisz tekst")
    if text:
       with st.spinner("Tłumaczę..."):
            tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-de")
            model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-de")
            inputs = tokenizer(text, return_tensors="pt", padding=True)
            translated = model.generate(**inputs)
            answer = tokenizer.decode(translated[0], skip_special_tokens=True)
            st.success(answer)
st.write("s27387")