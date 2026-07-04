
import io
import csv

import streamlit as st
from PIL import Image

from caption_engine import CaptionEngine, STYLE_PROMPTS


st.set_page_config(
    page_title="AI Image Caption Generator",
    page_icon="🖼️",
    layout="wide",
)

st.title("🖼️ AI Image Caption Generator")
st.caption("Upload one or more images and generate captions using BLIP (Salesforce/blip-image-captioning-base).")



@st.cache_resource(show_spinner="Loading BLIP model (first run only)...")
def load_engine(model_name: str) -> CaptionEngine:
    return CaptionEngine(model_name=model_name)



with st.sidebar:
    st.header("Settings")

    model_choice = st.selectbox(
        "Model size",
        options=[
            "Salesforce/blip-image-captioning-base",
            "Salesforce/blip-image-captioning-large",
        ],
        help="Base is faster and lighter; Large is slower but often more accurate.",
    )

    selected_styles = st.multiselect(
        "Caption styles to generate",
        options=list(STYLE_PROMPTS.keys()),
        default=["Descriptive (default)"],
    )

    with st.expander("Advanced generation settings"):
        max_length = st.slider("Max caption length (tokens)", 10, 100, 50)
        num_beams = st.slider("Beam search width", 1, 8, 5, help="Higher = more thorough search, slower.")

    st.divider()
    st.caption("Model runs locally via Transformers. First load may take a minute while weights download.")


uploaded_files = st.file_uploader(
    "Drag & drop images here (or click to browse)",
    type=["png", "jpg", "jpeg", "webp"],
    accept_multiple_files=True,
)

if not selected_styles:
    st.warning("Select at least one caption style in the sidebar to continue.")
    st.stop()

if not uploaded_files:
    st.info("Upload one or more images to get started.")
    st.stop()

engine = load_engine(model_choice)

process_clicked = st.button(f"Generate captions for {len(uploaded_files)} image(s)", type="primary")


if process_clicked:
    all_rows = []  # for CSV export: [filename, style, caption]
    progress = st.progress(0.0, text="Starting...")

    for idx, file in enumerate(uploaded_files):
        image = Image.open(file).convert("RGB")

        progress.progress(idx / len(uploaded_files), text=f"Captioning {file.name}...")

        results = engine.generate_many_styles(
            image, selected_styles, max_length=max_length, num_beams=num_beams
        )

        st.divider()
        col_img, col_captions = st.columns([1, 2])

        with col_img:
            st.image(image, caption=file.name, use_container_width=True)

        with col_captions:
            for r in results:
                st.markdown(f"**{r.style}**")
                # st.code renders a built-in copy-to-clipboard icon — no custom JS needed
                st.code(r.caption, language=None)
                all_rows.append([file.name, r.style, r.caption])

        progress.progress((idx + 1) / len(uploaded_files))

    progress.empty()
    st.success(f"Done — generated captions for {len(uploaded_files)} image(s).")


    csv_buffer = io.StringIO()
    writer = csv.writer(csv_buffer)
    writer.writerow(["filename", "style", "caption"])
    writer.writerows(all_rows)

    st.download_button(
        "⬇️ Download all captions as CSV",
        data=csv_buffer.getvalue(),
        file_name="captions.csv",
        mime="text/csv",
    )
