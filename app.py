import base64
import io

import numpy as np
import streamlit as st
from openai import OpenAI
from PIL import Image
from streamlit_drawable_canvas import st_canvas


def image_to_base64(image: Image.Image) -> str:
    """Convierte una imagen PIL a base64 sin guardarla en disco."""
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def canvas_is_empty(image_data) -> bool:
    """Revisa si el usuario realmente dibujó algo (el fondo es blanco)."""
    rgb = image_data[:, :, :3]
    return np.all(rgb >= 250)


# ---------------- Configuración de la página ----------------
st.set_page_config(page_title="Cuentacuentos Inteligente")
st.title("✏️ Cuentacuentos Inteligente")

with st.sidebar:
    st.subheader("Acerca de:")
    st.write(
        "Dibuja lo que quieras y la inteligencia artificial creará "
        "una historia original inspirada en tu dibujo."
    )
    stroke_width = st.slider("Ancho de línea", 1, 30, 5)
    stroke_color = st.color_picker("Color del trazo", "#000000")

    st.subheader("Opciones de la historia")
    genero = st.selectbox(
        "Género",
        ["Aventura", "Fantasía", "Comedia", "Misterio", "Ciencia ficción", "Fábula con moraleja"],
    )
    publico = st.selectbox("Público", ["Niños", "Jóvenes", "Adultos"])
    extension = st.select_slider("Extensión", ["Corta", "Media", "Larga"], value="Media")

st.subheader("Dibuja algo en el panel y presiona el botón para crear tu historia")

# ---------------- Lienzo ----------------
canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)",
    stroke_width=stroke_width,
    stroke_color=stroke_color,
    background_color="#FFFFFF",
    height=300,
    width=400,
    drawing_mode="freedraw",
    key="canvas",
)

api_key = st.text_input("Ingresa tu clave de OpenAI", type="password")

crear_button = st.button("✨ Crear historia", type="primary")

# ---------------- Lógica principal ----------------
if crear_button:
    if not api_key:
        st.warning("Por favor ingresa tu API key.")
    elif canvas_result.image_data is None or canvas_is_empty(canvas_result.image_data):
        st.warning("Primero dibuja algo en el lienzo 🙂")
    else:
        with st.spinner("Inventando tu historia..."):
            # Convertir el dibujo a imagen RGB con fondo blanco
            rgba = Image.fromarray(canvas_result.image_data.astype("uint8"), "RGBA")
            fondo = Image.new("RGB", rgba.size, (255, 255, 255))
            fondo.paste(rgba, mask=rgba.split()[3])
            base64_image = image_to_base64(fondo)

            palabras = {"Corta": "150", "Media": "300", "Larga": "500"}[extension]

            prompt_text = (
                "Observa este dibujo hecho a mano. Primero identifica qué elementos "
                "aparecen (personajes, objetos, lugares). Luego escribe en español una "
                f"historia original de género {genero.lower()} para un público de "
                f"{publico.lower()}, de aproximadamente {palabras} palabras, en la que "
                "esos elementos sean protagonistas.\n\n"
                "Formato de respuesta:\n"
                "**Lo que veo en tu dibujo:** (una frase breve)\n\n"
                "### (Título creativo de la historia)\n\n"
                "(La historia, con inicio, nudo y desenlace)"
            )

            try:
                client = OpenAI(api_key=api_key)
                message_placeholder = st.empty()
                full_response = ""

                stream = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {
                            "role": "system",
                            "content": "Eres un cuentacuentos creativo y cálido que "
                            "transforma dibujos sencillos en historias memorables.",
                        },
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt_text},
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                                },
                            ],
                        },
                    ],
                    max_tokens=1200,
                    temperature=0.9,
                    stream=True,
                )

                # Mostrar la historia mientras se escribe
                for chunk in stream:
                    delta = chunk.choices[0].delta.content if chunk.choices else None
                    if delta:
                        full_response += delta
                        message_placeholder.markdown(full_response + "▌")

                message_placeholder.markdown(full_response)
                st.session_state.mi_historia = full_response

                st.download_button(
                    "📥 Descargar historia",
                    data=full_response,
                    file_name="mi_historia.txt",
                    mime="text/plain",
                )

            except Exception as e:
                st.error(f"Ocurrió un error: {e}")
