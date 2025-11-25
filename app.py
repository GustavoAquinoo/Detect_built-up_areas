import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import io
import time
import os


#ESTILO PERSONALIZADO (CSS)
st.markdown("""
<style>
/* Fondo de la app */
.main {
    background-color: #F7F9FC !important;
}

/* Títulos principales */
h1 {
    color: #1E88E5;
    font-weight: 700;
}

/* Subtítulos */
h2, h3 {
    color: #1565C0;
    font-weight: 600;
}

/* Contenedor de imágenes */
.image-container {
    padding: 15px;
    background-color: #FFFFFF;
    border-radius: 12px;
    border: 1px solid #e0e0e0;
    box-shadow: 0 1px 4px rgba(0,0,0,0.08);
}

/* Sidebar */
.sidebar .sidebar-content {
    background-color: #FFFFFF !important;
}

/* Botón elegante */
.stDownloadButton button {
    background-color: #1E88E5 !important;
    color: white !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
}

/* Upload area */
.stFileUploader {
    padding: 12px;
    border-radius: 10px;
    background-color: #FFFFFF;
}

/* Separadores */
hr {
    border: 1px solid #D0D6DD;
}
</style>
""", unsafe_allow_html=True)



#CARGA DEL MODELO YOLO

@st.cache_resource
def load_model():
    return YOLO("best512.pt")

model = load_model()



#INTERFAZ PRINCIPAL

st.image("logo.png")
st.title("🔍 Detección de manzanas con imágenes satelitales (GRUPO 2)")
st.write("Sube una imagen satelital y visualiza las detecciones de manzanas realizadas.")


# Sidebar moderna
st.sidebar.title("⚙️ Configuración del modelo")
conf = st.sidebar.slider("Confidence Threshold", 0.1, 0.9, 0.5)
iou = st.sidebar.slider("IoU Threshold", 0.1, 0.9, 0.45)
st.sidebar.markdown("---")
st.sidebar.info("Ajusta los valores según la sensibilidad que necesites.")


#PROCESO DE DETECCIÓN
uploaded_file = st.file_uploader(
    "Sube una imagen JPG o PNG", 
    type=["jpg", "jpeg", "png"], 
    key="file_uploader_input"
)

###tomar foto
cam_option = st.selectbox(
    "Selecciona la cámara:",
    ["📷 Cámara trasera", "🤳 Cámara frontal"]
)

if cam_option == "📷 Cámara trasera":
    facing = "environment"
else:
    facing = "user"

st.markdown(f"""
<script>
navigator.mediaDevices.getUserMedia({{ video: {{ facingMode: "{facing}" }} }});
</script>
""", unsafe_allow_html=True)


image_photo = st.camera_input(
    "O toma una foto con tu cámara",
    key="camera_input"
)

if uploaded_file is not None:
    img = Image.open(uploaded_file)

elif image_photo is not None:
    img = Image.open(image_photo)

else:
    img = None


if img:
    img_array = np.array(img)
    tab1, tab2 = st.tabs(["📤 Imagen Original", "📡 Detección con YOLO"])

    # ---------------- TAB 1 -------------------
    with tab1:
        st.markdown('<div class="image-container">', unsafe_allow_html=True)
        st.image(img, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ---------------- TAB 2 -------------------
    with tab2:
        with st.spinner("Procesando la imagen con YOLO..."):
            start_time = time.time()
            results = model(img_array, conf=conf, iou=iou)
            end_time = time.time()

        # Imagen resultante
        result_img = results[0].plot(labels=False, conf=False)
        result_pil = Image.fromarray(result_img)

        st.markdown('<div class="image-container">', unsafe_allow_html=True)
        st.image(result_pil, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Métricas
        st.success(f"✔️ Detección completada en {end_time - start_time:.2f} segundos")

        #BOTON DE DESCARGA
        buffer = io.BytesIO()
        result_pil.save(buffer, format="PNG")

        st.download_button(
            label="⬇️ Descargar imagen procesada (PNG)",
            data=buffer.getvalue(),
            file_name="imagen_procesada.png",
            mime="image/png"
        )





