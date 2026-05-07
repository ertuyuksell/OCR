import os
import streamlit as st
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from PIL import Image
import torch

# VRAM kullanımında parçalı tahsis
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

st.set_page_config(page_title="Qwen2B‑VL Görsel Analiz", layout="wide")

# -------------------- Sidebar Kontrolleri --------------------
st.sidebar.header("⚙️ Ayarlar")

# Token (max_new_tokens) slider
max_new_tokens = st.sidebar.slider(
    "Maksimum yeni token (çıktı uzunluğu)",
    min_value=64, max_value=3072, value=1024, step=32,
    help="Modelin üreteceği maksimum token sayısı. Daha yüksek değer daha uzun çıktı demektir."
)

# Çözünürlük önayarları (thumbnail için genişlik x yükseklik sınırı)
res_presets = {
    "512p (512×512)": (512, 512),
    "720p (1280×720)": (1280, 720),
    "1080p (1920×1080)": (1920, 1080),
    "1440p (2560×1440)": (2560, 1440),
    "2160p (4K - 3840×2160)": (3840, 2160),
    "4320p (8K - 7680×4320)": (7680, 4320),
    "Kare 768×768": (768, 768),
    "Kare 1024×1024": (1024, 1024),
}

res_label = st.sidebar.selectbox(
    "Görüntü çözünürlük önayarı",
    options=list(res_presets.keys()),
    index=3,  # 1440p varsayılan
    help="Resim, seçilen sınırlar içinde orantılı olarak küçültülür."
)
target_size = res_presets[res_label]

# İsteğe bağlı: Otomatik karışık hassasiyet (VRAM dostu)
use_amp = st.sidebar.checkbox(
    "CUDA otomatik karma hassasiyet (amp) kullan", value=True,
    help="CUDA varsa mixed precision, yoksa etkisiz."
)

# -------------------- Model Yükleme --------------------
@st.cache_resource
def load_model():
    model_name = "Qwen/Qwen2-VL-2B-Instruct"
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_name,
        device_map="auto",
        load_in_4bit=True,                 # 4-bit quantization
        torch_dtype=torch.float16,
        llm_int8_enable_fp32_cpu_offload=True
    )
    processor = AutoProcessor.from_pretrained(model_name)
    return model, processor

model, processor = load_model()

# -------------------- Arayüz --------------------
st.title("Qwen2B‑VL Görsel Analiz (4GB VRAM için Optimize)")
st.write("Bir görsel yükleyin, prompt girin ve analiz alın. Soldaki ayarlardan **token** ve **çözünürlüğü** değiştirebilirsiniz.")

uploaded_file = st.file_uploader("Bir görsel yükleyin (jpg/png)", type=["jpg", "jpeg", "png"])
prompt = st.text_area(
    "Analiz için prompt:",
    "Step by step, analyze the tactical situation in the image. Identify mistakes and give recommendations for the coach.",
    height=120
)

# -------------------- Çalıştırma --------------------
if uploaded_file and st.button("🔎 Analizi Başlat"):
    # Resmi aç ve seçilen çözünürlüğe orantılı küçült
    image = Image.open(uploaded_file).convert("RGB")
    image.thumbnail(target_size)  # aspect ratio korunur

    st.image(image, caption=f"Yüklenen Görsel ({res_label})", use_container_width=True)

    messages = [{
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": prompt}
        ]
    }]

    text_prompt = processor.apply_chat_template(messages, add_generation_prompt=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = processor(text=[text_prompt], images=[image], return_tensors="pt").to(device)

    # Üretim
    amp_enabled = (device == "cuda") and use_amp
    with torch.cuda.amp.autocast(enabled=amp_enabled):
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )

    # Sadece yeni üretilen kısmı decode et
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
    output_text = processor.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )[0]

    st.subheader("🧠 Modelin Analizi")
    st.write(output_text)

    # Küçük bir özet/kayıt alanı
    with st.expander("Ayar Özeti"):
        st.json({
            "max_new_tokens": max_new_tokens,
            "resolution_preset": res_label,
            "target_size": {"width": target_size[0], "height": target_size[1]},
            "device": device,
            "amp_enabled": amp_enabled
        })
