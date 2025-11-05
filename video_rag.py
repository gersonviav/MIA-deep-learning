import os
import cv2
import numpy as np
import torch
import faiss
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
from typing import List
from openai import OpenAI
import librosa
import soundfile as sf

# ======================================================
# CONFIGURACIÓN
# ======================================================
VIDEO_PATH = "videoplayback.mp4"   # <-- tu video local
SEGMENT_SECONDS = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ======================================================
# (NUEVO) CARGAR TRADUCTOR ES -> QUECHUA
# ======================================================
TRANSLATOR_DIR = "./nllb-es-qu-ft"   # donde guardaste el modelo fine-tuneado
SRC_LANG = "spa_Latn"
TGT_LANG = "quy_Latn"

translator_tokenizer = AutoTokenizer.from_pretrained(
    TRANSLATOR_DIR,
    src_lang=SRC_LANG,
    tgt_lang=TGT_LANG,
)
translator_model = AutoModelForSeq2SeqLM.from_pretrained(TRANSLATOR_DIR).to(DEVICE)

def _get_lang_id(tokenizer, lang_code: str):
    if hasattr(tokenizer, "lang_code_to_id") and tokenizer.lang_code_to_id is not None:
        return tokenizer.lang_code_to_id[lang_code]
    return tokenizer.convert_tokens_to_ids(lang_code)

translator_model.config.forced_bos_token_id = _get_lang_id(translator_tokenizer, TGT_LANG)

def translate_es_to_qu(text: str, max_length: int = 128) -> str:
    inputs = translator_tokenizer(text, return_tensors="pt").to(DEVICE)
    generated = translator_model.generate(
        **inputs,
        max_length=max_length,
    )
    return translator_tokenizer.batch_decode(generated, skip_special_tokens=True)[0]


# ======================================================
# ETAPA 1: SEGMENTAR VIDEO
# ======================================================
def segment_video(video_path: str, segment_length: int = 10) -> List[tuple]:
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"❌ No se encontró el video: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"⚠️ No se pudo abrir el video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or np.isnan(fps):
        print("⚠️ No se detectó FPS, usando valor por defecto 30.")
        fps = 30

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    segments = [(i, min(i + segment_length, duration)) for i in range(0, int(duration), segment_length)]
    cap.release()
    return segments

# ======================================================
# ETAPA 2: EXTRAER AUDIO (solo ffmpeg)
# ======================================================
def extract_audio(video_path: str, output_audio="audio.wav"):
    print("🎧 Extrayendo audio con FFmpeg...")
    cmd = f'ffmpeg -y -i "{video_path}" -vn -acodec pcm_s16le -ar 16000 -ac 1 "{output_audio}" >nul 2>&1'
    result = os.system(cmd)
    if result != 0 or not os.path.exists(output_audio):
        raise RuntimeError("❌ Error extrayendo audio. Verifica que FFmpeg esté instalado y en PATH.")
    print("✅ Audio extraído correctamente.")
    return output_audio

# ======================================================
# ETAPA 3: TRANSCRIPCIÓN (Whisper)
# ======================================================
def transcribe_audio(audio_path: str):
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"❌ No existe el archivo de audio: {audio_path}")

    print("🗣️ Transcribiendo audio por fragmentos  (máx 30s cada uno)...")

    y, sr = librosa.load(audio_path, sr=16000)
    segment_duration = 30
    samples_per_segment = segment_duration * sr
    total_samples = len(y)

    asr = pipeline("automatic-speech-recognition", model="openai/whisper-base")

    full_text = ""
    for i in range(0, total_samples, samples_per_segment):
        segment = y[i:i + samples_per_segment]
        tmp_path = "tmp_segment.wav"
        sf.write(tmp_path, segment, sr)
        try:
            result = asr(tmp_path, return_timestamps=False)
            text = result["text"].strip()
            full_text += " " + text
            print(f"✅ Segmento {(i // samples_per_segment) + 1} transcrito.")
        except Exception as e:
            print(f"⚠️ Error en segmento {(i // samples_per_segment) + 1}: {e}")

    print("✅ Transcripción completa.")
    return full_text.strip()

# ======================================================
# ETAPA 4: EMBEDDINGS + FAISS
# ======================================================
def embed_texts(texts: List[str]):
    print("🔍 Generando embeddings...")
    model = SentenceTransformer("clip-ViT-B-32")
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    return embeddings, model

def create_faiss_index(embeddings: np.ndarray):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    print("✅ Índice FAISS creado.")
    return index

# ======================================================
# ETAPA 5: RETRIEVAL + GENERACIÓN
# ======================================================
def retrieve(query: str, texts: List[str], index, embed_model, k=3):
    q_emb = embed_model.encode([query])
    D, I = index.search(q_emb, k)
    return [texts[i] for i in I[0]]

# 👇 MODIFICADA
def generate_answer(question: str, context: str, out_lang: str = "es"):
    prompt = f"""
    Usa el siguiente contexto transcrito del video para responder con precisión:

    CONTEXTO:
    {context}

    PREGUNTA:
    {question}

    RESPUESTA EN ESPAÑOL CLARO:
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )
    answer_es = response.choices[0].message.content

    if out_lang == "qu":
        # lo pasamos por el modelo local afinado
        try:
            return translate_es_to_qu(answer_es)
        except Exception as e:
            # si falla, devolvemos español
            return answer_es + f"\n\n[no se pudo traducir: {e}]"
    return answer_es

# ======================================================
# PIPELINE PRINCIPAL
# ======================================================
def main():
    print("🎬 Segmentando video...")
    segments = segment_video(VIDEO_PATH, SEGMENT_SECONDS)
    print(f"✅ {len(segments)} segmentos creados ({SEGMENT_SECONDS}s c/u)")

    audio_path = extract_audio(VIDEO_PATH)
    text = transcribe_audio(audio_path)

    chunks = [text[i:i+400] for i in range(0, len(text), 400)]
    embeddings, embed_model = embed_texts(chunks)
    index = create_faiss_index(embeddings)

    print("\n✅ Sistema RAG listo. Puedes hacer preguntas sobre el video.\n")
    while True:
        q = input("❓ Pregunta (o 'salir'): ")
        if q.lower() == "salir":
            break

        # 👇 aquí decides el idioma de salida
        lang = input("🌐 Idioma respuesta (es/qu): ").strip().lower() or "es"

        retrieved = retrieve(q, chunks, index, embed_model)
        context = "\n".join(retrieved)
        answer = generate_answer(q, context, out_lang=lang)
        print("\n💬 Respuesta:")
        print(answer)
        print("-" * 60)

# ======================================================
if __name__ == "__main__":
    main()
