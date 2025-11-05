import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_DIR = "./nllb-es-qu-ft"   # carpeta donde guardaste el modelo entrenado
SRC_LANG = "spa_Latn"
TGT_LANG = "quy_Latn"

device = "cuda" if torch.cuda.is_available() else "cpu"

# cargamos tokenizer y modelo
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_DIR,
    src_lang=SRC_LANG,
    tgt_lang=TGT_LANG,
)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR).to(device)

def get_lang_id(tok, lang_code: str):
    # si tu versión tuviera lang_code_to_id lo usa, si no, convierte el token
    if hasattr(tok, "lang_code_to_id") and tok.lang_code_to_id is not None:
        return tok.lang_code_to_id[lang_code]
    return tok.convert_tokens_to_ids(lang_code)

# forzamos que el modelo genere en quechua
model.config.forced_bos_token_id = get_lang_id(tokenizer, TGT_LANG)

def translate_es_to_qu(text: str, max_length: int = 128) -> str:
    inputs = tokenizer(text, return_tensors="pt").to(device)
    # ojo: ya seteamos forced_bos_token_id en el config, así que no lo ponemos aquí
    generated = model.generate(
        **inputs,
        max_length=max_length,
    )
    return tokenizer.batch_decode(generated, skip_special_tokens=True)[0]

if __name__ == "__main__":
    out = translate_es_to_qu("El video explica el ahorro de energía.")
    print(out)
