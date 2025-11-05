# finetune_es_qu_nllb.py
import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

# -----------------------------
# CONFIG
# -----------------------------
MODEL_NAME = "facebook/nllb-200-distilled-600M"
SRC_LANG = "spa_Latn"   # español
TGT_LANG = "quy_Latn"   # quechua
DATA_FILE = "es_qu.csv"  # columnas: source,target
OUTPUT_DIR = "./nllb-es-qu-ft"
MAX_LEN = 128
BATCH_SIZE = 4
EPOCHS = 5
LR = 2e-5


def get_lang_id(tokenizer, lang_code: str):
    # tu versión no tiene lang_code_to_id
    if hasattr(tokenizer, "lang_code_to_id") and tokenizer.lang_code_to_id is not None:
        return tokenizer.lang_code_to_id[lang_code]
    return tokenizer.convert_tokens_to_ids(lang_code)


def main():
    # 1) dataset
    dataset = load_dataset("csv", data_files={"train": DATA_FILE})
    train_ds = dataset["train"]

    # 2) tokenizer y modelo
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        src_lang=SRC_LANG,
        tgt_lang=TGT_LANG,
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    # forzar salida en quechua
    tgt_id = get_lang_id(tokenizer, TGT_LANG)
    model.config.forced_bos_token_id = tgt_id

    # 3) preprocesar
    def preprocess(batch):
        model_inputs = tokenizer(
            batch["source"],
            max_length=MAX_LEN,
            truncation=True,
        )
        labels = tokenizer(
            text_target=batch["target"],
            max_length=MAX_LEN,
            truncation=True,
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized_train = train_ds.map(
        preprocess,
        batched=True,
        remove_columns=train_ds.column_names,
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    # 4) argumentos de entrenamiento
    # 👇 solo los que tu versión debería soportar
    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        learning_rate=LR,
        fp16=torch.cuda.is_available(),
        logging_steps=50,
        save_steps=500,
        save_total_limit=2,
    )

    # 5) trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    trainer.train()

    # 6) guardar
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("✅ Fine-tuning terminado en:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
