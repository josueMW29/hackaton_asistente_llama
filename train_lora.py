

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

BASE_DIR = Path(__file__).parent


def cargar_datos(path: Path, tone: str) -> Dataset:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if tone == "all" or row["tone"] == tone:
            rows.append({"messages": row["messages"]})
    if not rows:
        raise SystemExit(f"No hay ejemplos para el tono '{tone}' en {path}")
    return Dataset.from_list(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Entrena un adaptador LoRA de tono con SFTTrainer.")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--tone", choices=["formal", "amigable", "all"], default="formal")
    parser.add_argument("--data", type=Path, default=BASE_DIR / "datos_tono.jsonl")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    output = args.output or BASE_DIR / "adapters" / f"llama-academico-{args.tone}"
    dataset = cargar_datos(args.data, args.tone)
    print(f"Ejemplos de entrenamiento: {len(dataset)} | tono: {args.tone} | base: {args.base_model}")

    lora = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    config = SFTConfig(
        output_dir=str(output),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=args.lr,
        logging_steps=1,
        save_strategy="no",
        max_length=1024,
        bf16=use_bf16,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=args.base_model,
        args=config,
        train_dataset=dataset,
        peft_config=lora,
    )
    trainer.train()
    trainer.save_model(str(output))

    (output / "training_log.json").write_text(
        json.dumps(trainer.state.log_history, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Adaptador guardado en {output}")


if __name__ == "__main__":
    main()
