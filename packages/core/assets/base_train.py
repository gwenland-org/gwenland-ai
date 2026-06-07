import json, sys, argparse, signal
import torch
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig
from transformers import TrainerCallback
from datasets import load_dataset

# GPU check — fail fast before loading model
if not torch.cuda.is_available():
    print(json.dumps({"error": "no_gpu", "message": "CUDA not available"}), flush=True)
    sys.exit(1)

class JsonLoggerCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and state.global_step > 0:
            print(json.dumps({
                "step": state.global_step,
                "epoch": round(state.epoch, 2),
                "loss": logs.get("loss"),
                "lr": logs.get("learning_rate"),
            }), flush=True)

def train(args):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_len,
        load_in_4bit=args.qlora,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.lora_target.split(","),
        use_gradient_checkpointing=True,
    )

    def handle_sigint(sig, frame):
        print(json.dumps({"event": "interrupted", "message": "saving checkpoint..."}), flush=True)
        model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        sys.exit(0)
    signal.signal(signal.SIGINT, handle_sigint)

    dataset = load_dataset("json", data_files=args.dataset, split="train")

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=SFTConfig(
            output_dir=args.output_dir,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            learning_rate=args.learning_rate,
            max_seq_length=args.max_seq_len,
            fp16=args.fp16,
            optim=args.optimizer,
            lr_scheduler_type=args.scheduler,
            weight_decay=args.weight_decay,
            logging_steps=1,
        ),
        callbacks=[JsonLoggerCallback()],
    )
    trainer.train()
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(json.dumps({"event": "done", "output": args.output_dir}), flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-target", default="q_proj,v_proj")
    parser.add_argument("--qlora", action="store_true")
    parser.add_argument("--optimizer", default="adamw_8bit")
    parser.add_argument("--scheduler", default="cosine")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--weight-decay", type=float, default=0.01)
    train(parser.parse_args())
