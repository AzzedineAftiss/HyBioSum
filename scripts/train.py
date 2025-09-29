import argparse
import torch
from torch.utils.data import DataLoader
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModel

from extractive_stage_sum.model_longformer import LongformerSummarizer
from extractive_stage_sum.training_loop import train
from extractive_stage_sum.summary_utils import custom_collate_fn
from metrics.logger import MetricsLogger


from extractive_stage_sum.data_utils import LongformerDataset
from extractive_stage_sum.preprocess_longformer import preprocessUtilitiesLongformer



def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load tokenizer + dataset
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)
    dataset = load_from_disk(args.dataset_dir)

    train_dataset = LongformerDataset(dataset["train"], tokenizer, args.input_size)
    val_dataset = LongformerDataset(dataset["validation"], tokenizer, args.input_size)

    # train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
    #                           collate_fn=lambda b: custom_collate_fn(b, tokenizer))
    # val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
    #                         collate_fn=lambda b: custom_collate_fn(b, tokenizer))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Load model
    base_model = AutoModel.from_pretrained(args.model_name).to(device)
    model = LongformerSummarizer(base_model, tokenizer, input_size=args.input_size)

    loss = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    train(
        model=model,
        loss=loss,
        optimizer=optimizer,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        epochs=args.epochs,
        device=device,
        history_path=args.history_path,
        checkpoint=args.checkpoint,
        checkpoints_path=args.checkpoints_path,
        checkpoints_frequency=args.checkpoints_freq,
        checkpoint_best=args.checkpoint_best,
        use_mixed_precision=args.mixed_precision
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="allenai/longformer-base-4096")
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--input_size", type=int, default=4096)
    parser.add_argument("--history_path", type=str, default="./history.csv")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--checkpoints_path", type=str, default="./checkpoints/")
    parser.add_argument("--checkpoints_freq", type=int, default=1)
    parser.add_argument("--checkpoint_best", action="store_true")
    parser.add_argument("--mixed_precision", action="store_true")
    args = parser.parse_args()
    main(args)