import torch
import os
from tqdm import tqdm
from metrics.recall import recall
from metrics.rouge import evalROUGE
from metrics.logger import MetricsLogger


def writeHistoryHeader(history_path=None):
    if history_path:
        with open(history_path, "w") as f:
            f.write("epoch;train_loss;train_recall;val_loss;val_recall;val_r1_p;val_r1_r;val_r1_f1;"
                    "val_r2_p;val_r2_r;val_r2_f1;val_rl_p;val_rl_r;val_rl_f1\n")

def writeHistoryEntry(epoch, train_metrics, val_metrics, history_path=None):
    if history_path:
        with open(history_path, "a") as f:
            train_avgs, val_avgs = train_metrics.averages(), val_metrics.averages()
            f.write(f"{epoch};{train_avgs['loss']};{train_avgs['recall']};{val_avgs['loss']};{val_avgs['recall']};"
                    f"{val_avgs['rouge1']['precision']};{val_avgs['rouge1']['recall']};{val_avgs['rouge1']['fmeasure']};"
                    f"{val_avgs['rouge2']['precision']};{val_avgs['rouge2']['recall']};{val_avgs['rouge2']['fmeasure']};"
                    f"{val_avgs['rougeL']['precision']};{val_avgs['rougeL']['recall']};{val_avgs['rougeL']['fmeasure']}\n")

def perSentenceLoss(loss, batch_predictions, batch_labels, batch_num_sentences):
    total_loss = 0.0
    for batch_i in range(len(batch_predictions)):
        predictions = batch_predictions[batch_i][: batch_num_sentences[batch_i]]
        labels = batch_labels[batch_i][: batch_num_sentences[batch_i]]
        total_loss += loss(predictions, labels)
    return total_loss / len(batch_predictions)

def train(model, loss, optimizer, train_dataloader, val_dataloader, epochs, device,
          history_path, checkpoint, checkpoints_path, checkpoints_frequency,
          checkpoint_best, accumulation_steps=1, use_mixed_precision=False):

    def _createCheckpoint(path, epoch_num, model, optimizer, metrics):
        torch.save({
            "epoch": epoch_num,
            "model_state_dict": model.state_dict(),
            "model_name": model.model_name,
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics
        }, path)

    model, loss = model.to(device), loss.to(device)
    starting_epoch, curr_best_val_recall = 1, -1
    train_metrics, val_metrics = MetricsLogger(), MetricsLogger()
    scaler = torch.cuda.amp.GradScaler()

    if checkpoint:
        checkpoint = torch.load(checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        starting_epoch = checkpoint["epoch"] + 1

    os.makedirs(checkpoints_path, exist_ok=True)
    writeHistoryHeader(history_path)
    epochs = epochs + starting_epoch - 1

    for epoch_num in range(starting_epoch, epochs+1):
        train_metrics.reset()
        val_metrics.reset()

        # === Training ===
        model.train()
        optimizer.zero_grad()
        for i, (documents, labels) in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch_num}/{epochs}")):
            labels = labels.float().to(device)
            if use_mixed_precision:
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    predictions, logits = model(documents)
                    batch_loss = perSentenceLoss(loss, logits, labels, documents["num_sentences"])
                    acc_loss = batch_loss / accumulation_steps
                scaler.scale(acc_loss).backward()
            else:
                predictions, logits = model(documents)
                batch_loss = perSentenceLoss(loss, logits, labels, documents["num_sentences"])
                acc_loss = batch_loss / accumulation_steps
                acc_loss.backward()

            if ((i+1) % accumulation_steps == 0) or ((i+1) == len(train_dataloader)):
                if use_mixed_precision:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()

            train_metrics.add("loss", batch_loss.item())
            train_metrics.add("recall", recall(labels, predictions))

        # === Validation ===
        model.eval()
        with torch.no_grad():
            for documents, labels in tqdm(val_dataloader, desc="Validation"):
                labels = labels.float().to(device)
                outputs, logits = model(documents)
                batch_loss = perSentenceLoss(loss, logits, labels, documents["num_sentences"])

                ext_summaries = []
                for i in range(len(labels)):
                    ext_summary_size = len((labels[i] == 1).nonzero(as_tuple=True)[0])
                    ext_sentences, _ = model.summarizeFromDataset(outputs[i], documents["ids"][i], ext_summary_size)
                    ext_summaries.append("\n".join(ext_sentences))

                val_metrics.add("loss", batch_loss.item())
                val_metrics.add("recall", recall(labels, outputs))
                val_metrics.add("rouge", evalROUGE(documents["ref_summary"], ext_summaries))

        # === Logs & checkpoints ===
        is_best = (val_metrics.averages()["recall"] > curr_best_val_recall and checkpoint_best)
        if (epoch_num % checkpoints_frequency == 0) or is_best or (epoch_num == epochs):
            checkpoint_path = os.path.join(checkpoints_path, f"cp_{model.model_name.replace('/', '_')}_ep{epoch_num:03d}.tar")
            _createCheckpoint(checkpoint_path, epoch_num, model, optimizer, val_metrics.averages())
            if is_best:
                curr_best_val_recall = val_metrics.averages()["recall"]
                with open(os.path.join(checkpoints_path, "best.txt"), "w") as f:
                    f.write(f"Epoch {epoch_num} | {val_metrics.format(['loss', 'recall', 'rouge'])}")

        writeHistoryEntry(epoch_num, train_metrics, val_metrics, history_path)
