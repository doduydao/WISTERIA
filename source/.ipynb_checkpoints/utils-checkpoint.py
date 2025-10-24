import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    precision_recall_fscore_support, confusion_matrix
)
import warnings
from sklearn.exceptions import UndefinedMetricWarning
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)


# ====================================================
# 🔍 HÀM EVALUATE – hỗ trợ is_TBD, is_TDD, is_I2B2, ...
# ====================================================
def evaluate(dataloader,
             model,
             top_k,
             device,
             threshold=1.0,
             return_preds=False,
             return_topk_tokens=False,
             return_logs=False,
             is_I2B2=False,
             is_TDD=False,
             is_TBD=False,
             is_MATRES=False,
             vague_label_id=None):
    """
    Evaluate model, có thể trả về log attention/top-k.
    Bỏ các mẫu có nhãn VAGUE khỏi metric nếu vague_label_id != None.
    """

    model.eval()
    start_time = time.time()

    all_predictions = []
    all_labels = []
    logs = []
    skipped_vague = 0
    total_samples = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids      = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            entity_mask_e1 = batch['entity_mask_e1'].to(device)
            entity_mask_e2 = batch['entity_mask_e2'].to(device)
            spacy_masks    = batch['spacy_masks'].to(device)
            spacy_tokens   = batch['spacy_tokens']
            labels         = batch['labels'].to(device)
            total_samples += labels.size(0)

            token_type_ids = batch.get('token_type_ids', None)
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)

            if return_topk_tokens:
                logits, batch_logs = model(
                    input_ids=input_ids,
                    token_type_ids=token_type_ids,
                    attention_mask=attention_mask,
                    entity_mask_e1=entity_mask_e1,
                    entity_mask_e2=entity_mask_e2,
                    spacy_masks=spacy_masks,
                    spacy_tokens=spacy_tokens,
                    top_k=top_k,
                    return_topk_tokens=True
                )
            else:
                logits = model(
                    input_ids=input_ids,
                    token_type_ids=token_type_ids,
                    attention_mask=attention_mask,
                    entity_mask_e1=entity_mask_e1,
                    entity_mask_e2=entity_mask_e2,
                    spacy_masks=spacy_masks,
                    spacy_tokens=spacy_tokens,
                    top_k=top_k,
                    return_topk_tokens=False
                )
                batch_logs = None

            preds = torch.argmax(logits, dim=1)

            # ⚠️ Bỏ VAGUE
            if vague_label_id is not None:
                mask = (labels != vague_label_id)
                skipped_vague += (~mask).sum().item()
                labels = labels[mask]
                preds  = preds[mask]
                if batch_logs is not None:
                    # lọc log tương ứng
                    batch_logs = [l for (i, l) in enumerate(batch_logs) if mask[i]]

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(preds.cpu().numpy())

            if return_topk_tokens and return_logs and batch_logs is not None:
                logs.extend(batch_logs)

    elapsed_time = time.time() - start_time
    print(f"Evaluation finished in {elapsed_time:.2f}s.")
    if vague_label_id is not None:
        print(f"Filtered {skipped_vague}/{total_samples} "
              f"({100*skipped_vague/total_samples:.2f}%) VAGUE samples.")

    # === TÍNH METRIC THEO FLAG ===
    if is_I2B2:
        acc = accuracy_score(all_labels, all_predictions)
        f1 = f1_score(all_labels, all_predictions, average='weighted')
        p = precision_score(all_labels, all_predictions, average='weighted')
        r = recall_score(all_labels, all_predictions, average='weighted')
        cm = confusion_matrix(all_labels, all_predictions)
        precisions, recalls, f1s, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average=None)
        print(f"F1={f1:.4f}, P={p:.4f}, R={r:.4f}")
        result = (cm, f1, p, r, precisions, recalls, f1s)

    elif is_TDD or is_TBD or is_MATRES:
        cm = confusion_matrix(all_labels, all_predictions)
        TP = np.diag(cm).sum()
        P = TP / cm.sum(axis=0).sum() if cm.sum(axis=0).sum() > 0 else 0
        R = TP / cm.sum(axis=1).sum() if cm.sum(axis=1).sum() > 0 else 0
        F1 = 2 * P * R / (P + R) if (P + R) > 0 else 0
        precisions, recalls, f1s, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average=None)
        print(f"F1={F1:.4f}, P={P:.4f}, R={R:.4f}")
        result = (cm, F1, P, R, precisions, recalls, f1s)
    else:
        result = (None, 0, 0, 0, [], [], [])

    # Trả logs nếu yêu cầu
    if return_preds and return_logs and return_topk_tokens:
        return *result, all_predictions, all_labels, logs
    else:
        return result


# ======================================================
# 🧩 TRAIN MODEL – mask loss cho VAGUE, giữ toàn bộ flag
# ======================================================
def train_model(model,
                top_k,
                train_dataloader,
                validation_dataloader,
                optimizer,
                criterion,
                device,
                num_epochs,
                save_path,
                eval_step,
                max_grad_norm: float = 1.0,
                lambda_aux: float = 0.3,
                is_I2B2=False,
                is_TDD=False,
                is_TBD=False,
                is_MATRES=False,
                vague_label_id: int = None):

    losses = []
    best_loss = float("inf")
    total_start_time = time.time()

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        skipped_vague = 0
        total_samples = 0
        epoch_start_time = time.time()

        for batch in tqdm(train_dataloader, desc=f"Training epoch {epoch+1}:"):
            input_ids      = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            entity_mask_e1 = batch['entity_mask_e1'].to(device)
            entity_mask_e2 = batch['entity_mask_e2'].to(device)
            spacy_masks    = batch['spacy_masks'].to(device)
            spacy_tokens   = batch['spacy_tokens']
            labels         = batch['labels'].to(device)
            total_samples += labels.size(0)

            token_type_ids = batch.get('token_type_ids', None)
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)

            optimizer.zero_grad(set_to_none=True)

            logits, aux_logits = model(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
                entity_mask_e1=entity_mask_e1,
                entity_mask_e2=entity_mask_e2,
                spacy_masks=spacy_masks,
                spacy_tokens=spacy_tokens,
                top_k=top_k,
                return_aux_logits=True
            )

            # ⚠️ MASK LOSS CHO NHÃN VAGUE
            if vague_label_id is not None:
                mask = (labels != vague_label_id)
                skipped_vague += (~mask).sum().item()
                if mask.sum() == 0:
                    continue
                masked_logits = logits[mask]
                masked_labels = labels[mask]
            else:
                masked_logits = logits
                masked_labels = labels

            loss = criterion(masked_logits, masked_labels)

            if lambda_aux > 0 and aux_logits is not None:
                aux_loss = 0.0
                for lg in aux_logits:
                    if vague_label_id is not None:
                        aux_loss += criterion(lg[mask], masked_labels)
                    else:
                        aux_loss += criterion(lg, labels)
                loss += lambda_aux * aux_loss

            if not torch.isfinite(loss):
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            optimizer.step()
            total_loss += loss.item()

        epoch_loss = total_loss / max(1, len(train_dataloader))
        losses.append(epoch_loss)
        epoch_time = time.time() - epoch_start_time
        print(f"Epoch {epoch+1}: Loss={epoch_loss:.4f}, Skipped {skipped_vague}/{total_samples} "
              f"({100*skipped_vague/total_samples:.2f}% VAGUE) | Time={epoch_time:.2f}s")

        # Save best model
        if epoch_loss <= best_loss:
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'loss': epoch_loss,
            }, save_path)
            best_loss = epoch_loss

        # Evaluate định kỳ
        if epoch % eval_step == 0:
            _, f1, p, r, _, _, _ = evaluate(
                validation_dataloader,
                model=model,
                top_k=top_k,
                device=device,
                is_I2B2=is_I2B2,
                is_TDD=is_TDD,
                is_TBD=is_TBD,
                is_MATRES=is_MATRES,
                vague_label_id=vague_label_id
            )
            print(f"→ Eval epoch {epoch+1}: F1={f1:.4f}, P={p:.4f}, R={r:.4f}")

    print(f"Training finished in {time.time() - total_start_time:.2f}s.")
    return losses


# ======================================================
# 📊 Hiển thị kết quả từng nhãn + log attention
# ======================================================
def evaluate_and_show(dataloader,
                      model,
                      top_k,
                      device,
                      relations,
                      return_preds=False,
                      return_topk_tokens=False,
                      return_logs=False,
                      is_I2B2=False,
                      is_TDD=False,
                      is_TBD=False,
                      is_MATRES=False,
                      vague_label_id=None):
    result = evaluate(
        dataloader,
        model=model,
        top_k=top_k,
        device=device,
        return_preds=return_preds,
        return_topk_tokens=return_topk_tokens,
        return_logs=return_logs,
        is_I2B2=is_I2B2,
        is_TDD=is_TDD,
        is_TBD=is_TBD,
        is_MATRES=is_MATRES,
        vague_label_id=vague_label_id
    )

    if return_preds and return_logs and return_topk_tokens:
        cm, f1, p, r, precisions, recalls, f1s, preds, labels, logs = result
    else:
        cm, f1, p, r, precisions, recalls, f1s = result
        preds, labels, logs = None, None, None

    print(f"\nEval Summary: F1={f1:.4f}, P={p:.4f}, R={r:.4f}")
    for i in range(len(relations)):
        print(f"{relations[i]:<15}: F1={f1s[i]:.2f}, P={precisions[i]:.2f}, R={recalls[i]:.2f}")
    print("Confusion Matrix:")
    print(cm)
    if return_logs:
        return cm, f1, p, r, precisions, recalls, f1s, preds, labels, logs
    return cm, f1, p, r, precisions, recalls, f1s
