import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
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


from i2b2_eval import build_relations_from_df, evaluate_i2b2_closure


def evaluate(
    dataloader,
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
    vague_label_id=None,
    test_df=None,
    id2label=None,
):
    """
    Evaluate model cho các bộ dữ liệu:
    - I2B2: nếu có test_df -> closure-based (document-level). Nếu không -> weighted F1.
    - TDD/TBD/MATRES: dùng harmonic F1 như bạn đang tính.
    - Chỉ lọc VAGUE cho TDD/TBD/MATRES. KHÔNG lọc khi is_I2B2=True.
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

            # ===== Forward =====
            if return_topk_tokens:
                out = model(
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
                # Tương thích cả model cũ (2 outputs) & multi-head mới (3 outputs)
                if isinstance(out, tuple) and len(out) == 3:
                    logits, _, batch_logs = out
                else:
                    logits, batch_logs = out
            else:
                out = model(
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
                if isinstance(out, tuple):
                    logits = out[0]
                else:
                    logits = out
                batch_logs = None

            preds = torch.argmax(logits, dim=1)

            # ===== Chỉ lọc VAGUE cho TDD/TBD/MATRES =====
            if (is_TDD or is_TBD or is_MATRES) and (vague_label_id is not None):
                mask = (labels != vague_label_id)
                skipped_vague += (~mask).sum().item()
                labels = labels[mask]
                preds  = preds[mask]

                # Chuẩn hóa dạng logs: có thể là list hoặc {"logs": list}
                if batch_logs is not None:
                    if isinstance(batch_logs, dict) and "logs" in batch_logs:
                        batch_logs = batch_logs["logs"]
                    # lọc theo mask
                    batch_logs = [l for (i, l) in enumerate(batch_logs) if mask[i]]

            # ===== I2B2: không lọc gì cả =====
            else:
                # Nếu có logs dạng dict -> lấy list bên trong
                if batch_logs is not None and isinstance(batch_logs, dict) and "logs" in batch_logs:
                    batch_logs = batch_logs["logs"]

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(preds.cpu().numpy())

            if return_logs and return_topk_tokens and batch_logs is not None:
                logs.extend(batch_logs)

    elapsed_time = time.time() - start_time
    print(f"Evaluation finished in {elapsed_time:.2f}s.")
    if (is_TDD or is_TBD or is_MATRES) and (vague_label_id is not None):
        print(f"Filtered {skipped_vague}/{total_samples} "
              f"({100*skipped_vague/total_samples:.2f}%) VAGUE samples.")

    # ==================================================
    # ===  I2B2 EVALUATION
    # ==================================================
    if is_I2B2:
        from i2b2_eval import build_relations_from_df, evaluate_i2b2_closure

        # Nếu có test_df → closure-based
        if test_df is not None:
            print("Using closure-based evaluation for I2B2 (document-level).")
            # Chú ý: all_predictions phải khớp thứ tự record trong test_df
            gold_relations, pred_relations = build_relations_from_df(
                test_df,
                preds=all_predictions,
                label_col="label_encoded",
                id2label=id2label
            )
            metrics = evaluate_i2b2_closure(gold_relations, pred_relations)
            F1, P, R = metrics["F1"], metrics["Precision"], metrics["Recall"]

            print("====== [I2B2 CLOSURE-BASED EVALUATION] ======")
            print(f"Precision: {P:.4f}")
            print(f"Recall:    {R:.4f}")
            print(f"F1-score:  {F1:.4f}")
            print("==============================================")

            if return_preds or return_logs:
                return (
                    None, F1, P, R, [], [], [],
                    all_predictions, all_labels, logs
                )
            else:
                return None, F1, P, R, [], [], []

        # Nếu KHÔNG có test_df → weighted-F1
        else:
            print("Using standard weighted-F1 evaluation for I2B2 (no test_df provided).")
            acc = accuracy_score(all_labels, all_predictions)
            f1 = f1_score(all_labels, all_predictions, average='weighted')
            p = precision_score(all_labels, all_predictions, average='weighted')
            r = recall_score(all_labels, all_predictions, average='weighted')
            cm = confusion_matrix(all_labels, all_predictions)
            precisions, recalls, f1s, _ = precision_recall_fscore_support(
                all_labels, all_predictions, average=None)
            print(f"[I2B2] Accuracy={acc:.4f}, F1={f1:.4f}, P={p:.4f}, R={r:.4f}")

            if return_preds or return_logs:
                return (
                    cm, f1, p, r, precisions, recalls, f1s,
                    all_predictions, all_labels, logs
                )
            else:
                return cm, f1, p, r, precisions, recalls, f1s

    # ==================================================
    # ===  TDD / TBD / MATRES EVALUATION
    # ==================================================
    if is_TDD or is_TBD or is_MATRES:
        cm = confusion_matrix(all_labels, all_predictions)
        TP = np.diag(cm).sum()
        P = TP / cm.sum(axis=0).sum() if cm.sum(axis=0).sum() > 0 else 0
        R = TP / cm.sum(axis=1).sum() if cm.sum(axis=1).sum() > 0 else 0
        F1 = 2 * P * R / (P + R) if (P + R) > 0 else 0
        precisions, recalls, f1s, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average=None
        )
        print(f"[TDD/TBD/MATRES] F1={F1:.4f}, P={P:.4f}, R={R:.4f}")

        if return_preds or return_logs:
            return (
                cm, F1, P, R, precisions, recalls, f1s,
                all_predictions, all_labels, logs
            )
        else:
            return cm, F1, P, R, precisions, recalls, f1s

    # ==================================================
    # === DEFAULT CASE (NO FLAGS)
    # ==================================================
    if return_preds or return_logs:
        return None, 0, 0, 0, [], [], [], all_predictions, all_labels, logs
    else:
        return None, 0, 0, 0, [], [], []

# ===========================================================
#  Hàm train_model – giữ nguyên is_TBD, is_TDD, is_I2B2 flag
# và mask loss cho nhãn VAGUE
# ===========================================================
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
                lambda_vague: float = 0.2,   # thêm trọng số vague head
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

            logits, vague_logits, aux_logits = model(
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

            # ⚠️ Mask loss cho nhãn VAGUE (relation head)
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

            # === Loss chính (quan hệ thời gian)
            rel_loss = criterion(masked_logits, masked_labels)

            # === Loss phụ: aux logits
            aux_loss = 0.0
            if lambda_aux > 0 and aux_logits is not None:
                for lg in aux_logits:
                    if vague_label_id is not None:
                        aux_loss += criterion(lg[mask], masked_labels)
                    else:
                        aux_loss += criterion(lg, labels)
                aux_loss = aux_loss / len(aux_logits)

            # === Loss cho head VAGUE
            if vague_label_id is not None:
                vague_target = (labels == vague_label_id).float()
                vague_loss = F.binary_cross_entropy_with_logits(vague_logits, vague_target)
            else:
                vague_loss = 0.0

            loss = rel_loss + lambda_aux * aux_loss + lambda_vague * vague_loss

            if not torch.isfinite(loss):
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            optimizer.step()
            total_loss += loss.item()

        epoch_loss = total_loss / max(1, len(train_dataloader))
        losses.append(epoch_loss)
        epoch_time = time.time() - epoch_start_time

        print(f"Epoch {epoch+1}: Loss={epoch_loss:.4f}, Rel={rel_loss:.4f}, "
              f"Vague={vague_loss:.4f}, Skipped {skipped_vague}/{total_samples} "
              f"({100*skipped_vague/total_samples:.2f}% VAGUE) | Time={epoch_time:.2f}s")

        # === Save best model ===
        if epoch_loss <= best_loss:
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'loss': epoch_loss,
            }, save_path)
            best_loss = epoch_loss

        # === Evaluate định kỳ ===
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


# ===========================================================
# 🧾 Evaluate và hiển thị chi tiết từng nhãn
# ===========================================================
def evaluate_and_show(dataloader,
                      model,
                      top_k,
                      device,
                      relations,
                      return_preds=False,
                      is_I2B2=False,
                      is_TDD=False,
                      is_TBD=False,
                      is_MATRES=False,
                      vague_label_id=None,
                      test_df=None):
    cm, f1, p, r, precisions, recalls, f1s = evaluate(
        dataloader,
        model=model,
        top_k=top_k,
        device=device,
        return_preds=return_preds,
        is_I2B2=is_I2B2,
        is_TDD=is_TDD,
        is_TBD=is_TBD,
        is_MATRES=is_MATRES,
        vague_label_id=vague_label_id,
        test_df=test_df
    )

    print(f"\nEval Summary: F1={f1:.4f}, P={p:.4f}, R={r:.4f}")
    for i in range(len(relations)):
        print(f"{relations[i]:<15}: F1={f1s[i]:.2f}, P={precisions[i]:.2f}, R={recalls[i]:.2f}")
    print("Confusion Matrix:")
    print(cm)
