import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns





def make_token(token, vocab=None):
    """
    Chuẩn hóa 1 token (có thể là spacy.Token hoặc dict) thành dict dễ đọc.
    """
    def to_str(val):
        if val is None:
            return ""
        if isinstance(val, str):
            return val
        if vocab is not None:
            try:
                return vocab.strings[val]
            except Exception:
                return str(val)
        return str(val)

    # nếu là dict
    if isinstance(token, dict):
        return {
            "idx": token.get("idx", ""),
            "text": token.get("text", ""),
            "lemma": to_str(token.get("lemma", "")),
            "pos": to_str(token.get("pos", "")),
            "tag": to_str(token.get("tag", "")),
            "dep": to_str(token.get("dep", "")),
            "morph": str(token.get("morph", ""))
        }
    else:
        # giả sử là spacy.Token
        return {
            "text": getattr(token, "text", ""),
            "idx": getattr(token, "idx", ""),
            "lemma": to_str(getattr(token, "lemma_", getattr(token, "lemma", ""))),
            "pos": to_str(getattr(token, "pos_", getattr(token, "pos", ""))),
            "tag": to_str(getattr(token, "tag_", getattr(token, "tag", ""))),
            "dep": to_str(getattr(token, "dep_", getattr(token, "dep", ""))),
            "morph": str(getattr(token, "morph", ""))
        }


def build_analysis_dataset(all_labels, all_predictions, logs, id2label, vocab=None, entity_pairs=None):
    """
    Tạo dataset phân tích từ kết quả evaluate và logs, có bổ sung is_intra.
    
    Args:
        all_labels: list[int] ground truth labels
        all_predictions: list[int] predicted labels
        logs: list[dict] (từ model khi return_topk_tokens=True)
        id2label: dict {label_id: label_name}
        vocab: spacy.vocab.Vocab để decode id sang string (nếu có)
        entity_pairs: list[list] (mỗi dòng từ df.values.tolist()) chứa thông tin entity + context + is_intra

    Returns:
        data: list[dict] dạng JSON phân tích, có thêm is_intra
    """
    data = []
    N = len(all_labels)

    for i in range(N):
        gold = id2label[all_labels[i]]
        pred = id2label[all_predictions[i]]
        log = logs[i]
        
        # Lấy is_intra từ entity_pairs (giả sử đã có cột is_intra ở vị trí cuối cùng)
        is_intra = entity_pairs[i][-1]  # cột cuối là is_intra
        context = entity_pairs[i][9]    # context vẫn ở cột 9 như cũ

        entry = {
            "e1_text": entity_pairs[i][6],
            "e2_text": entity_pairs[i][7],
            "e1_span": (entity_pairs[i][2], entity_pairs[i][4]),
            "e2_span": (entity_pairs[i][3], entity_pairs[i][5]),
            "context": context,
            "relation": gold,
            "prediction": pred,
            "is_intra": is_intra,
            "e1_tokens": [make_token(tok, vocab) for tok in log.get("e1_tokens", [])],
            "e2_tokens": [make_token(tok, vocab) for tok in log.get("e2_tokens", [])],
            "e1_topk": [make_token(tok, vocab) for tok in log.get("e1_topk_tokens", [])],
            "e2_topk": [make_token(tok, vocab) for tok in log.get("e2_topk_tokens", [])],
            "p_topk": [make_token(tok, vocab) for tok in log.get("pair_topk_tokens", [])],
            "e1_scores": log.get("e1_scores", []),
            "e2_scores": log.get("e2_scores", []),
            "p_scores": log.get("pair_scores", []),
        }
        data.append(entry)
    return data


def add_is_intra_batch(df, spacy_nlp, batch_size=64, n_process=None):
    """
    Xác định 2 entity cùng câu hay không sử dụng SpaCy pipe theo batch.
    Args:
        df: DataFrame gốc, có cột 'text', 'entity1_start', 'entity2_start'
        batch_size: số lượng câu mỗi batch
        n_process: số process để song song hóa
    Returns:
        df mới có cột 'is_intra' (1 = cùng câu, 0 = khác câu)
    """
    if n_process is None:
        from multiprocessing import cpu_count
        n_process = max(1, cpu_count() - 1)

    texts = df['text'].tolist()
    e1_starts = df['entity1_start'].tolist()
    e2_starts = df['entity2_start'].tolist()
    
    is_intra_list = []

    # Sử dụng nlp.pipe với batch và n_process
    for doc, e1_start, e2_start in zip(
        spacy_nlp.pipe(texts, batch_size=batch_size, n_process=n_process, disable=["ner", "tagger", "lemmatizer"]),
        e1_starts,
        e2_starts
    ):
        # Xác định sentence chứa entity
        e1_sent_id = e2_sent_id = None
        for i, sent in enumerate(doc.sents):
            if sent.start_char <= e1_start < sent.end_char:
                e1_sent_id = i
            if sent.start_char <= e2_start < sent.end_char:
                e2_sent_id = i

        is_intra = 1 if e1_sent_id is not None and e2_sent_id is not None and e1_sent_id == e2_sent_id else 0
        is_intra_list.append(is_intra)
    
    df['is_intra'] = is_intra_list
    return df


def create_analysis_dataframe(data):
    """
    Tạo DataFrame phân tích từ dataset JSON (trả về từ build_analysis_dataset)
    
    Mỗi token (top-k hoặc thực thể) là một dòng, có đầy đủ:
        - entity_type: e1 / e2 / pair
        - token, lemma, pos, tag, dep, morph
        - attention_score (nếu có)
        - prediction, relation (label)
        - is_intra: 1 nếu 2 entity cùng sentence, 0 nếu khác
    """
    rows = []

    for entry in data:
        # e1_topk
        e1_topk = entry.get("e1_topk", [])
        e1_scores = entry.get("e1_scores", [])
        for tok, score in zip(e1_topk, e1_scores):
            rows.append({
                "entity_type": "e1",
                "token": tok["text"],
                "lemma": tok.get("lemma", tok["text"]),
                "pos": tok.get("pos", None),
                "tag": tok.get("tag", None),
                "dep": tok.get("dep", None),
                "morph": tok.get("morph", None),
                "attention_score": score,
                "prediction": entry["prediction"],
                "relation": entry["relation"],
                "is_intra": entry.get("is_intra", 1)
            })
        # e2_topk
        e2_topk = entry.get("e2_topk", [])
        e2_scores = entry.get("e2_scores", [])
        for tok, score in zip(e2_topk, e2_scores):
            rows.append({
                "entity_type": "e2",
                "token": tok["text"],
                "lemma": tok.get("lemma", tok["text"]),
                "pos": tok.get("pos", None),
                "tag": tok.get("tag", None),
                "dep": tok.get("dep", None),
                "morph": tok.get("morph", None),
                "attention_score": score,
                "prediction": entry["prediction"],
                "relation": entry["relation"],
                "is_intra": entry.get("is_intra", 1)
            })
        # pair_topk
        p_topk = entry.get("p_topk", [])
        p_scores = entry.get("p_scores", [])
        for tok, score in zip(p_topk, p_scores):
            rows.append({
                "entity_type": "pair",
                "token": tok["text"],
                "lemma": tok.get("lemma", tok["text"]),
                "pos": tok.get("pos", None),
                "tag": tok.get("tag", None),
                "dep": tok.get("dep", None),
                "morph": tok.get("morph", None),
                "attention_score": score,
                "prediction": entry["prediction"],
                "relation": entry["relation"],
                "is_intra": entry.get("is_intra", 1)
            })

    df = pd.DataFrame(rows)
    return df


def pos_distribution(df_analysis):
    # ===============================
    # Thống kê POS distribution
    # ===============================
    pos_counts = df_analysis.groupby('entity_type')['pos'].value_counts(normalize=True).unstack(fill_value=0)
    print("POS distribution per entity_type:")
    print(pos_counts)
    
    # Vẽ heatmap POS distribution
    plt.figure(figsize=(12,6))
    sns.heatmap(pos_counts, annot=True, cmap='Blues')
    plt.title('POS Distribution per Entity Type')
    plt.ylabel('Entity Type')
    plt.xlabel('POS Tag')
    plt.show()

def dep_distribution(df_analysis):
    # ===============================
    # Thống kê Dependency distribution
    # ===============================
    dep_counts = df_analysis.groupby('entity_type')['dep'].value_counts(normalize=True).unstack(fill_value=0)
    print("Dependency distribution per entity_type:")
    print(dep_counts)
    
    # Vẽ heatmap Dependency distribution
    plt.figure(figsize=(12,6))
    sns.heatmap(dep_counts, annot=True, cmap='Greens')
    plt.title('Dependency Distribution per Entity Type')
    plt.ylabel('Entity Type')
    plt.xlabel('Dependency Tag')
    plt.show()


def morph_distribution(df_analysis):
    # ===============================
    # Thống kê Morphology distribution
    # ===============================
    morph_counts = df_analysis.groupby('entity_type')['morph'].value_counts(normalize=True).unstack(fill_value=0)
    print("Morphology distribution per entity_type:")
    print(morph_counts)
    
    # Heatmap Morphology distribution
    plt.figure(figsize=(12,6))
    sns.heatmap(morph_counts, annot=True, cmap='Oranges')
    plt.title('Morphology Distribution per Entity Type')
    plt.ylabel('Entity Type')
    plt.xlabel('Morphological Features')
    plt.show()

def summarize_by_intra(df_analysis):
    """
    Thống kê POS / dep / morph phân theo entity_type và is_intra
    """
    # POS distribution
    pos_counts = df_analysis.groupby(['entity_type','is_intra'])['pos'] \
                            .value_counts(normalize=True).unstack(fill_value=0)
    print("POS distribution by entity_type and is_intra:")
    print(pos_counts)

    plt.figure(figsize=(14,6))
    sns.heatmap(pos_counts, annot=True, cmap='Blues')
    plt.title('POS Distribution by Entity Type and is_intra')
    plt.ylabel('Entity Type + is_intra')
    plt.xlabel('POS Tag')
    plt.show()

    # Dependency distribution
    dep_counts = df_analysis.groupby(['entity_type','is_intra'])['dep'] \
                            .value_counts(normalize=True).unstack(fill_value=0)
    print("Dependency distribution by entity_type and is_intra:")
    print(dep_counts)

    plt.figure(figsize=(14,6))
    sns.heatmap(dep_counts, annot=True, cmap='Greens')
    plt.title('Dependency Distribution by Entity Type and is_intra')
    plt.ylabel('Entity Type + is_intra')
    plt.xlabel('Dependency Tag')
    plt.show()

    # Morphology distribution
    morph_counts = df_analysis.groupby(['entity_type','is_intra'])['morph'] \
                              .value_counts(normalize=True).unstack(fill_value=0)
    print("Morphology distribution by entity_type and is_intra:")
    print(morph_counts)

    plt.figure(figsize=(14,6))
    sns.heatmap(morph_counts, annot=True, cmap='Oranges')
    plt.title('Morphology Distribution by Entity Type and is_intra')
    plt.ylabel('Entity Type + is_intra')
    plt.xlabel('Morphological Features')
    plt.show()


def analyze_correct_prediction_intra(df_analysis):
    # Lọc các token cùng câu và prediction đúng
    df_intra = df_analysis[df_analysis['is_intra']==1]
    df_correct = df_intra[df_intra['prediction'] == df_intra['relation']]

    print("Số token intra-sentence và prediction đúng:", len(df_correct))

    # POS distribution
    pos_dist = df_correct.groupby('entity_type')['pos'].value_counts(normalize=True).unstack(fill_value=0)
    print("POS distribution (intra, correct prediction):")
    print(pos_dist)

    plt.figure(figsize=(12,5))
    sns.heatmap(pos_dist, annot=True, cmap='Blues')
    plt.title('POS Distribution by Entity Type (Intra-sentence, Correct Prediction)')
    plt.ylabel('Entity Type')
    plt.xlabel('POS Tag')
    plt.show()

    # Dependency distribution
    dep_dist = df_correct.groupby('entity_type')['dep'].value_counts(normalize=True).unstack(fill_value=0)
    print("Dependency distribution (intra, correct prediction):")
    print(dep_dist)

    plt.figure(figsize=(12,5))
    sns.heatmap(dep_dist, annot=True, cmap='Greens')
    plt.title('Dependency Distribution by Entity Type (Intra-sentence, Correct Prediction)')
    plt.ylabel('Entity Type')
    plt.xlabel('Dependency Tag')
    plt.show()

    # Morphology distribution
    morph_dist = df_correct.groupby('entity_type')['morph'].value_counts(normalize=True).unstack(fill_value=0)
    print("Morphology distribution (intra, correct prediction):")
    print(morph_dist)

    plt.figure(figsize=(12,5))
    sns.heatmap(morph_dist, annot=True, cmap='Oranges')
    plt.title('Morphology Distribution by Entity Type (Intra-sentence, Correct Prediction)')
    plt.ylabel('Entity Type')
    plt.xlabel('Morphological Features')
    plt.show()

def analyze_pos_dep_morph_by_relation(df_analysis):
    """
    Phân tích POS / dep / morph theo relation, chỉ xem prediction đúng và is_intra = 1
    """
    # Lọc token cùng câu và prediction đúng
    # df_intra_correct = df_analysis[(df_analysis['is_intra']==1) &
    #                                (df_analysis['prediction'] == df_analysis['relation'])]

    df_intra_correct = df_analysis[df_analysis['prediction'] == df_analysis['relation']]

    print("Số cặp với prediction đúng:", len(df_intra_correct))

    relations = df_intra_correct['relation'].unique()
    entity_types = df_intra_correct['entity_type'].unique()

    for rel in relations:
        print(f"\n=== Relation: {rel} ===")
        df_rel = df_intra_correct[df_intra_correct['relation']==rel]

        # POS distribution
        pos_dist = df_rel.groupby('entity_type')['pos'].value_counts(normalize=True).unstack(fill_value=0)
        print("POS distribution:")
        print(pos_dist)

        plt.figure(figsize=(12,4))
        sns.heatmap(pos_dist, annot=True, cmap='Blues')
        plt.title(f'POS Distribution - Relation={rel} (Correct Prediction)')
        plt.ylabel('Entity Type')
        plt.xlabel('POS Tag')
        plt.show()

        # Dependency distribution
        dep_dist = df_rel.groupby('entity_type')['dep'].value_counts(normalize=True).unstack(fill_value=0)
        print("Dependency distribution:")
        print(dep_dist)

        plt.figure(figsize=(12,4))
        sns.heatmap(dep_dist, annot=True, cmap='Greens')
        plt.title(f'Dependency Distribution - Relation={rel} (Correct Prediction)')
        plt.ylabel('Entity Type')
        plt.xlabel('Dependency Tag')
        plt.show()

        # Morphology distribution
        morph_dist = df_rel.groupby('entity_type')['morph'].value_counts(normalize=True).unstack(fill_value=0)
        print("Morphology distribution:")
        print(morph_dist)

        plt.figure(figsize=(12,4))
        sns.heatmap(morph_dist, annot=True, cmap='Oranges')
        plt.title(f'Morphology Distribution - Relation={rel} (Correct Prediction)')
        plt.ylabel('Entity Type')
        plt.xlabel('Morphological Features')
        plt.show()
        
