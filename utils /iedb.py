import pandas as pd

def load_iedb(csv_file):
    df = pd.read_csv(csv_file)
    return df


def compute_overlap(pred_start, pred_end, iedb_start, iedb_end):
    overlap_start = max(pred_start, iedb_start)
    overlap_end = min(pred_end, iedb_end)

    if overlap_start > overlap_end:
        return 0

    return overlap_end - overlap_start + 1


def annotate_overlap(pred_range, iedb_df):
    ps, pe = pred_range
    records = []

    for _, row in iedb_df.iterrows():
        ov_len = compute_overlap(ps, pe, row.start, row.end)
        if ov_len > 0:
            records.append({
                "IEDB_epitope": row.epitope,
                "Overlap_length": ov_len,
                "Overlap_ratio": ov_len / (pe - ps + 1)
            })

    return records
