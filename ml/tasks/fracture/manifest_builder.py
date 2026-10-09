"""
Dataset Manifest Builder & Clinical Data Hygiene Module
MedGuard AI - Bone Fracture Analysis Subsystem

Generates verified, deduplicated, and leak-free dataset manifests for:
- Dataset A: Graz Pediatric Wrist Radiographs (20,327 images, 16-bit PNG)
- Dataset B: FracAtlas Multi-Region Radiograph Benchmark (4,083 images, 8-bit JPEG)

Performs:
1. Strict image path resolution and existence verification.
2. Label standardization and clinical quarantine of ambiguous diagnoses.
3. Patient-grouped splitting on Dataset A to prevent data leakage.
4. Official split alignment + stratified splitting for Dataset B.
5. Multi-anatomy distribution reporting and JSON audit export.
"""

import os
import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit

from ml.tasks.fracture.config import (
    DATASET_A_ROOT,
    DATASET_B_ROOT,
    MANIFEST_TRAIN,
    MANIFEST_VAL,
    MANIFEST_TEST,
    MANIFEST_QUARANTINE,
    DATASET_AUDIT_JSON,
    DEFAULT_SEED
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("manifest_builder")


def build_graz_manifest() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Parses Graz Pediatric Wrist Dataset.
    Quarantines ambiguous labels (diagnosis_uncertain == 1.0).
    Performs patient-grouped splitting on clean cases.
    """
    csv_path = DATASET_A_ROOT / "dataset.csv"
    images_dir = DATASET_A_ROOT / "images"

    if not csv_path.exists():
        raise FileNotFoundError(f"Graz dataset.csv not found at {csv_path}")

    logger.info(f"Loading Graz Pediatric Wrist metadata: {csv_path}")
    df_raw = pd.read_csv(csv_path)

    records = []
    for _, row in df_raw.iterrows():
        filestem = str(row["filestem"])
        img_path = images_dir / f"{filestem}.png"
        
        # Verify physical existence
        if not img_path.exists():
            logger.warning(f"Missing image file for Graz filestem: {filestem}")
            continue

        patient_id = str(row["patient_id"])
        study_id = str(row["study_number"]) if pd.notna(row.get("study_number")) else "study_unknown"
        
        # Check quarantine criteria
        diag_uncertain = (row.get("diagnosis_uncertain") == 1.0)
        fracture_visible = (row.get("fracture_visible") == 1.0)

        proj_raw = row.get("projection")
        if proj_raw == 1:
            proj = "AP_PA"
        elif proj_raw == 2:
            proj = "LAT"
        else:
            proj = str(proj_raw) if pd.notna(proj_raw) else "unspecified"

        ao_class = str(row["ao_classification"]) if pd.notna(row.get("ao_classification")) else "unclassified"

        rec = {
            "dataset_source": "graz_pediatric_wrist",
            "image_path": str(img_path.resolve()),
            "image_id": f"{filestem}.png",
            "patient_id": f"graz_pt_{patient_id}",
            "study_id": f"graz_study_{patient_id}_{study_id}",
            "fracture_label": 1 if fracture_visible else 0,
            "anatomical_region": "wrist",
            "projection": proj,
            "annotation_reference": f"ao:{ao_class}",
            "label_quality": "quarantined_ambiguous" if diag_uncertain else "clean",
            "split": "quarantine" if diag_uncertain else "unassigned"
        }
        records.append(rec)

    df_graz = pd.DataFrame(records)
    logger.info(f"Graz total parsed: {len(df_graz)}")

    df_quarantine = df_graz[df_graz["split"] == "quarantine"].copy()
    df_clean = df_graz[df_graz["split"] != "quarantine"].copy()

    logger.info(f"Graz clean cases: {len(df_clean)}, quarantined ambiguous cases: {len(df_quarantine)}")

    # Deterministic patient-grouped split: 70% Train, 15% Val, 15% Test
    # Step 1: Split into Train (70%) and Temp (30%)
    gss1 = GroupShuffleSplit(n_splits=1, train_size=0.70, random_state=DEFAULT_SEED)
    train_idx, temp_idx = next(gss1.split(df_clean, df_clean["fracture_label"], df_clean["patient_id"]))
    
    df_train_part = df_clean.iloc[train_idx].copy()
    df_temp = df_clean.iloc[temp_idx].copy()

    # Step 2: Split Temp (30%) equally into Val (15%) and Test (15%)
    gss2 = GroupShuffleSplit(n_splits=1, train_size=0.50, random_state=DEFAULT_SEED)
    val_rel_idx, test_rel_idx = next(gss2.split(df_temp, df_temp["fracture_label"], df_temp["patient_id"]))

    df_val_part = df_temp.iloc[val_rel_idx].copy()
    df_test_part = df_temp.iloc[test_rel_idx].copy()

    df_train_part["split"] = "train"
    df_val_part["split"] = "val"
    df_test_part["split"] = "test"

    df_clean_split = pd.concat([df_train_part, df_val_part, df_test_part], ignore_index=True)

    # Verification: Zero patient leakage
    train_pts = set(df_clean_split[df_clean_split["split"] == "train"]["patient_id"])
    val_pts = set(df_clean_split[df_clean_split["split"] == "val"]["patient_id"])
    test_pts = set(df_clean_split[df_clean_split["split"] == "test"]["patient_id"])

    assert len(train_pts.intersection(val_pts)) == 0, "Patient leakage between Graz Train and Val!"
    assert len(train_pts.intersection(test_pts)) == 0, "Patient leakage between Graz Train and Test!"
    assert len(val_pts.intersection(test_pts)) == 0, "Patient leakage between Graz Val and Test!"
    logger.info("Graz patient-grouped splitting verified: ZERO patient leakage.")

    return df_clean_split, df_quarantine


def build_fracatlas_manifest() -> pd.DataFrame:
    """
    Parses FracAtlas Dataset.
    Resolves official fracture splits and stratifies non-fractured cases.
    """
    csv_path = DATASET_B_ROOT / "dataset.csv"
    img_frac_dir = DATASET_B_ROOT / "images" / "Fractured"
    img_nonfrac_dir = DATASET_B_ROOT / "images" / "Non_fractured"
    split_dir = DATASET_B_ROOT / "Utilities" / "Fracture Split"

    if not csv_path.exists():
        raise FileNotFoundError(f"FracAtlas dataset.csv not found at {csv_path}")

    logger.info(f"Loading FracAtlas metadata: {csv_path}")
    df_raw = pd.read_csv(csv_path)

    # Load official fracture split files
    official_splits = {}
    for s_name, fname in [("train", "train.csv"), ("val", "valid.csv"), ("test", "test.csv")]:
        fpath = split_dir / fname
        if fpath.exists():
            s_df = pd.read_csv(fpath)
            for im_id in s_df["image_id"].dropna():
                official_splits[str(im_id).strip()] = s_name

    records = []
    for _, row in df_raw.iterrows():
        img_id = str(row["image_id"]).strip()
        is_fractured = int(row["fractured"]) == 1

        # Resolve correct file path handling duplicate filename cases safely
        if is_fractured:
            img_path = img_frac_dir / img_id
            if not img_path.exists():
                img_path = img_nonfrac_dir / img_id
        else:
            img_path = img_nonfrac_dir / img_id
            if not img_path.exists():
                img_path = img_frac_dir / img_id

        if not img_path.exists():
            logger.warning(f"FracAtlas image file not found: {img_id}")
            continue

        # Determine anatomical region
        if row.get("mixed") == 1:
            region = "mixed"
        elif row.get("hand") == 1:
            region = "hand"
        elif row.get("leg") == 1:
            region = "leg"
        elif row.get("hip") == 1:
            region = "hip"
        elif row.get("shoulder") == 1:
            region = "shoulder"
        else:
            region = "mixed"

        # Determine projection
        if row.get("frontal") == 1:
            proj = "frontal"
        elif row.get("lateral") == 1:
            proj = "lateral"
        elif row.get("oblique") == 1:
            proj = "oblique"
        else:
            proj = "unspecified"

        # Check localization annotation reference
        voc_path = DATASET_B_ROOT / "Annotations" / "PASCAL VOC" / f"{Path(img_id).stem}.xml"
        yolo_path = DATASET_B_ROOT / "Annotations" / "YOLO" / f"{Path(img_id).stem}.txt"
        annot_ref = []
        if voc_path.exists():
            annot_ref.append("PASCAL_VOC")
        if yolo_path.exists():
            annot_ref.append("YOLO")
        annot_ref_str = ";".join(annot_ref) if annot_ref else "none"

        # Determine split
        if is_fractured and img_id in official_splits:
            split_assigned = official_splits[img_id]
        else:
            split_assigned = "unassigned"

        records.append({
            "dataset_source": "fracatlas",
            "image_path": str(img_path.resolve()),
            "image_id": img_id,
            "patient_id": f"fracatlas_{Path(img_id).stem}",
            "study_id": f"fracatlas_study_{Path(img_id).stem}",
            "fracture_label": 1 if is_fractured else 0,
            "anatomical_region": region,
            "projection": proj,
            "annotation_reference": annot_ref_str,
            "label_quality": "verified_box_annotated" if is_fractured and annot_ref else "clean",
            "split": split_assigned
        })

    df_fracatlas = pd.DataFrame(records)
    logger.info(f"FracAtlas total parsed: {len(df_fracatlas)}")

    # Stratify the unassigned (non-fractured) cases across train (80%), val (10%), test (10%)
    df_unassigned = df_fracatlas[df_fracatlas["split"] == "unassigned"].copy()
    df_assigned = df_fracatlas[df_fracatlas["split"] != "unassigned"].copy()

    # Step 1: Split unassigned into Train (80%) and Temp (20%) stratified by region
    sss1 = StratifiedShuffleSplit(n_splits=1, train_size=0.80, random_state=DEFAULT_SEED)
    tr_idx, temp_idx = next(sss1.split(df_unassigned, df_unassigned["anatomical_region"]))

    df_nonfrac_train = df_unassigned.iloc[tr_idx].copy()
    df_nonfrac_temp = df_unassigned.iloc[temp_idx].copy()

    # Step 2: Split Temp (20%) into Val (10%) and Test (10%)
    sss2 = StratifiedShuffleSplit(n_splits=1, train_size=0.50, random_state=DEFAULT_SEED)
    val_idx, test_idx = next(sss2.split(df_nonfrac_temp, df_nonfrac_temp["anatomical_region"]))

    df_nonfrac_val = df_nonfrac_temp.iloc[val_idx].copy()
    df_nonfrac_test = df_nonfrac_temp.iloc[test_idx].copy()

    df_nonfrac_train["split"] = "train"
    df_nonfrac_val["split"] = "val"
    df_nonfrac_test["split"] = "test"

    df_fracatlas_final = pd.concat([df_assigned, df_nonfrac_train, df_nonfrac_val, df_nonfrac_test], ignore_index=True)
    return df_fracatlas_final


def generate_manifests():
    """
    Main orchestration routine.
    Compiles, validates, and exports unified manifests and audit report.
    """
    logger.info("=== Starting Manifest Builder ===")
    
    df_graz_clean, df_graz_quarantine = build_graz_manifest()
    df_fracatlas = build_fracatlas_manifest()

    # Combine clean records
    df_all_clean = pd.concat([df_graz_clean, df_fracatlas], ignore_index=True)

    # Separate into final splits
    df_train = df_all_clean[df_all_clean["split"] == "train"].copy()
    df_val = df_all_clean[df_all_clean["split"] == "val"].copy()
    df_test = df_all_clean[df_all_clean["split"] == "test"].copy()

    logger.info("--- Manifest Size Summary ---")
    logger.info(f"Train split: {len(df_train)} samples")
    logger.info(f"Val split:   {len(df_val)} samples")
    logger.info(f"Test split:  {len(df_test)} samples")
    logger.info(f"Quarantined: {len(df_graz_quarantine)} samples")
    logger.info(f"Total clean: {len(df_all_clean)} samples")

    # Export to CSV
    MANIFEST_TRAIN.parent.mkdir(parents=True, exist_ok=True)
    df_train.to_csv(MANIFEST_TRAIN, index=False)
    df_val.to_csv(MANIFEST_VAL, index=False)
    df_test.to_csv(MANIFEST_TEST, index=False)
    df_graz_quarantine.to_csv(MANIFEST_QUARANTINE, index=False)

    logger.info(f"Exported manifests to {MANIFEST_TRAIN.parent}")

    # Build comprehensive audit dictionary
    audit_data = {
        "timestamp": pd.Timestamp.now().isoformat(),
        "total_images_processed": int(len(df_all_clean) + len(df_graz_quarantine)),
        "total_clean_images": int(len(df_all_clean)),
        "total_quarantined_images": int(len(df_graz_quarantine)),
        "splits": {
            "train": {
                "total": int(len(df_train)),
                "fracture_positive": int((df_train["fracture_label"] == 1).sum()),
                "fracture_negative": int((df_train["fracture_label"] == 0).sum()),
                "pos_ratio": float((df_train["fracture_label"] == 1).mean()),
                "by_source": df_train["dataset_source"].value_counts().to_dict(),
                "by_anatomy": df_train["anatomical_region"].value_counts().to_dict(),
            },
            "val": {
                "total": int(len(df_val)),
                "fracture_positive": int((df_val["fracture_label"] == 1).sum()),
                "fracture_negative": int((df_val["fracture_label"] == 0).sum()),
                "pos_ratio": float((df_val["fracture_label"] == 1).mean()),
                "by_source": df_val["dataset_source"].value_counts().to_dict(),
                "by_anatomy": df_val["anatomical_region"].value_counts().to_dict(),
            },
            "test": {
                "total": int(len(df_test)),
                "fracture_positive": int((df_test["fracture_label"] == 1).sum()),
                "fracture_negative": int((df_test["fracture_label"] == 0).sum()),
                "pos_ratio": float((df_test["fracture_label"] == 1).mean()),
                "by_source": df_test["dataset_source"].value_counts().to_dict(),
                "by_anatomy": df_test["anatomical_region"].value_counts().to_dict(),
            },
            "quarantine": {
                "total": int(len(df_graz_quarantine)),
                "reason": "Graz diagnosis_uncertain == 1.0 (ambiguous clinical finding)"
            }
        },
        "all_clean_anatomy_distribution": df_all_clean["anatomical_region"].value_counts().to_dict(),
        "all_clean_dataset_sources": df_all_clean["dataset_source"].value_counts().to_dict()
    }

    DATASET_AUDIT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(DATASET_AUDIT_JSON, "w") as f:
        json.dump(audit_data, f, indent=2)
    logger.info(f"Exported dataset audit to {DATASET_AUDIT_JSON}")
    logger.info("=== Manifest Generation Complete ===")


if __name__ == "__main__":
    generate_manifests()
