from __future__ import annotations

import argparse
import unicodedata
from pathlib import Path

import pandas as pd
import pytest

from scripts.clean_data import normalize_text, remove_repeated_sentences, run


def make_args(
        input_dir: Path,
        output_dir: Path,
        leakage_policy: str = "test-priority",
) -> argparse.Namespace:
    return argparse.Namespace(
        input_dir=input_dir,
        output_dir=output_dir,
        pattern="*.csv",
        output_format="auto",
        leakage_policy=leakage_policy,
        dedupe_by="pair",
        dedupe_repeated_sentences=True,
        min_repeated_sentence_chars=30,
        near_duplicate_threshold=0.90,
        min_article_words=10,
        min_summary_words=3,
        max_summary_ratio=1.0,
        train_ratio=0.8,
        validation_ratio=0.1,
        audit_only=False,
    )


def article(label: str) -> str:
    return (
        f"{label} là một bài viết y khoa có đủ nội dung để vượt qua "
        "ngưỡng lọc tối thiểu"
    )


def write_split(root: Path, split: str, rows: list[tuple[str, str]]) -> Path:
    path = root / split / "sample.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["Document", "Summary"]).to_csv(path, index=False)
    return path


def build_fixture(root: Path) -> dict[str, Path]:
    duplicate = article("Bệnh nhân")
    duplicate_nfd = unicodedata.normalize("NFD", duplicate)
    leaked = article("Tài liệu dùng chung")
    return {
        "train": write_split(
            root,
            "train",
            [
                (duplicate, "Đây là tóm tắt hợp lệ"),
                (duplicate_nfd, "Đây là tóm tắt hợp lệ"),
                (leaked, "Tóm tắt của tài liệu chung"),
            ],
        ),
        "validation": write_split(
            root,
            "validation",
            [
                (leaked, "Tóm tắt của tài liệu chung"),
                (article("Dữ liệu validation"), "Một tóm tắt validation hợp lệ"),
            ],
        ),
        "test": write_split(
            root,
            "test",
            [
                (leaked, "Tóm tắt của tài liệu chung"),
                (article("Dữ liệu kiểm thử"), "Một tóm tắt kiểm thử hợp lệ"),
            ],
        ),
    }


def test_normalize_text_preserves_vietnamese_and_removes_markup() -> None:
    raw = "  Bệnh\u200b nhân <b>ổn định</b>\n"
    assert normalize_text(raw) == "Bệnh nhân ổn định"


def test_repeated_long_sentences_are_removed_conservatively() -> None:
    repeated = "Đây là một câu đủ dài để được nhận diện là câu lặp."
    unique = "Nội dung thứ hai vẫn được giữ nguyên trong bài viết."
    cleaned, removed = remove_repeated_sentences(
        f'{repeated} {unique} {repeated} "Ngắn." "Ngắn."'
    )

    assert cleaned == f'{repeated} {unique} "Ngắn." "Ngắn."'
    assert removed == 1


def test_cleaning_deduplicates_and_removes_cross_split_leakage(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    original_paths = build_fixture(input_dir)
    original_bytes = {name: path.read_bytes() for name, path in original_paths.items()}

    output_dir = tmp_path / "cleaned"
    report = run(make_args(input_dir, output_dir))

    assert report["splits"]["train"]["dropped_duplicates"] == 1
    assert report["overlap_articles_before"] == {
        "train_validation": 1,
        "train_test": 1,
        "validation_test": 1,
    }
    assert report["overlap_articles_after"] == {
        "train_validation": 0,
        "train_test": 0,
        "validation_test": 0,
    }
    assert {
               split: report["splits"][split]["output_rows"]
               for split in ("train", "validation", "test")
           } == {"train": 1, "validation": 1, "test": 2}

    for split in ("train", "validation", "test"):
        cleaned = pd.read_csv(output_dir / split / "test_1.csv")
        assert list(cleaned.columns) == ["article", "summary"]
        assert original_paths[split].read_bytes() == original_bytes[split]

    with pytest.raises(FileExistsError):
        run(make_args(input_dir, output_dir))


def test_resplit_is_deterministic_by_article(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    rows = [
        (article(f"Mẫu dữ liệu {index}"), f"Tóm tắt hợp lệ số {index}")
        for index in range(100)
    ]
    write_split(input_dir, "train", rows[:50])
    write_split(input_dir, "validation", rows[50:])
    first = tmp_path / "first"
    second = tmp_path / "second"

    run(make_args(input_dir, first, leakage_policy="resplit"))
    run(make_args(input_dir, second, leakage_policy="resplit"))

    for split in ("train", "validation", "test"):
        first_data = pd.read_csv(first / split / "test_1.csv")
        second_data = pd.read_csv(second / split / "test_1.csv")
        pd.testing.assert_frame_equal(first_data, second_data)


def test_near_duplicate_documents_keep_first_row(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    words = [f"từ{index}" for index in range(120)]
    first_article = " ".join(words)
    second_words = words.copy()
    second_words[-1] = "thayđổi"
    second_article = " ".join(second_words)

    write_split(
        input_dir,
        "train",
        [(article("Dữ liệu train"), "Một tóm tắt train hợp lệ")],
    )
    write_split(
        input_dir,
        "validation",
        [
            (first_article, "Một tóm tắt gần trùng hợp lệ"),
            (second_article, "Tóm tắt khác vẫn hợp lệ cho bài"),
        ],
    )

    report = run(make_args(input_dir, tmp_path / "cleaned"))

    assert report["splits"]["validation"]["dropped_near_duplicates"] == 1
    assert report["near_duplicate_examples"][0]["dropped_row"] == 1


def test_summary_must_be_shorter_than_article(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    write_split(
        input_dir,
        "train",
        [
            (article("Dữ liệu hợp lệ"), "Một tóm tắt train hợp lệ"),
            (
                "một hai ba bốn năm sáu bảy tám chín mười",
                "không hai ba bốn năm sáu bảy tám chín mười",
            ),
        ],
    )
    write_split(
        input_dir,
        "validation",
        [(article("Dữ liệu validation"), "Một tóm tắt validation hợp lệ")],
    )

    report = run(make_args(input_dir, tmp_path / "cleaned"))

    assert report["splits"]["train"]["dropped_quality"] == {
        "summary_not_shorter_than_article": 1
    }


def test_canonical_article_summary_equality_is_rejected(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    same_content = "một hai ba bốn năm sáu bảy tám chín mười"
    write_split(
        input_dir,
        "train",
        [
            (article("Dữ liệu hợp lệ"), "Một tóm tắt train hợp lệ"),
            (same_content, same_content.upper()),
        ],
    )
    write_split(
        input_dir,
        "validation",
        [(article("Dữ liệu validation"), "Một tóm tắt validation hợp lệ")],
    )

    report = run(make_args(input_dir, tmp_path / "cleaned"))

    assert report["splits"]["train"]["dropped_quality"] == {
        "summary_equals_article": 1
    }


def test_multiple_csv_variants_are_rejected(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    for split in ("train", "validation"):
        first = write_split(
            input_dir,
            split,
            [(article(split), "Một tóm tắt hợp lệ cho dữ liệu")],
        )
        second = first.with_name("another_variant.csv")
        second.write_bytes(first.read_bytes())

    with pytest.raises(ValueError, match="multiple CSV variants"):
        run(make_args(input_dir, tmp_path / "cleaned"))


def test_csv_variant_name_must_match_across_splits(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    write_split(
        input_dir,
        "train",
        [(article("train"), "Một tóm tắt hợp lệ cho dữ liệu")],
    )
    validation = write_split(
        input_dir,
        "validation",
        [(article("validation"), "Một tóm tắt hợp lệ cho dữ liệu")],
    )
    validation.rename(validation.with_name("different_variant.csv"))

    with pytest.raises(ValueError, match="same in every populated split"):
        run(make_args(input_dir, tmp_path / "cleaned"))


def test_cleaning_refuses_to_write_empty_train_or_validation(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    leaked = article("Tài liệu giống nhau")
    for split in ("train", "validation", "test"):
        write_split(
            input_dir,
            split,
            [(leaked, "Một tóm tắt dùng chung hợp lệ")],
        )

    with pytest.raises(ValueError, match="leave train or validation empty"):
        run(make_args(input_dir, tmp_path / "cleaned"))
    assert not (tmp_path / "cleaned").exists()


def test_flat_original_layout_creates_separate_clean_datasets(tmp_path: Path) -> None:
    input_dir = tmp_path / "original"
    input_dir.mkdir()

    parquet_rows = [
        (article(f"Parquet {index}"), f"Tóm tắt parquet hợp lệ {index}")
        for index in range(6)
    ]
    for filename, rows in (
            ("train-00000.parquet", parquet_rows[:3]),
            ("valid-00000.parquet", parquet_rows[3:5]),
            ("test-00000.parquet", parquet_rows[5:]),
    ):
        pd.DataFrame(rows, columns=["article", "summary"]).to_parquet(
            input_dir / filename,
            index=False,
        )

    medical_rows = [
        (index, article(f"Medical {index}"), f"Tóm tắt medical hợp lệ {index}")
        for index in range(100)
    ]
    pd.DataFrame(
        medical_rows,
        columns=["Unnamed: 0", "Document", "Summary"],
    ).to_csv(input_dir / "herding_512_bio_medicine.csv", index=False)
    pd.DataFrame(medical_rows[:2], columns=["Unnamed: 0", "Document", "Summary"]).to_csv(
        input_dir / "another_transformed_variant.csv",
        index=False,
    )

    summary_rows = [
        (index, f"Tóm tắt corpus hợp lệ {index}", article(f"Corpus {index}"))
        for index in range(100)
    ]
    pd.DataFrame(
        summary_rows,
        columns=["Unnamed: 0", "Summary", "Text"],
    ).to_csv(input_dir / "data_summary.csv", index=False)

    output_dir = tmp_path / "clean"
    args = make_args(input_dir, output_dir)
    args.layout = "original"
    args.medical_variant = "herding_512_bio_medicine.csv"
    args.dedupe_repeated_sentences = False
    args.near_duplicate_threshold = 0.0
    report = run(args)

    assert set(report["datasets"]) == {"parquet", "medical"}
    assert report["skipped_transformed_variants"] == [
        "another_transformed_variant.csv"
    ]
    assert (output_dir / "parquet" / "train" / "test_1.parquet").is_file()
    assert (output_dir / "medical" / "train" / "test_1.csv").is_file()
    assert not (output_dir / "data_summary").exists()
    assert (output_dir / "cleaning_manifest.json").is_file()

    with pytest.raises(FileExistsError, match="never overwrites"):
        run(args)
