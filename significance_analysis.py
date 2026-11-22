#!/usr/bin/env python3
"""
Significance analysis for PPSG/data.xlsx.

The workbook is expected to contain rows grouped by Dataset, Method, and Seed,
with PoseScript and PPSG results under the same seeds. The script performs
paired exact sign-flip permutation tests within each dataset and metric.

Outputs:
  significance_results.csv
  significance_results.md
"""

from __future__ import annotations

import argparse
import csv
import itertools
import math
import re
import statistics
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path


NS = {
    "a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
}


DEFAULT_METRIC_ORDER = [
    "FID",
    "MPJPE_avg",
    "MPJPE_dis",
    "PA-MPJPE_avg",
    "PA-MPJPE_dis",
    "MPVE_avg",
    "MPVE_dis",
    "PA-MPVE_avg",
    "PA-MPVE_dis",
]


def col_to_num(cell_ref: str) -> int | None:
    match = re.match(r"([A-Z]+)", cell_ref or "")
    if not match:
        return None
    value = 0
    for char in match.group(1):
        value = value * 26 + ord(char) - ord("A") + 1
    return value


def clean_metric_name(value: object) -> str:
    text = "" if value is None else str(value).strip()
    text = re.sub(r"[^A-Za-z0-9_-]+", "", text)
    return text


def parse_cell_text(cell: ET.Element, shared_strings: list[str]) -> str:
    cell_type = cell.attrib.get("t")
    if cell_type == "s":
        value = cell.find("a:v", NS)
        if value is None or value.text is None:
            return ""
        return shared_strings[int(value.text)]
    if cell_type == "inlineStr":
        inline = cell.find("a:is", NS)
        if inline is None:
            return ""
        return "".join(node.text or "" for node in inline.iter() if node.tag.endswith("}t"))
    value = cell.find("a:v", NS)
    return "" if value is None or value.text is None else value.text


def normalize_sheet_path(target: str) -> str:
    if target.startswith("/"):
        return target[1:]
    if target.startswith("xl/"):
        return target
    return "xl/" + target


def read_xlsx_first_sheet(path: Path) -> list[list[str]]:
    with zipfile.ZipFile(path) as zf:
        shared_strings: list[str] = []
        if "xl/sharedStrings.xml" in zf.namelist():
            root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
            for item in root.findall("a:si", NS):
                shared_strings.append("".join(t.text or "" for t in item.findall(".//a:t", NS)))

        workbook = ET.fromstring(zf.read("xl/workbook.xml"))
        rels = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
        rel_map = {rel.attrib["Id"]: rel.attrib["Target"] for rel in rels}
        first_sheet = workbook.find("a:sheets/a:sheet", NS)
        if first_sheet is None:
            raise ValueError("No sheet found in workbook.")
        rel_id = first_sheet.attrib[f"{{{NS['r']}}}id"]
        sheet_path = normalize_sheet_path(rel_map[rel_id])

        sheet = ET.fromstring(zf.read(sheet_path))
        rows: list[list[str]] = []
        for row in sheet.findall("a:sheetData/a:row", NS):
            values: list[str] = []
            last_col = 0
            for cell in row.findall("a:c", NS):
                col_num = col_to_num(cell.attrib.get("r", "")) or last_col + 1
                values.extend([""] * (col_num - last_col - 1))
                values.append(parse_cell_text(cell, shared_strings))
                last_col = col_num
            rows.append(values)
        return rows


def to_float(value: object) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def sample_std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return statistics.stdev(values)


def mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else math.nan


def exact_sign_flip_pvalue(diffs: list[float], alternative: str = "two-sided") -> float:
    clean = [d for d in diffs if d is not None]
    if not clean:
        return math.nan
    if all(abs(d) < 1e-12 for d in clean):
        return 1.0

    magnitudes = [abs(d) for d in clean]
    observed = mean(clean)
    stats = []
    for signs in itertools.product((-1.0, 1.0), repeat=len(magnitudes)):
        stats.append(mean([s * v for s, v in zip(signs, magnitudes)]))

    eps = 1e-12
    if alternative == "candidate-less":
        count = sum(stat <= observed + eps for stat in stats)
    elif alternative == "candidate-greater":
        count = sum(stat >= observed - eps for stat in stats)
    else:
        threshold = abs(observed)
        count = sum(abs(stat) >= threshold - eps for stat in stats)
    return count / len(stats)


def benjamini_hochberg(pvalues: list[float]) -> list[float]:
    m = len(pvalues)
    indexed = sorted(enumerate(pvalues), key=lambda x: (math.inf if math.isnan(x[1]) else x[1]))
    qvalues = [math.nan] * m
    prev = 1.0
    for rank_from_end, (idx, pvalue) in enumerate(reversed(indexed), start=1):
        rank = m - rank_from_end + 1
        if math.isnan(pvalue):
            qvalues[idx] = math.nan
            continue
        qvalue = min(prev, pvalue * m / rank)
        qvalues[idx] = min(qvalue, 1.0)
        prev = qvalues[idx]
    return qvalues


def stars(qvalue: float) -> str:
    if math.isnan(qvalue):
        return ""
    if qvalue < 0.001:
        return "***"
    if qvalue < 0.01:
        return "**"
    if qvalue < 0.05:
        return "*"
    return "ns"


def format_float(value: float, digits: int = 4) -> str:
    if value is None or math.isnan(value):
        return ""
    return f"{value:.{digits}f}"


def discover_table(rows: list[list[str]]) -> tuple[list[str], list[dict[str, object]]]:
    header_idx = None
    for idx, row in enumerate(rows):
        normalized = [str(v).strip() for v in row[:4]]
        if len(normalized) >= 4 and normalized[:4] == ["Dataset", "Method", "Seed", "FID"]:
            header_idx = idx
            break
    if header_idx is None:
        raise ValueError("Could not find header row: Dataset, Method, Seed, FID.")

    header = rows[header_idx]
    metric_header = rows[header_idx - 1] if header_idx > 0 else []
    max_cols = max(len(header), len(metric_header))

    columns: list[str] = []
    current_metric = ""
    for col in range(max_cols):
        lower = header[col].strip() if col < len(header) else ""
        upper = metric_header[col].strip() if col < len(metric_header) else ""
        if col == 0:
            columns.append("Dataset")
        elif col == 1:
            columns.append("Method")
        elif col == 2:
            columns.append("Seed")
        elif col == 3:
            columns.append("FID")
        else:
            if upper:
                current_metric = clean_metric_name(upper)
            if not current_metric:
                current_metric = f"Metric{col + 1}"
            columns.append(f"{current_metric}_{lower}")

    records: list[dict[str, object]] = []
    last_dataset = ""
    last_method = ""
    for row in rows[header_idx + 1 :]:
        padded = row + [""] * (len(columns) - len(row))
        if not any(str(v).strip() for v in padded):
            continue
        dataset = str(padded[0]).strip() or last_dataset
        method = str(padded[1]).strip() or last_method
        seed = to_float(padded[2])
        if not dataset or not method or seed is None:
            continue
        last_dataset = dataset
        last_method = method

        record: dict[str, object] = {
            "Dataset": dataset,
            "Method": method,
            "Seed": int(seed),
        }
        for col_idx, name in enumerate(columns[3:], start=3):
            record[name] = to_float(padded[col_idx])
        records.append(record)
    return columns, records


def ordered_unique(values: list[object]) -> list[object]:
    seen = set()
    ordered = []
    for value in values:
        if value not in seen:
            seen.add(value)
            ordered.append(value)
    return ordered


def analyze(records: list[dict[str, object]], baseline: str, candidate: str, alternative: str) -> list[dict[str, object]]:
    datasets = ordered_unique([r["Dataset"] for r in records])
    metric_names = [m for m in DEFAULT_METRIC_ORDER if any(m in r and r[m] is not None for r in records)]

    results: list[dict[str, object]] = []
    by_key = {(r["Dataset"], r["Method"], r["Seed"]): r for r in records}
    for dataset in datasets:
        seeds = sorted(
            {
                r["Seed"]
                for r in records
                if r["Dataset"] == dataset and r["Method"] in {baseline, candidate}
            }
        )
        for metric in metric_names:
            base_values = []
            cand_values = []
            used_seeds = []
            for seed in seeds:
                base_record = by_key.get((dataset, baseline, seed))
                cand_record = by_key.get((dataset, candidate, seed))
                if not base_record or not cand_record:
                    continue
                base = base_record.get(metric)
                cand = cand_record.get(metric)
                if base is None or cand is None:
                    continue
                base_values.append(float(base))
                cand_values.append(float(cand))
                used_seeds.append(int(seed))
            if not used_seeds:
                continue

            diffs = [cand - base for cand, base in zip(cand_values, base_values)]
            base_mean = mean(base_values)
            cand_mean = mean(cand_values)
            delta = cand_mean - base_mean
            improvement = (base_mean - cand_mean) / abs(base_mean) * 100 if abs(base_mean) > 1e-12 else math.nan
            diff_std = sample_std(diffs)
            cohen_dz = delta / diff_std if diff_std > 1e-12 else (math.inf if abs(delta) > 1e-12 else 0.0)
            results.append(
                {
                    "Dataset": dataset,
                    "Metric": metric,
                    "Baseline": baseline,
                    "Candidate": candidate,
                    "Seeds": ",".join(str(s) for s in used_seeds),
                    "N": len(used_seeds),
                    "Baseline_mean": base_mean,
                    "Baseline_sd": sample_std(base_values),
                    "Candidate_mean": cand_mean,
                    "Candidate_sd": sample_std(cand_values),
                    "Delta_candidate_minus_baseline": delta,
                    "Relative_improvement_percent": improvement,
                    "Mean_paired_diff": mean(diffs),
                    "Paired_diff_sd": diff_std,
                    "Cohen_dz": cohen_dz,
                    "P_value": exact_sign_flip_pvalue(diffs, alternative=alternative),
                }
            )

    qvalues = benjamini_hochberg([float(r["P_value"]) for r in results])
    for row, qvalue in zip(results, qvalues):
        row["Q_value_BH_FDR"] = qvalue
        row["Significance_FDR"] = stars(qvalue)
    return results


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("No analysis rows to write.")
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, object]], alternative: str) -> None:
    lines = [
        "# Significance Analysis Results",
        "",
        f"Test: paired exact sign-flip permutation test ({alternative}).",
        "Multiple-comparison correction: Benjamini-Hochberg FDR across all rows.",
        "Delta is Candidate - Baseline. Relative improvement assumes lower is better.",
        "",
        "Note: with N=3 paired seeds, the smallest possible two-sided exact p-value is 0.25.",
        "",
        "| Dataset | Metric | N | Baseline mean +/- sd | Candidate mean +/- sd | Delta | Improvement % | p | q | Sig |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {Dataset} | {Metric} | {N} | {base} | {cand} | {delta} | {imp} | {p} | {q} | {sig} |".format(
                Dataset=row["Dataset"],
                Metric=row["Metric"],
                N=row["N"],
                base=f"{format_float(row['Baseline_mean'], 3)} +/- {format_float(row['Baseline_sd'], 3)}",
                cand=f"{format_float(row['Candidate_mean'], 3)} +/- {format_float(row['Candidate_sd'], 3)}",
                delta=format_float(row["Delta_candidate_minus_baseline"], 4),
                imp=format_float(row["Relative_improvement_percent"], 2),
                p=format_float(row["P_value"], 4),
                q=format_float(row["Q_value_BH_FDR"], 4),
                sig=row["Significance_FDR"],
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run paired significance analysis for PPSG data.xlsx.")
    parser.add_argument("--input", default="data.xlsx", type=Path, help="Path to data.xlsx.")
    parser.add_argument("--baseline", default="PoseScript", help="Baseline method name.")
    parser.add_argument("--candidate", default="PPSG", help="Candidate method name.")
    parser.add_argument(
        "--alternative",
        default="two-sided",
        choices=["two-sided", "candidate-less", "candidate-greater"],
        help="Permutation-test alternative. Use candidate-less when lower metric values are expected for candidate.",
    )
    parser.add_argument("--output-prefix", default="significance_results", help="Output file prefix.")
    args = parser.parse_args()

    input_path = args.input
    if not input_path.is_absolute():
        input_path = Path(__file__).resolve().parent / input_path
    if not input_path.is_file():
        raise FileNotFoundError(f"Input workbook not found: {input_path}")

    rows = read_xlsx_first_sheet(input_path)
    _, records = discover_table(rows)
    results = analyze(records, args.baseline, args.candidate, args.alternative)

    output_dir = input_path.parent
    csv_path = output_dir / f"{args.output_prefix}.csv"
    md_path = output_dir / f"{args.output_prefix}.md"
    write_csv(csv_path, results)
    write_markdown(md_path, results, args.alternative)

    print(f"Loaded {len(records)} rows from {input_path}")
    print(f"Wrote {len(results)} tests to {csv_path}")
    print(f"Wrote Markdown summary to {md_path}")


if __name__ == "__main__":
    main()
