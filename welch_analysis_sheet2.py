#!/usr/bin/env python3
"""
Welch t-tests from PPSG/data.xlsx Sheet2 summary statistics.

Sheet2 is expected to store values as "mean +/- sd" or "mean ± sd".
The script compares two networks within each dataset using Welch's t-test:
  default: Network=PVCP2 vs Network=PoseScript

Outputs:
  welch_results_sheet2.csv
  welch_results_sheet2.md
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path


NS = {
    "a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
}

METRICS = [
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


def cell_text(cell: ET.Element, shared_strings: list[str]) -> str:
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


def read_sheet(path: Path, sheet_name: str) -> list[list[str]]:
    with zipfile.ZipFile(path) as zf:
        shared_strings: list[str] = []
        if "xl/sharedStrings.xml" in zf.namelist():
            root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
            for item in root.findall("a:si", NS):
                shared_strings.append("".join(t.text or "" for t in item.findall(".//a:t", NS)))

        workbook = ET.fromstring(zf.read("xl/workbook.xml"))
        rels = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
        rel_map = {rel.attrib["Id"]: rel.attrib["Target"] for rel in rels}

        target_path = None
        available = []
        for sheet in workbook.findall("a:sheets/a:sheet", NS):
            name = sheet.attrib["name"]
            available.append(name)
            if name == sheet_name:
                rel_id = sheet.attrib[f"{{{NS['r']}}}id"]
                target_path = normalize_sheet_path(rel_map[rel_id])
                break
        if target_path is None:
            raise ValueError(f"Sheet {sheet_name!r} not found. Available sheets: {available}")

        sheet = ET.fromstring(zf.read(target_path))
        rows: list[list[str]] = []
        for row in sheet.findall("a:sheetData/a:row", NS):
            values: list[str] = []
            last_col = 0
            for cell in row.findall("a:c", NS):
                col_num = col_to_num(cell.attrib.get("r", "")) or last_col + 1
                values.extend([""] * (col_num - last_col - 1))
                values.append(cell_text(cell, shared_strings).strip())
                last_col = col_num
            rows.append(values)
        return rows


def parse_mean_sd(value: str) -> tuple[float, float] | None:
    numbers = re.findall(r"[-+]?(?:\d+\.\d+|\d+)", value)
    if len(numbers) < 2:
        return None
    return float(numbers[0]), float(numbers[1])


def parse_records(rows: list[list[str]]) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    last_dataset = ""
    for row in rows:
        padded = row + [""] * (2 + len(METRICS) - len(row))
        dataset = padded[0].strip() or last_dataset
        network = padded[1].strip()
        if dataset and network and padded[2].strip():
            parsed_metrics = [parse_mean_sd(padded[idx]) for idx in range(2, 2 + len(METRICS))]
            if all(item is not None for item in parsed_metrics):
                record: dict[str, object] = {"Dataset": dataset, "Network": network}
                for metric, pair in zip(METRICS, parsed_metrics):
                    assert pair is not None
                    record[f"{metric}_mean"] = pair[0]
                    record[f"{metric}_sd"] = pair[1]
                records.append(record)
        if padded[0].strip():
            last_dataset = padded[0].strip()
    return records


def betacf(a: float, b: float, x: float) -> float:
    max_iter = 200
    eps = 3e-14
    fpmin = 1e-300
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < fpmin:
        d = fpmin
    d = 1.0 / d
    h = d
    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def regularized_beta(x: float, a: float, b: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    log_bt = (
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log1p(-x)
    )
    bt = math.exp(log_bt)
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * betacf(a, b, x) / a
    return 1.0 - bt * betacf(b, a, 1.0 - x) / b


def student_t_two_sided_p(t_value: float, df: float) -> float:
    if df <= 0 or math.isnan(t_value):
        return math.nan
    x = df / (df + t_value * t_value)
    return min(max(regularized_beta(x, df / 2.0, 0.5), 0.0), 1.0)


def welch_test(mean_a: float, sd_a: float, n_a: int, mean_b: float, sd_b: float, n_b: int) -> tuple[float, float, float]:
    se2_a = sd_a * sd_a / n_a
    se2_b = sd_b * sd_b / n_b
    se = math.sqrt(se2_a + se2_b)
    if se == 0:
        if mean_a == mean_b:
            return 0.0, math.inf, 1.0
        return math.copysign(math.inf, mean_a - mean_b), math.inf, 0.0
    t_value = (mean_a - mean_b) / se
    denominator = (se2_a * se2_a) / (n_a - 1) + (se2_b * se2_b) / (n_b - 1)
    df = (se2_a + se2_b) ** 2 / denominator if denominator > 0 else math.inf
    p_value = student_t_two_sided_p(abs(t_value), df)
    return t_value, df, p_value


def benjamini_hochberg(pvalues: list[float]) -> list[float]:
    m = len(pvalues)
    indexed = sorted(enumerate(pvalues), key=lambda x: (math.inf if math.isnan(x[1]) else x[1]))
    qvalues = [math.nan] * m
    prev = 1.0
    for rank_from_end, (idx, pvalue) in enumerate(reversed(indexed), start=1):
        rank = m - rank_from_end + 1
        if math.isnan(pvalue):
            continue
        qvalue = min(prev, pvalue * m / rank, 1.0)
        qvalues[idx] = qvalue
        prev = qvalue
    return qvalues


def significance(qvalue: float) -> str:
    if math.isnan(qvalue):
        return ""
    if qvalue < 0.001:
        return "***"
    if qvalue < 0.01:
        return "**"
    if qvalue < 0.05:
        return "*"
    return "ns"


def fmt(value: float, digits: int = 4) -> str:
    if value is None or math.isnan(value):
        return ""
    if math.isinf(value):
        return "inf" if value > 0 else "-inf"
    return f"{value:.{digits}f}"


def analyze(records: list[dict[str, object]], baseline: str, candidate: str, n: int) -> list[dict[str, object]]:
    by_dataset: dict[str, dict[str, dict[str, object]]] = {}
    for record in records:
        by_dataset.setdefault(str(record["Dataset"]), {})[str(record["Network"])] = record

    rows: list[dict[str, object]] = []
    for dataset, networks in by_dataset.items():
        if baseline not in networks or candidate not in networks:
            continue
        base = networks[baseline]
        cand = networks[candidate]
        for metric in METRICS:
            base_mean = float(base[f"{metric}_mean"])
            base_sd = float(base[f"{metric}_sd"])
            cand_mean = float(cand[f"{metric}_mean"])
            cand_sd = float(cand[f"{metric}_sd"])
            t_value, df, p_value = welch_test(cand_mean, cand_sd, n, base_mean, base_sd, n)
            delta = cand_mean - base_mean
            improvement = (base_mean - cand_mean) / abs(base_mean) * 100 if abs(base_mean) > 1e-12 else math.nan
            rows.append(
                {
                    "Dataset": dataset,
                    "Metric": metric,
                    "Baseline": baseline,
                    "Candidate": candidate,
                    "N_baseline": n,
                    "N_candidate": n,
                    "Baseline_mean": base_mean,
                    "Baseline_sd": base_sd,
                    "Candidate_mean": cand_mean,
                    "Candidate_sd": cand_sd,
                    "Delta_candidate_minus_baseline": delta,
                    "Relative_improvement_percent": improvement,
                    "Welch_t": t_value,
                    "Welch_df": df,
                    "P_value_two_sided": p_value,
                }
            )
    qvalues = benjamini_hochberg([float(row["P_value_two_sided"]) for row in rows])
    for row, qvalue in zip(rows, qvalues):
        row["Q_value_BH_FDR"] = qvalue
        row["Significance_FDR"] = significance(qvalue)
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("No Welch results to write.")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, object]]) -> None:
    lines = [
        "# Welch Test Results From Sheet2",
        "",
        "Test: two-sided Welch independent-samples t-test from mean +/- sd summary values.",
        "Multiple-comparison correction: Benjamini-Hochberg FDR across all rows.",
        "Delta is Candidate - Baseline. Relative improvement assumes lower is better.",
        "",
        "| Dataset | Metric | Baseline mean +/- sd | Candidate mean +/- sd | Delta | Improvement % | t | df | p | q | Sig |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {dataset} | {metric} | {base} | {cand} | {delta} | {impr} | {t} | {df} | {p} | {q} | {sig} |".format(
                dataset=row["Dataset"],
                metric=row["Metric"],
                base=f"{fmt(row['Baseline_mean'], 3)} +/- {fmt(row['Baseline_sd'], 3)}",
                cand=f"{fmt(row['Candidate_mean'], 3)} +/- {fmt(row['Candidate_sd'], 3)}",
                delta=fmt(row["Delta_candidate_minus_baseline"], 4),
                impr=fmt(row["Relative_improvement_percent"], 2),
                t=fmt(row["Welch_t"], 4),
                df=fmt(row["Welch_df"], 3),
                p=fmt(row["P_value_two_sided"], 4),
                q=fmt(row["Q_value_BH_FDR"], 4),
                sig=row["Significance_FDR"],
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Welch t-tests from data.xlsx Sheet2.")
    parser.add_argument("--input", default="data.xlsx", type=Path)
    parser.add_argument("--sheet", default="Sheet2")
    parser.add_argument("--baseline", default="PoseScript")
    parser.add_argument("--candidate", default="PVCP2")
    parser.add_argument("--n", default=3, type=int, help="Sample size used to compute each mean +/- sd.")
    parser.add_argument("--output-prefix", default="welch_results_sheet2")
    args = parser.parse_args()

    input_path = args.input
    if not input_path.is_absolute():
        input_path = Path(__file__).resolve().parent / input_path
    rows = read_sheet(input_path, args.sheet)
    records = parse_records(rows)
    results = analyze(records, args.baseline, args.candidate, args.n)

    output_dir = input_path.parent
    csv_path = output_dir / f"{args.output_prefix}.csv"
    md_path = output_dir / f"{args.output_prefix}.md"
    write_csv(csv_path, results)
    write_markdown(md_path, results)

    print(f"Loaded {len(records)} summary rows from {input_path}::{args.sheet}")
    print(f"Wrote {len(results)} Welch tests to {csv_path}")
    print(f"Wrote Markdown summary to {md_path}")


if __name__ == "__main__":
    main()
