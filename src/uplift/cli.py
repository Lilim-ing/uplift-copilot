"""CLI."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from .pipeline import run as run_pipeline
from .power import minimum_detectable_effect, sample_size

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

app = typer.Typer(add_completion=False, help="uplift-copilot · 增量建模与预算分配")
console = Console()

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "fixtures"


@app.command()
def run(
    budget: float = typer.Option(4_000, help="触达预算（元）"),
    cost: float = typer.Option(8.0, help="单人触达成本（元）"),
    model: str = typer.Option("t_learner", help="t_learner | t_learner_gbm | to_ridge | to_gbm"),
    campaign: str = typer.Option("", help="逗号分隔的 campaign_id，留空表示全部"),
    json_out: bool = typer.Option(False, "--json"),
) -> None:
    """训练、评估并给出预算分配。"""
    campaigns = [c.strip() for c in campaign.split(",") if c.strip()] or None
    result = run_pipeline(DATA, campaign_ids=campaigns, budget=budget, cost_per_touch=cost, model_kind=model)

    if json_out:
        console.print_json(json.dumps(result.to_dict(), ensure_ascii=False, default=str))
        return

    console.print(
        f"[bold]样本 {result.n_rows}[/bold] · 训练 {result.n_train} / 测试 {result.n_test}"
        f" · 处理组占比 {result.treatment_share:.1%}"
    )
    console.print(
        f"朴素读数：处理组 {result.naive['treated_rate']:.2%} vs 对照组 "
        f"{result.naive['control_rate']:.2%}（+{result.naive['naive_lift']:.2%}）"
    )
    console.print(f"Qini [bold]{result.qini}[/bold] · AUUC {result.auuc} · uplift@10% {result.uplift_at_10pct:.2%}")
    if result.note:
        console.print(f"[yellow]{result.note}[/yellow]")

    table = Table(title=f"预算分配（{budget:.0f} 元，单人成本 {cost:.0f} 元）", header_style="bold")
    for col in ("策略", "人数", "花费", "预测增量", "实测增量（95% CI）", "每元增量"):
        table.add_column(col)
    for row in result.allocation:
        if row["observed_uplift"] is None:
            observed = "—"
        else:
            lo, hi = row["observed_ci95"]
            observed = f"{row['observed_uplift']:.2%} [{lo:+.1%}, {hi:+.1%}]"
        table.add_row(
            row["policy"],
            str(row["n"]),
            f"{row['spend']:.0f}",
            f"{row['predicted_uplift']:.2f}",
            observed,
            f"{row['uplift_per_yuan']:.4f}",
        )
    console.print(table)

    best = max(
        (r for r in result.allocation if r["observed_uplift"] is not None),
        key=lambda r: r["observed_uplift"],
        default=None,
    )
    if best:
        lo, hi = best["observed_ci95"]
        caveat = "；区间跨 0，不能据此判定优于随机" if lo is not None and lo <= 0 <= hi else ""
        console.print(
            f"\n[bold]实测点估计最优：{best['policy']}[/bold] "
            f"（{best['observed_uplift']:.2%}，{best['n']} 人，花费 {best['spend']:.0f} 元）"
            f"{caveat}"
        )


@app.command("power")
def power_cmd(
    baseline: float = typer.Option(0.07, help="对照组转化率"),
    mde: float = typer.Option(0.03, help="要检测的绝对增量"),
    alpha: float = typer.Option(0.05),
    power: float = typer.Option(0.8),
) -> None:
    """实验样本量。"""
    result = sample_size(baseline, mde, alpha, power)
    table = Table(show_header=False, header_style="bold")
    for key, value in result.items():
        table.add_row(key, str(value))
    console.print(table)
    console.print(
        f"结论：每组至少 {result['per_group']:,} 人。"
        f"当前每 campaign 约 600 人/组，只能检测约 "
        f"{minimum_detectable_effect(600, baseline, power, alpha):.1%} 的增量。"
    )


@app.command("compare-models")
def compare_models(
    budget: float = typer.Option(4_000),
    cost: float = typer.Option(8.0),
) -> None:
    """四种模型横向对比：Qini 与实测增量。"""
    table = Table(header_style="bold")
    for col in ("模型", "Qini", "uplift@10%", "最优策略", "实测增量"):
        table.add_column(col)
    for kind in ("t_learner", "t_learner_gbm", "to_ridge", "to_gbm"):
        r = run_pipeline(DATA, budget=budget, cost_per_touch=cost, model_kind=kind)
        best = max(
            (x for x in r.allocation if x["observed_uplift"] is not None),
            key=lambda x: x["observed_uplift"],
            default=None,
        )
        table.add_row(
            kind,
            str(r.qini),
            f"{r.uplift_at_10pct:.2%}",
            best["policy"] if best else "—",
            f"{best['observed_uplift']:.2%}" if best else "—",
        )
    console.print(table)


if __name__ == "__main__":
    app()
