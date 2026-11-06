"""回归门禁：候选版本相对基线有没有变差。命令行退出码 0 = 通过，1 = 不通过，可直接放进 CI。

    evalkit gate --baseline runs/<基线> --candidate runs/<候选> --max-drop 0.02 --alpha 0.05

判定规则（任一条触发即不通过）：
1. 显著变差：通过率下降，且 McNemar 精确检验 p < alpha；
2. 降幅超限：通过率下降超过 max_drop（冒烟集小、检验功效低时的兜底）；
3. 覆盖不足：候选缺了基线里的题（没跑完或被花费上限截停），比较不成立；
4. 出错变多：候选的 status=error 比例超过 max_error_rate（可选）；
5. 成本超限：候选总花费超过基线的 max_cost_ratio 倍（可选）。

为什么两条都要：题少时（如 20 题冒烟集）即使掉 10 个点也可能不显著，只靠检验会放过回归；
题多时 1 个点的波动也可能"显著"，只靠阈值又会误报。两者结合，阈值管业务上能容忍多少，检验管是不是噪声。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from evalkit.compare import compare


@dataclass
class GateResult:
    passed: bool
    reasons: list[str] = field(default_factory=list)
    comparison: dict = field(default_factory=dict)

    def render(self) -> str:
        c = self.comparison
        head = "✅ 门禁通过" if self.passed else "❌ 门禁不通过"
        lines = [
            f"### {head}",
            "",
            "| 项 | 基线 | 候选 |",
            "|---|---|---|",
            f"| 变体 | {c.get('variant_a')} | {c.get('variant_b')} |",
            f"| 通过率（共同 {c.get('n')} 题） | {c.get('rate_a', 0):.1%} | {c.get('rate_b', 0):.1%} |",
            f"| 独对题数 | {len(c.get('a_only', []))} | {len(c.get('b_only', []))} |",
            f"| 花费 | ${c.get('cost_a', 0):.4f} | ${c.get('cost_b', 0):.4f} |",
            "",
            f"差值 {c.get('delta', 0):+.1%}，McNemar p = {c.get('p_value', 1):.3f}",
        ]
        if self.reasons:
            lines += ["", *[f"- {r}" for r in self.reasons]]
        return "\n".join(lines)


def gate(
    baseline_rows: list[dict],
    candidate_rows: list[dict],
    variant_baseline: str | None = None,
    variant_candidate: str | None = None,
    max_drop: float = 0.02,
    alpha: float = 0.05,
    max_error_rate: float | None = None,
    max_cost_ratio: float | None = None,
) -> GateResult:
    c = compare(baseline_rows, candidate_rows, variant_baseline, variant_candidate)
    reasons = []
    if c["delta"] < 0 and c["p_value"] < alpha:
        reasons.append(f"显著变差：{c['delta']:+.1%}，p = {c['p_value']:.3f} < {alpha}")
    if -c["delta"] > max_drop + 1e-12:
        reasons.append(f"降幅 {-c['delta']:.1%} 超过允许的 {max_drop:.1%}")
    if c["only_in_a"]:
        reasons.append(f"候选缺少基线中的 {c['only_in_a']} 道题，比较不完整")
    if max_error_rate is not None and c["n"] and c["errors_b"] / c["n"] > max_error_rate:
        reasons.append(f"出错率 {c['errors_b'] / c['n']:.1%} 超过 {max_error_rate:.1%}")
    if (
        max_cost_ratio is not None
        and c["cost_a"] > 0
        and c["cost_b"] > c["cost_a"] * max_cost_ratio
    ):
        reasons.append(f"花费是基线的 {c['cost_b'] / c['cost_a']:.2f} 倍，超过 {max_cost_ratio} 倍")
    return GateResult(not reasons, reasons, c)
