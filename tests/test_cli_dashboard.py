import json
import subprocess
import sys
from pathlib import Path

import pytest

from evalkit import store
from evalkit.cli import main
from evalkit.dashboard import build, slice_keys

ROOT = Path(__file__).resolve().parents[1]
QS = ROOT / "examples" / "quickstart"


@pytest.fixture
def two_runs(tmp_path, monkeypatch):
    monkeypatch.chdir(ROOT)
    for v in ("v1", "v2"):
        code = main(
            [
                "run",
                "--dataset",
                str(QS / "capitals.jsonl"),
                "--task",
                f"examples.quickstart.app:answer_{v}",
                "--scorer",
                "exact_match",
                "--variant",
                "app",
                "--tag",
                v,
                "--runs-dir",
                str(tmp_path),
            ]
        )
        assert code == 0
    a, b = sorted(store.list_runs(tmp_path))
    return a, b


def test_cli_report_compare_repeats(two_runs, capsys):
    a, b = two_runs
    assert main(["report", str(a), "--slice", "region"]) == 0
    assert "100.0%" in capsys.readouterr().out
    assert main(["compare", str(a), str(b), "--json"]) == 0
    c = json.loads(capsys.readouterr().out)
    assert c["n"] == 24 and len(c["a_only"]) == 6
    assert main(["repeats", str(a)]) == 0
    assert "每次都对 24" in capsys.readouterr().out


def test_cli_gate_exit_codes(two_runs, tmp_path, capsys):
    a, b = two_runs
    summary = tmp_path / "gate.md"
    assert (
        main(["gate", "--baseline", str(a), "--candidate", str(b), "--summary-file", str(summary)])
        == 1
    )
    assert "门禁不通过" in summary.read_text(encoding="utf-8")
    assert main(["gate", "--baseline", str(b), "--candidate", str(a)]) == 0
    with pytest.raises(SystemExit, match="results.jsonl"):
        main(["gate", "--baseline", str(a), "--candidate", str(tmp_path / "missing")])


def test_cli_rescore(two_runs, capsys):
    a, b = two_runs
    assert (
        main(
            ["rescore", str(b), "--dataset", str(QS / "capitals.jsonl"), "--scorer", "exact_match"]
        )
        == 0
    )
    assert (b.parent / f"{b.name}-rescored" / "results.jsonl").exists()


def test_cli_feedback(tmp_path, capsys):
    f = tmp_path / "fb.jsonl"
    f.write_text(
        "\n".join(
            json.dumps(x, ensure_ascii=False)
            for x in [
                {"session_id": "s", "target": 1, "label": "not_risk", "ts": 1},
                {"session_id": "s", "target": 1, "label": "useful", "ts": 2},
                {"session_id": "s", "target": 2, "label": "not_risk", "ts": 3},
            ]
        ),
        encoding="utf-8",
    )
    assert (
        main(
            [
                "feedback",
                "summary",
                str(f),
                "--key",
                "session_id,target",
                "--labels",
                "useful,not_risk",
                "--positive",
                "useful",
            ]
        )
        == 0
    )
    s = json.loads(capsys.readouterr().out)
    assert s["rated"] == 2 and s["positive_rate"] == 0.5


def test_dashboard_builds_self_contained_html(two_runs, tmp_path):
    a, b = two_runs
    out = build([a.parent], tmp_path / "site" / "index.html", title="测试看板")
    html = out.read_text(encoding="utf-8")
    assert "<title>测试看板</title>" in html and "__DATA__" not in html
    blob = html.split('<script id="data" type="application/json">')[1].split("</script>")[0]
    data = json.loads(blob.replace("<\\/", "</"))
    assert len(data["runs"]) == 2
    failed = [r for r in data["runs"][1]["rows"] if not r["p"]]
    assert failed and all("o" in r and "tr" in r for r in failed)
    assert all("o" not in r for r in data["runs"][0]["rows"])  # 通过的题不带输出
    assert "http" not in html.split("<script id=")[0].lower().replace("http-equiv", "")


def test_slice_keys_skip_high_cardinality():
    rows = [{"tags": {"id_like": i, "d": i % 2}, "split": None} for i in range(30)]
    assert slice_keys(rows) == ["d"]


@pytest.mark.parametrize(
    "script", ["examples/datapilot/reproduce.py", "examples/contract_review/reproduce.py"]
)
def test_examples_reproduce_bundled_numbers(script):
    """两个适配示例用自带样例复算，必须和原项目 numbers.md 全部一致（已知差异除外）。"""
    r = subprocess.run(
        [sys.executable, str(ROOT / script)], capture_output=True, text=True, cwd=ROOT
    )
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    assert "0 项不一致" in r.stdout
