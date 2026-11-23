"""Smoke test for comparing real training CSV formats."""

from scripts.ve_do_thi import plot


def test_plot_multiple_runs_with_missing_values(tmp_path):
    runs = []
    for seed in (1, 2):
        run = tmp_path / f"seed-{seed}"
        run.mkdir()
        (run / "progress.csv").write_text(
            "global_step,ep_return,x_pos,flag_rate,entropy,approx_kl,clipfrac\n"
            f"4096,,{100 * seed},0,1.2,0.004,0.02\n"
            f"8192,{seed},{200 * seed},0.2,1.1,0.005,0.03\n",
            encoding="utf-8",
        )
        (run / "evals.csv").write_text(
            f"global_step,flag_rate,x_pos\n8192,0.2,{100 * seed} {200 * seed}\n",
            encoding="utf-8",
        )
        runs.append(run)
    output = tmp_path / "comparison.png"
    assert plot(runs, ["gamma=.9", "gamma=.99"], output) == output
    assert output.is_file() and output.stat().st_size > 1000
