import json

from xscope.dashboard import build_grouped_scalar_chart_options, build_text_tables, get_run_label
from xscope.data_manager import RunDataManager


def write_run(tmp_path, folder, meta, **files):
    run_dir = tmp_path / folder
    run_dir.mkdir()
    (run_dir / "meta.json").write_text(json.dumps(meta))
    for name, records in files.items():
        (run_dir / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    return run_dir


def load(tmp_path):
    manager = RunDataManager(str(tmp_path))
    runs = manager.load_runs_metadata()
    for run in runs:
        run['color'] = '#2563eb'
    return manager, runs


def test_group_is_part_of_the_label(tmp_path):
    write_run(tmp_path, "v3_exp002_100_20261014_103000",
              {"experiment_name": "100", "experiment_number": 2, "group": "v3"})
    write_run(tmp_path, "exp001_demo_20261014_102000", {"experiment_name": "demo", "experiment_number": 1})
    _, runs = load(tmp_path)
    assert [get_run_label(run) for run in runs] == ["demo #1", "v3/100 #2"]


def test_epoch_zero_keeps_its_place(tmp_path):
    write_run(tmp_path, "exp001_a_20261014_102000", {"experiment_name": "a", "experiment_number": 1},
              metrics=[{"epoch": 0, "cer": 0.9}, {"epoch": 1, "cer": 0.5}])
    manager, runs = load(tmp_path)
    (chart,) = build_grouped_scalar_chart_options(runs, data_manager=manager)
    assert chart['series'][0]['data'] == [[0, 0.9], [1, 0.5]]


def test_text_values_are_not_plotted(tmp_path):
    write_run(tmp_path, "exp001_a_20261014_102000", {"experiment_name": "a", "experiment_number": 1},
              metrics=[{"epoch": 0, "loss": 1.0, "phase": "warmup"}, {"epoch": 1, "loss": 0.5, "phase": "train"}])
    manager, runs = load(tmp_path)
    charts = build_grouped_scalar_chart_options(runs, data_manager=manager)
    assert [chart['title']['text'] for chart in charts] == ["loss"]


def test_text_table_is_newest_first(tmp_path):
    write_run(tmp_path, "exp001_a_20261014_102000", {"experiment_name": "a", "experiment_number": 1},
              text=[{"epoch": 0, "Good-bye": "bood", "_time": 1.0}, {"epoch": 1, "Good-bye": "Good-bee ", "_time": 2.0}])
    manager, runs = load(tmp_path)
    (table,) = build_text_tables(runs, data_manager=manager)
    assert [column['label'] for column in table['columns']] == ["Epoch", "Good-bye"]
    assert [(row['id'], row['x'], row['c0']) for row in table['rows']] == [(1, 1, "Good-bee "), (0, 0, "bood")]


def test_corrupt_line_is_skipped_but_a_half_written_one_waits(tmp_path):
    run_dir = write_run(tmp_path, "exp001_a_20261014_102000", {"experiment_name": "a", "experiment_number": 1})
    path = run_dir / "metrics.jsonl"
    path.write_text('{"epoch": 0, "loss": 1.0}\nnot json\n{"epoch": 1, "loss": 0.5}\n{"epoch": 2, "lo')
    manager = RunDataManager(str(tmp_path))
    _, records = manager.get_records(str(run_dir), "metrics.jsonl")
    assert [r['epoch'] for r in records] == [0, 1]

    with open(path, "a") as f:
        f.write('ss": 0.25}\n')
    _, records = manager.get_records(str(run_dir), "metrics.jsonl")
    assert [r['epoch'] for r in records] == [0, 1, 2]
