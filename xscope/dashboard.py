import os
import json
from nicegui import app, ui


def load_runs_metadata(dir: str = "metrics") -> list[dict]:
    runs: list[dict] = []
    if not os.path.exists(dir):
        return runs

    for folder in sorted(os.listdir(dir)):
        folder_path = os.path.join(dir, folder)
        meta_path = os.path.join(folder_path, "meta.json")
        if os.path.isfile(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            meta['run_path'] = folder_path
            runs.append(meta)

    return runs


def load_records(run_path: str) -> list[dict]:
    """Loads metric records from metrics.jsonl inside an experiment folder."""
    metrics_path = os.path.join(run_path, "metrics.jsonl")
    records: list[dict] = []
    if os.path.isfile(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    return records


def build_grouped_scalar_chart_options(selected_runs: list[dict], x_key: str = "epoch", font_family: str = "sans-serif") -> list[dict]:
    """Formats time-series scalar metrics (metrics.jsonl) into ECharts line plots grouped by metric prefix."""
    if not selected_runs:
        return []

    is_multi_run = len(selected_runs) > 1
    charts: dict[str, dict] = {}

    for run in selected_runs:
        records = load_records(run['run_path'])
        if not records:
            continue

        exp_name = run.get('experiment_name', '')
        exp_num = run.get('experiment_number', '')
        run_label = f"{exp_name} #{exp_num}" if exp_num != "" else str(exp_name)

        # Take a record and filter all unique "key"s for this run
        unique_metric_keys = sorted({key for key in records[0].keys() if key != x_key})

        for key in unique_metric_keys:
            chart_title = key.split('/')[0] if '/' in key else key
            if chart_title not in charts.keys():
                charts[chart_title] = {
                    'textStyle': {'fontFamily': font_family},
                    'title': {'text': chart_title.upper()},
                    'tooltip': {'trigger': 'axis'},
                    'legend': {'top': '8%'},
                    'toolbox': {
                        'feature': {
                            'saveAsImage': {
                                'title': 'Save SVG',
                                'type': 'svg',
                                'backgroundColor': '#ffffff',
                            }
                        }
                    },
                    'xAxis': {'type': 'value', 'name': x_key, 'nameLocation': 'middle'},
                    'yAxis': {'type': 'value', 'name': chart_title, 'nameLocation': 'middle'},
                    'series': [],
                }

            series_name = f"{run_label}: {key}" if is_multi_run else key

            data_points = []
            for i, r in enumerate(records):
                x_val = r.get(x_key, i + 1)
                y_val = r.get(key)
                if y_val is not None:
                    data_points.append([x_val, y_val])

            charts[chart_title]['series'].append({
                'name': series_name,
                'type': 'line',
                'data': data_points,
                'showSymbol': False,
            })

    return list(charts.values())


def create_dashboard_page(metrics_dir: str = "metrics"):
    """Registers NiceGUI root page layout for metric visualization."""

    @ui.page('/')
    def layout():
        all_runs = load_runs_metadata(metrics_dir)
        selected_runs: list[dict] = []

        # Title Bar
        with ui.header(elevated=True).style('background-color: #3874c8').classes('items-center justify-left'):
            ui.button(on_click=lambda: left_drawer.toggle(), icon='menu').props('flat color=white')
            ui.label('XSCOPE')
            ui.space()
            ui.button(on_click=lambda: right_drawer.toggle(), icon='settings').props('flat color=white')

        # Main dynamic container for metric charts
        charts_container = ui.element('div').classes('w-full p-4 grid gap-6 grid-cols-1')

        def update_chart():
            charts_container.clear()
            cols = columns_select.value
            charts_container.classes(replace=f'w-full p-4 grid gap-6 grid-cols-1 md:grid-cols-{cols}')
            with charts_container:
                for options in build_grouped_scalar_chart_options(selected_runs, font_family=font_select.value):
                    ui.echart(options, renderer='svg').classes('w-full h-[400px]')

        checkboxes: dict[str, ui.checkbox] = {}

        def select_all():
            selected_runs.clear()
            selected_runs.extend(all_runs)
            for cb in checkboxes.values():
                cb.value = True
            update_chart()

        def clear_all():
            selected_runs.clear()
            for cb in checkboxes.values():
                cb.value = False
            update_chart()

        # Left Pane
        with ui.left_drawer(top_corner=True, bottom_corner=True).style('background-color: #edf2f7').classes('p-3 gap-3') as left_drawer:
            with ui.row():
                ui.button('Select All', on_click=select_all)
                ui.button('Clear All', on_click=clear_all)

            for run in all_runs:
                folder_name = run['run_path']
                exp_name = run.get('experiment_name', '')
                exp_num = run.get('experiment_number', '')
                title_text = f"{exp_name} #{exp_num}" if exp_num != "" else str(exp_name)

                ts_raw = str(run.get('timestamp', ''))
                if len(ts_raw) == 15 and '_' in ts_raw:
                    ts_fmt = f"{ts_raw[:4]}-{ts_raw[4:6]}-{ts_raw[6:8]} {ts_raw[9:11]}:{ts_raw[11:13]}:{ts_raw[13:15]}"
                else:
                    ts_fmt = ts_raw
                commit = run.get('git_commit', 'unknown')
                sub_text = f"{ts_fmt} • {commit}" if ts_fmt else str(commit)

                note_text = run.get('note') or 'Add a note...'

                def make_handler(target_run):
                    def handler(e):
                        if e.value:
                            if target_run not in selected_runs:
                                selected_runs.append(target_run)
                        else:
                            if target_run in selected_runs:
                                selected_runs.remove(target_run)
                        update_chart()
                    return handler

                with ui.card().classes('w-full p-3 shadow-sm gap-1'):
                    with ui.row().classes('items-center gap-2 no-wrap w-full'):
                        cb = ui.checkbox(
                            value=(run in selected_runs),
                            on_change=make_handler(run)
                        ).props('dense')
                        checkboxes[folder_name] = cb
                        ui.element('div').classes('w-3.5 h-3.5 bg-blue-500 shrink-0')
                        ui.label(title_text).classes('font-bold font-mono text-sm text-slate-900')

                    ui.label(sub_text).classes('font-mono text-xs text-slate-800 font-medium leading-none')

                    with ui.element('div').classes('w-full bg-slate-200 text-slate-800 px-2 py-1.5 font-mono text-xs mt-1'):
                        ui.label(note_text)


        with ui.right_drawer(top_corner=True, bottom_corner=True).style('background-color: #edf2f7').classes('p-3 gap-3') as right_drawer:
            columns_select = ui.select(
                options={1: 'Single Column', 2: 'Double Column', 3: 'Triple Column'},
                value=1,
                label='Grid Layout',
                on_change=lambda: update_chart(),
            ).classes('w-full')

            font_select = ui.select(
                options=['sans-serif', 'Times New Roman', 'Arial', 'Courier New'],
                value='sans-serif',
                label='Font Family',
                on_change=lambda: update_chart(),
            ).classes('w-full')


        with ui.page_scroller(position='bottom-right', x_offset=20, y_offset=20):
            ui.button('Scroll to Top')


def run_dashboard(metrics_dir: str = "metrics", **kwargs):
    """Starts the mtrick dashboard web server."""
    create_dashboard_page(metrics_dir=metrics_dir)
    kwargs.setdefault('title', 'xscope')
    kwargs.setdefault('favicon', 'data:image/x-icon;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII=')
    kwargs.setdefault('show_welcome_message', False)

    port = kwargs.get('port', 8080)
    host = kwargs.get('host', 'localhost')
    if host in ('0.0.0.0', ''):
        host = 'localhost'

    @app.on_startup
    def _print_url():
        print(f"http://{host}:{port}")

    ui.run(**kwargs)
