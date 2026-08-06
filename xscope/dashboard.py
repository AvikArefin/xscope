import os
import json
import colorsys

from nicegui import app, ui

RUN_PALETTE = [
    '#2563eb',  # Blue
    '#16a34a',  # Green
    '#9333ea',  # Purple
    '#ea580c',  # Orange
    '#0891b2',  # Cyan
    '#e11d48',  # Rose
    '#d97706',  # Amber
    '#4f46e5',  # Indigo
    '#059669',  # Emerald
    '#c026d3',  # Fuchsia
]

LINE_STYLES = ['solid', 'dashed', 'dotted', 'dash-dot']


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

            note_path = os.path.join(folder_path, "note.txt")
            if os.path.isfile(note_path):
                with open(note_path, "r", encoding="utf-8") as f:
                    meta['note'] = f.read()
            else:
                meta['note'] = ""

            runs.append(meta)
    for i, run in enumerate(runs): 
        run['color'] = get_run_color(i)
    return runs


def load_records(run_path: str, filename: str) -> list[dict]:
    """Loads metric records from metrics.jsonl inside an experiment folder."""
    path = os.path.join(run_path, filename)
    records: list[dict] = []
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    return records

def save_run_note(target_run: dict, new_note: str):
    target_run['note'] = new_note
    note_path = os.path.join(target_run['run_path'], 'note.txt')
    try:
        with open(note_path, 'w', encoding='utf-8') as f:
            f.write(new_note)
    except Exception as err:
        print(f"[XSCOPE] Error saving note to {note_path}: {err}")


def get_run_color(run_index: int) -> str:
    return RUN_PALETTE[run_index % len(RUN_PALETTE)]

def get_chart_base_config(font_family: str, chart_title: str, x_key = None, y_key = None, scale: bool = False):
    return {
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
        'xAxis': {
            'type': 'value',
            'name': x_key,
            'scale': scale,
            'nameLocation': 'middle',
            'nameTextStyle': {'color': '#000000'},
            'axisLabel': {'color': '#000000'},
        },
        'yAxis': {
            'type': 'value',
            'name': y_key,
            'scale': scale,
            'nameLocation': 'middle',
            'nameTextStyle': {'color': '#000000'},
            'axisLabel': {'color': '#000000'},
        },
        'series': [],
    }

def get_line_style(base_color: str, line_idx: int, num_lines: int) -> tuple[str, str]:
    """Returns (line_type, series_color) for a line in a chart based on index and total lines."""
    if num_lines == 1:
        return 'solid', base_color

    line_type = LINE_STYLES[line_idx % len(LINE_STYLES)]
    tier = line_idx // len(LINE_STYLES)
    if tier == 0:
        return line_type, base_color

    factor = 0.7 ** tier
    r, g, b = (c / 255.0 for c in bytes.fromhex(base_color.lstrip('#')))
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    r_new, g_new, b_new = colorsys.hls_to_rgb(h, max(0.15, min(0.85, l * factor)), s)
    series_color = f"#{round(r_new * 255):02x}{round(g_new * 255):02x}{round(b_new * 255):02x}"
    return line_type, series_color


def build_grouped_scalar_chart_options(selected_runs: list[dict], x_key: str = "epoch", font_family: str = "sans-serif") -> list[dict]:
    """Formats time-series scalar metrics (metrics.jsonl) into ECharts line plots grouped by metric prefix."""
    if not selected_runs:
        return []

    is_multi_run = len(selected_runs) > 1
    charts: dict[str, dict] = {}

    for run in selected_runs:
        records = load_records(run['run_path'], "metrics.jsonl")
        if not records:
            continue

        exp_name = run.get('experiment_name', '')
        exp_num = run.get('experiment_number', '')
        run_label = f"{exp_name} #{exp_num}" if exp_num != "" else str(exp_name)
        base_color = run['color']

        unique_metric_keys = sorted({key for key in records[0].keys() if key != x_key})

        chart_groups: dict[str, list[str]] = {}
        for key in unique_metric_keys:
            chart_title = key.split('/')[0] if '/' in key else key
            chart_groups.setdefault(chart_title, []).append(key)

        for chart_title, keys_in_chart in chart_groups.items():
            keys_in_chart = sorted(keys_in_chart)
            num_lines = len(keys_in_chart)

            if chart_title not in charts:
                charts[chart_title] = get_chart_base_config(font_family, chart_title, x_key, chart_title, False)

            for line_idx, key in enumerate(keys_in_chart):
                series_name = f"{run_label}: {key}" if is_multi_run else key

                data_points = []
                for i, r in enumerate(records):
                    x_val = r.get(x_key, i + 1)
                    y_val = r.get(key)
                    if y_val is not None:
                        data_points.append([x_val, y_val])

                line_type, series_color = get_line_style(base_color, line_idx, num_lines)

                charts[chart_title]['series'].append({
                    'name': series_name,
                    'type': 'line',
                    'data': data_points,
                    'showSymbol': False,
                    'lineStyle': {
                        'color': series_color,
                        'type': line_type,
                    },
                    'itemStyle': {
                        'color': series_color,
                    },
                })

    return list(charts.values())


def build_2d_chart_options(
    selected_runs: list[dict],  
    font_family: str = "sans-serif",
    equal_aspect: bool = True,
    draw_type: str = "dots",
) -> list[dict]:
    """Formats 2D spatial points/lines (2d.jsonl) into ECharts plots grouped by key prefix."""
    if not selected_runs:
        return []

    is_multi_run = len(selected_runs) > 1
    charts: dict[str, dict] = {}

    for run in selected_runs:
        records = load_records(run['run_path'], "2d.jsonl")
        if not records:
            continue

        exp_name = run.get('experiment_name', '')
        exp_num = run.get('experiment_number', '')
        run_label = f"{exp_name} #{exp_num}" if exp_num != "" else str(exp_name)
        base_color = run['color']

        latest_record = records[-1]
        unique_keys = sorted([k for k in latest_record.keys() if k not in ('epoch', 'step')])

        chart_groups: dict[str, list[str]] = {}
        for key in unique_keys:
            chart_title = key.split('/')[0] if '/' in key else key
            chart_groups.setdefault(chart_title, []).append(key)

        for chart_title, keys_in_chart in chart_groups.items():
            keys_in_chart = sorted(keys_in_chart)
            num_lines = len(keys_in_chart)

            if chart_title not in charts:
                charts[chart_title] = get_chart_base_config(font_family, chart_title, None, None, True)

            for line_idx, key in enumerate(keys_in_chart):
                clean_name = key.split('/', 1)[1] if '/' in key else key
                series_name = f"{run_label}: {clean_name}" if is_multi_run else clean_name

                points = latest_record.get(key, [])
                line_type, series_color = get_line_style(base_color, line_idx, num_lines)

                charts[chart_title]['series'].append({
                    'name': series_name,
                    'type': 'scatter' if draw_type == 'dots' else 'line',
                    'data': points,
                    'showSymbol': True,
                    'symbolSize': 6,
                    'lineStyle': {
                        'color': series_color,
                        'type': line_type,
                    },
                    'itemStyle': {
                        'color': series_color,
                    },
                })

    if equal_aspect:
        for chart in charts.values():
            all_x = []
            all_y = []
            for s in chart['series']:
                for pt in s.get('data', []):
                    if len(pt) >= 2 and pt[0] is not None and pt[1] is not None:
                        all_x.append(pt[0])
                        all_y.append(pt[1])

            if all_x and all_y:
                min_x, max_x = min(all_x), max(all_x)
                min_y, max_y = min(all_y), max(all_y)
                max_span = max(max_x - min_x, max_y - min_y)
                if max_span == 0:
                    max_span = 2.0

                margin = max_span * 0.05
                max_span_padded = max_span + 2 * margin

                mid_x = (min_x + max_x) / 2
                mid_y = (min_y + max_y) / 2

                chart['xAxis']['min'] = round(mid_x - max_span_padded / 2, 2)
                chart['xAxis']['max'] = round(mid_x + max_span_padded / 2, 2)
                chart['yAxis']['min'] = round(mid_y - max_span_padded / 2, 2)
                chart['yAxis']['max'] = round(mid_y + max_span_padded / 2, 2)

                chart['grid'] = {
                    'left': 'center',
                    'top': 70,
                    'width': 280,
                    'height': 280,
                }

    return list(charts.values())


def build_matrix_chart_options(
    selected_runs: list[dict],
    font_family: str = "sans-serif",
) -> list[dict]:
    """Formats matrix records (matrix.jsonl) into ECharts Heatmap plots."""
    if not selected_runs:
        return []

    is_multi_run = len(selected_runs) > 1
    charts: list[dict] = []

    for run in selected_runs:
        records = load_records(run['run_path'], "matrix.jsonl")
        if not records:
            continue

        exp_name = run.get('experiment_name', '')
        exp_num = run.get('experiment_number', '')
        run_label = f"{exp_name} #{exp_num}" if exp_num != "" else str(exp_name)

        latest_record = records[-1]
        step = latest_record.get('step')
        labels = latest_record.get('labels', [])
        matrix = latest_record.get('matrix', [])

        if not matrix:
            continue

        num_rows = len(matrix)
        num_cols = len(matrix[0]) if num_rows > 0 else 0

        if not labels:
            labels = [f"Class {i}" for i in range(max(num_rows, num_cols))]

        heatmap_data = []
        max_val = 0
        for i in range(num_rows):
            for j in range(len(matrix[i])):
                val = matrix[i][j]
                heatmap_data.append([j, i, val])
                if val > max_val:
                    max_val = val

        title_suffix = f" @ STEP {step}" if step is not None else ""
        chart_title = f"{run_label}: MATRIX{title_suffix}" if is_multi_run else f"MATRIX{title_suffix}"

        chart_config = {
            'textStyle': {'fontFamily': font_family},
            'title': {'text': chart_title.upper()},
            'tooltip': {'position': 'top'},
            'toolbox': {
                'feature': {
                    'saveAsImage': {
                        'title': 'Save SVG',
                        'type': 'svg',
                        'backgroundColor': '#ffffff',
                    }
                }
            },
            'grid': {'height': '65%', 'top': '15%'},
            'xAxis': {
                'type': 'category',
                'data': labels,
                'name': 'Predicted',
                'nameLocation': 'middle',
                'nameGap': 25,
                'splitArea': {'show': True},
            },
            'yAxis': {
                'type': 'category',
                'data': labels,
                'name': 'True',
                'nameLocation': 'middle',
                'nameGap': 35,
                'splitArea': {'show': True},
                'inverse': True,
            },
            'visualMap': {
                'min': 0,
                'max': max_val if max_val > 0 else 1,
                'calculable': True,
                'orient': 'horizontal',
                'left': 'center',
                'bottom': '0%',
                'inRange': {
                    'color': ['#f8fafc', '#93c5fd', '#1d4ed8']
                }
            },
            'series': [{
                'name': 'Count',
                'type': 'heatmap',
                'data': heatmap_data,
                'label': {'show': True},
                'emphasis': {
                    'itemStyle': {
                        'shadowBlur': 10,
                        'shadowColor': 'rgba(0, 0, 0, 0.5)'
                    }
                }
            }]
        }
        charts.append(chart_config)

    return charts


def create_dashboard_page(metrics_dir: str = "metrics"):
    """Registers NiceGUI root page layout for metric visualization."""

    @ui.page('/')
    def layout():
        all_runs = load_runs_metadata(metrics_dir)
        selected_map: dict[str, bool] = {r['run_path']: False for r in all_runs}

        ui.add_head_html('''
            <style>
                .compact-input .q-field__control {
                    height: 28px !important;
                    min-height: 28px !important;
                }
                .compact-input .q-field__native {
                    padding-top: 0 !important;
                    padding-bottom: 0 !important;
                    font-size: 13px;
                }
            </style>
        ''')

        # Title Bar
        with ui.header(elevated=False).classes('bg-white text-slate-800 border-b border-slate-200 items-center justify-left h-12 px-4 py-0'):
            ui.button(on_click=lambda: left_drawer.toggle(), icon='menu').props('flat dense color=slate-700')
            ui.label('XSCOPE').classes('font-bold text-sm font-mono tracking-wider text-slate-900')
            ui.space()
            ui.button(on_click=lambda: right_drawer.toggle(), icon='settings').props('flat dense color=slate-700')

        # Main dynamic container for metric charts
        charts_container = ui.element('div').classes('w-full p-4 grid gap-6 grid-cols-1')

        def update_chart():
            selected_runs = [r for r in all_runs if selected_map.get(r['run_path'])]
            charts_container.clear()
            cols = columns_select.value
            charts_container.classes(replace=f'w-full p-4 grid gap-6 grid-cols-1 md:grid-cols-{cols}')
            with charts_container:
                for options in build_grouped_scalar_chart_options(selected_runs, font_family=font_select.value):
                    ui.echart(options, renderer='svg').classes('w-full h-[400px]')
                for options in build_2d_chart_options(
                    selected_runs,
                    font_family=font_select.value,
                    equal_aspect=aspect_2d_toggle.value,
                    draw_type=style_2d_toggle.value,
                ):
                    ui.echart(options, renderer='svg').classes('w-full h-[400px]')
                for options in build_matrix_chart_options(selected_runs, font_family=font_select.value):
                    ui.echart(options, renderer='svg').classes('w-full h-[400px]')


        def select_all():
            for path in selected_map:
                selected_map[path] = True
            update_chart()

        def clear_all():
            for path in selected_map:
                selected_map[path] = False
            update_chart()

        # Left Pane
        with ui.left_drawer(top_corner=True, bottom_corner=True).style('background-color: #edf2f7').classes('p-3 gap-3') as left_drawer:
            with ui.row().classes('w-full gap-2'):
                ui.button('Select All', icon='select_all', on_click=select_all).props('unelevated square no-caps color=white text-color=slate-800').classes('flex-1')
                ui.button('Clear All', icon='clear_all', on_click=clear_all).props('unelevated square no-caps color=white text-color=slate-800').classes('flex-1')

            for run_idx, run in enumerate(all_runs):
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

                with ui.card().classes('w-full p-3 shadow-sm gap-1'):
                    with ui.row().classes('items-center gap-2 no-wrap w-full'):
                        ui.checkbox(
                            on_change=update_chart
                        ).bind_value(selected_map, run['run_path']).props('dense')
                        ui.element('div').style(f'background-color: {run["color"]}').classes('w-3.5 h-3.5 shrink-0')
                        ui.label(title_text).classes('font-bold font-mono text-sm text-slate-900')

                    ui.label(sub_text).classes('font-mono text-xs text-slate-800 font-medium leading-none')

                    ui.input(
                        value=run.get('note', ''), 
                        placeholder='Add a note...', 
                        on_change=lambda e, r=run: save_run_note(r, e.value)
                    ).props('filled dense square').classes('w-full compact-input')


        with ui.right_drawer(top_corner=True, bottom_corner=True).style('background-color: #edf2f7').classes('p-3 gap-3') as right_drawer:
            ui.label('Grid Layout').classes('text-xs text-slate-600')
            columns_select = ui.toggle(
                options={1: '1 Col', 2: '2 Col', 3: '3 Col'},
                value=1,
                on_change=lambda: update_chart(),
            ).props('spread no-caps toggle-color=dark toggle-text-color=white color=white text-color=slate-800 unelevated square').classes('w-full')

            font_select = ui.select(
                options=['sans-serif', 'Times New Roman', 'Arial', 'Courier New'],
                value='sans-serif',
                label='Font Family',
                on_change=lambda: update_chart(),
            ).classes('w-full')

            ui.label('2D Chart Style').classes('text-xs text-slate-600')
            style_2d_toggle = ui.toggle(
                options={'dots': 'Dots', 'lines': 'Lines'},
                value='dots',
                on_change=lambda: update_chart(),
            ).props('spread no-caps toggle-color=dark toggle-text-color=white color=white text-color=slate-800 unelevated square').classes('w-full')

            ui.label('2D Aspect Ratio').classes('text-xs text-slate-600')
            aspect_2d_toggle = ui.toggle(
                options={True: '1:1', False: 'Auto'},
                value=True,
                on_change=lambda: update_chart(),
            ).props('spread no-caps toggle-color=dark toggle-text-color=white color=white text-color=slate-800 unelevated square').classes('w-full')


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
