import os
import html
import json
import colorsys

from nicegui import app, ui
from xscope.data_manager import RunDataManager
from xscope.diagnostics import run_diagnostics_for_runs
from xscope.diagnostics.core import run_label as get_run_label
from xscope.diagnostics.series import is_number

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


def load_records(run_path: str, filename: str, data_manager: RunDataManager | None = None) -> list[dict]:
    """Loads metric records from a jsonl file inside an experiment folder."""
    dm = data_manager or RunDataManager()
    _, records = dm.get_records(run_path, filename)
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

def get_chart_base_config(
    font_family: str,
    chart_title: str,
    x_key = None,
    y_key = None,
    scale: bool = False,
    renderer: str = "svg",
    title_size: int = 18,
    x_axis_size: int = 12,
    y_axis_size: int = 12,
    show_legend: bool = True,
):
    return {
        'textStyle': {'fontFamily': font_family},
        'title': {
            'text': chart_title,
            'textStyle': {'color': '#000000', 'fontSize': title_size, 'fontWeight': 'normal'},
        },
        'tooltip': {'trigger': 'axis'},
        'legend': {'top': '8%', 'show': show_legend},
        'toolbox': {
            'feature': {
                'saveAsImage': {
                    'title': 'Save SVG' if renderer == 'svg' else 'Save PNG',
                    'type': 'svg' if renderer == 'svg' else 'png',
                    'pixelRatio': 400 / 96,
                    'backgroundColor': '#ffffff',
                }
            }
        },
        'grid': {
            'left': 48,
            'right': 16,
            'top': 65,
            'bottom': 35,
            'containLabel': True,
        },
        'xAxis': {
            'type': 'value',
            'name': x_key.capitalize() if isinstance(x_key, str) else x_key,
            'scale': scale,
            'nameLocation': 'middle',
            'nameGap': 25,
            'nameTextStyle': {'color': '#000000', 'fontSize': x_axis_size},
            'axisLabel': {'color': '#000000', 'fontSize': x_axis_size},
        },
        'yAxis': {
            'type': 'value',
            'name': y_key,
            'scale': scale,
            'nameLocation': 'middle',
            'nameGap': 35,
            'nameTextStyle': {'color': '#000000', 'fontSize': y_axis_size},
            'axisLabel': {'color': '#000000', 'fontSize': y_axis_size},
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
    try:
        r, g, b = (c / 255.0 for c in bytes.fromhex(base_color.lstrip('#')))
        h, l, s = colorsys.rgb_to_hls(r, g, b)
        r_new, g_new, b_new = colorsys.hls_to_rgb(h, max(0.15, min(0.85, l * factor)), s)
        series_color = f"#{round(r_new * 255):02x}{round(g_new * 255):02x}{round(b_new * 255):02x}"
    except Exception:
        series_color = base_color
    return line_type, series_color


def build_grouped_scalar_chart_options(
    selected_runs: list[dict],
    x_key: str = "epoch",
    font_family: str = "sans-serif",
    data_manager: RunDataManager | None = None,
    renderer: str = "svg",
    color_override: str | None = None,
    title_size: int = 18,
    x_axis_size: int = 12,
    y_axis_size: int = 12,
    show_legend: bool = True,
    scale: bool = False,
) -> list[dict]:
    """Formats time-series scalar metrics (metrics.jsonl) into ECharts line plots grouped by metric prefix."""
    if not selected_runs:
        return []

    is_multi_run = len(selected_runs) > 1
    charts: dict[str, dict] = {}
    ignored_keys = {x_key.lower(), 'epoch', 'step', 'timestamp', 'time', 'wall_time'}

    for run in selected_runs:
        records = load_records(run['run_path'], "metrics.jsonl", data_manager=data_manager)
        if not records:
            continue

        run_label = get_run_label(run)
        base_color = color_override or run['color']

        # Only numbers can be plotted; mtrick also accepts text values such as {"phase": "warmup"}
        numeric_keys = {key for r in records for key, value in r.items() if is_number(value)}
        unique_metric_keys = sorted({key for key in numeric_keys if key.lower() not in ignored_keys and not key.startswith('_')})

        chart_groups: dict[str, list[str]] = {}
        for key in unique_metric_keys:
            chart_title = key.split('/')[0] if '/' in key else key
            chart_groups.setdefault(chart_title, []).append(key)

        for chart_title, keys_in_chart in chart_groups.items():
            keys_in_chart = sorted(keys_in_chart)
            num_lines = len(keys_in_chart)

            if chart_title not in charts:
                charts[chart_title] = get_chart_base_config(
                    font_family, chart_title, x_key, chart_title, scale,
                    renderer=renderer, title_size=title_size,
                    x_axis_size=x_axis_size, y_axis_size=y_axis_size,
                    show_legend=show_legend,
                )

            for line_idx, key in enumerate(keys_in_chart):
                series_name = f"{run_label}: {key}" if is_multi_run else key

                data_points = []
                for i, r in enumerate(records):
                    # 'is None' rather than 'or': epoch 0 is a real x value
                    x_val = next((r[k] for k in (x_key, x_key.lower(), x_key.capitalize()) if r.get(k) is not None), i + 1)
                    y_val = r.get(key)
                    if is_number(y_val):
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
    data_manager: RunDataManager | None = None,
    renderer: str = "svg",
    color_override: str | None = None,
    title_size: int = 18,
    x_axis_size: int = 12,
    y_axis_size: int = 12,
    show_legend: bool = True,
) -> list[dict]:
    """Formats 2D spatial points/lines (2d.jsonl) into ECharts plots grouped by key prefix."""
    if not selected_runs:
        return []

    is_multi_run = len(selected_runs) > 1
    charts: dict[str, dict] = {}
    ignored_2d_keys = {'epoch', 'step', 'timestamp', 'time', 'wall_time'}

    for run in selected_runs:
        records = load_records(run['run_path'], "2d.jsonl", data_manager=data_manager)
        if not records:
            continue

        run_label = get_run_label(run)
        base_color = color_override or run['color']

        latest_record = records[-1]
        unique_keys = sorted([k for k in latest_record.keys() if k.lower() not in ignored_2d_keys and not k.startswith('_')])

        chart_groups: dict[str, list[str]] = {}
        for key in unique_keys:
            chart_title = key.split('/')[0] if '/' in key else key
            chart_groups.setdefault(chart_title, []).append(key)

        for chart_title, keys_in_chart in chart_groups.items():
            keys_in_chart = sorted(keys_in_chart)
            num_lines = len(keys_in_chart)

            if chart_title not in charts:
                charts[chart_title] = get_chart_base_config(
                    font_family, chart_title, None, None, True,
                    renderer=renderer, title_size=title_size,
                    x_axis_size=x_axis_size, y_axis_size=y_axis_size,
                    show_legend=show_legend,
                )

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


MATRIX_LABEL_MAX_CHARS = 14   # longer class names are truncated with an ellipsis (full name in tooltip)
MATRIX_CELL_LABEL_MAX_CLASSES = 15   # beyond this, counts are only shown in the tooltip
MATRIX_ALL_TICKS_MAX_CLASSES = 40    # beyond this, ECharts thins out the axis labels


def matrix_chart_size(num_classes: int) -> int:
    """Max side length (px) of the square matrix chart; grows with the number of classes."""
    return max(420, min(900, 260 + 22 * num_classes))


def build_matrix_chart_options(
    selected_runs: list[dict],
    font_family: str = "sans-serif",
    data_manager: RunDataManager | None = None,
    renderer: str = "svg",
    title_size: int = 18,
    x_axis_size: int = 12,
    y_axis_size: int = 12,
) -> list[dict]:
    """Formats matrix records (matrix.jsonl) into square ECharts heatmaps.

    The plot area is kept square: the chart container is rendered with a 1:1 aspect ratio
    (see `_layout`), and the grid margins are balanced so that left + right == top + bottom.
    """
    if not selected_runs:
        return []

    is_multi_run = len(selected_runs) > 1
    charts: list[dict] = []

    for run in selected_runs:
        records = load_records(run['run_path'], "matrix.jsonl", data_manager=data_manager)

        if not records:
            continue

        run_label = get_run_label(run)

        latest_record = records[-1]
        step = latest_record.get('step')
        labels = latest_record.get('labels', [])
        matrix = latest_record.get('matrix', [])

        if not matrix or not isinstance(matrix, list) or not all(isinstance(row, list) for row in matrix):
            continue

        num_classes = max(len(matrix), max(len(row) for row in matrix))
        if not isinstance(labels, list) or len(labels) != num_classes:
            labels = [f"Class {i}" for i in range(num_classes)]
        labels = [str(label) for label in labels]

        heatmap_data = []
        max_val = 0
        for i, row in enumerate(matrix):
            row_total = sum(v for v in row if isinstance(v, (int, float)))
            for j, val in enumerate(row):
                if not isinstance(val, (int, float)):
                    continue
                row_pct = round(100 * val / row_total, 1) if row_total else 0
                heatmap_data.append([j, i, val, row_pct])
                max_val = max(max_val, val)

        title_suffix = f" @ STEP {step}" if step is not None else ""
        chart_title = f"{run_label}: MATRIX{title_suffix}" if is_multi_run else f"MATRIX{title_suffix}"

        # --- Layout: estimate label extents so the plot area can be made exactly square. ---
        label_font = min(x_axis_size, y_axis_size)
        if num_classes > MATRIX_CELL_LABEL_MAX_CLASSES:
            label_font = max(8, min(label_font, int(label_font * 24 / num_classes) + 4))
        longest = min(MATRIX_LABEL_MAX_CHARS, max((len(label) for label in labels), default=1))
        label_px = int(longest * label_font * 0.62) + 8
        rotate_x = num_classes > 8 or longest > 6
        x_label_px = label_px if rotate_x else label_font + 8
        axis_name_px = 28
        visual_map_px = 72

        left = axis_name_px + label_px
        right = visual_map_px
        top = title_size + 30
        bottom = axis_name_px + x_label_px
        # Balance margins so a square container yields a square plot area.
        horizontal, vertical = left + right, top + bottom
        if horizontal > vertical:
            top += (horizontal - vertical) // 2
            bottom += horizontal - vertical - (horizontal - vertical) // 2
        else:
            right += vertical - horizontal

        show_cell_labels = num_classes <= MATRIX_CELL_LABEL_MAX_CLASSES
        if show_cell_labels and max_val > 0:
            # White text on dark cells, dark text on light cells.
            heatmap_data = [
                {'value': item, 'label': {'color': '#ffffff' if item[2] / max_val > 0.55 else '#0f172a'}}
                for item in heatmap_data
            ]
        cell_font = 12 if num_classes <= 6 else 10 if num_classes <= 10 else 9
        label_interval = 0 if num_classes <= MATRIX_ALL_TICKS_MAX_CLASSES else 'auto'
        axis_label = {
            'fontSize': label_font,
            'interval': label_interval,
            'width': label_px - 8,
            'overflow': 'truncate',
            'ellipsis': '…',
        }
        tooltip_labels = json.dumps([html.escape(label) for label in labels])

        chart_config = {
            '_layout': {'square': True, 'max_size': matrix_chart_size(num_classes)},
            '_key': f"matrix:{run['run_path']}",   # the title changes with every step, the chart stays the same one
            'textStyle': {'fontFamily': font_family},
            'title': {
                'text': chart_title,
                'textStyle': {'color': '#000000', 'fontSize': title_size, 'fontWeight': 'normal'},
            },
            'tooltip': {
                'position': 'top',
                ':formatter': (
                    f"(p) => {{ const L = {tooltip_labels}; const v = p.value; "
                    "return 'True: <b>' + L[v[1]] + '</b><br>Predicted: <b>' + L[v[0]] + "
                    "'</b><br>Count: ' + v[2] + ' (' + v[3] + '% of true class)'; }"
                ),
            },
            'toolbox': {
                'feature': {
                    'saveAsImage': {
                        'title': 'Save SVG' if renderer == 'svg' else 'Save PNG',
                        'type': 'svg' if renderer == 'svg' else 'png',
                        'pixelRatio': 400 / 96,
                        'backgroundColor': '#ffffff',
                    }
                }
            },
            'grid': {
                'left': left,
                'right': right,
                'top': top,
                'bottom': bottom,
                'containLabel': False,
            },
            'xAxis': {
                'type': 'category',
                'data': labels,
                'name': 'Predicted',
                'nameLocation': 'middle',
                'nameGap': x_label_px + 8,
                'nameTextStyle': {'fontSize': x_axis_size},
                'axisLabel': {**axis_label, 'rotate': 90 if rotate_x else 0},
                'axisTick': {'show': False},
            },
            'yAxis': {
                'type': 'category',
                'data': labels,
                'name': 'True',
                'nameLocation': 'middle',
                'nameGap': label_px + 8,
                'nameTextStyle': {'fontSize': y_axis_size},
                'axisLabel': axis_label,
                'axisTick': {'show': False},
                'inverse': True,
            },
            'visualMap': {
                'min': 0,
                'max': max_val if max_val > 0 else 1,
                'dimension': 2,
                'calculable': True,
                'orient': 'vertical',
                'right': 8,
                'top': 'middle',
                'itemHeight': 140,
                'inRange': {
                    'color': ['#f8fafc', '#93c5fd', '#1d4ed8']
                }
            },
            'series': [{
                'name': 'Count',
                'type': 'heatmap',
                'data': heatmap_data,
                'progressive': 0,  # draw every cell in one pass (large matrices otherwise render in chunks)
                'label': {
                    'show': show_cell_labels,
                    'fontSize': cell_font,
                    ':formatter': '(p) => String(p.value[2])',
                },
                'itemStyle': {
                    'borderColor': '#ffffff',
                    'borderWidth': 1 if num_classes <= MATRIX_ALL_TICKS_MAX_CLASSES else 0,
                },
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


TEXT_TABLE_MAX_ROWS = 1000   # newest records kept per run; the table scrolls (see .xscope-text)


def build_text_tables(selected_runs: list[dict], data_manager: RunDataManager | None = None) -> list[dict]:
    """Formats logged text (text.jsonl) into one table per run: a row per record, newest first."""
    is_multi_run = len(selected_runs) > 1
    tables: list[dict] = []

    for run in selected_runs:
        records = load_records(run['run_path'], "text.jsonl", data_manager=data_manager)
        if not records:
            continue

        first = max(0, len(records) - TEXT_TABLE_MAX_ROWS)
        recent = records[first:][::-1]
        x_key = next((k for k in ('epoch', 'step') if any(k in r for r in recent)), None)
        text_keys = list(dict.fromkeys(
            k for r in recent for k in r if k != x_key and not k.startswith('_')
        ))

        # Columns are numbered: the keys are free text and may hold any character
        columns = [{'name': f'c{i}', 'label': key, 'field': f'c{i}', 'align': 'left'} for i, key in enumerate(text_keys)]
        if x_key:
            columns.insert(0, {'name': 'x', 'label': x_key.capitalize(), 'field': 'x', 'align': 'left'})
        # 'id' is the record's position in the file, so a row keeps its id when newer rows arrive above it
        rows = [
            {'id': len(records) - 1 - row_idx, 'x': r.get(x_key, '') if x_key else '',
             **{f'c{i}': str(r.get(key, '')) for i, key in enumerate(text_keys)}}
            for row_idx, r in enumerate(recent)
        ]

        tables.append({
            'key': run['run_path'],
            'title': f"{get_run_label(run)}: TEXT" if is_multi_run else "TEXT",
            'columns': columns,
            'rows': rows,
        })

    return tables


def chart_key(options: dict) -> str:
    """What identifies a chart between refreshes: its '_key' if it has one, else its title."""
    return options.get('_key') or options.get('title', {}).get('text', '')


def create_dashboard_page(metrics_dir: str = "metrics"):
    """Registers NiceGUI root page layout for metric visualization."""

    @ui.page('/')
    def layout():
        data_manager = RunDataManager(metrics_dir)
        all_runs = data_manager.load_runs_metadata()
        for i, run in enumerate(all_runs):
            run['color'] = get_run_color(i)

        selected_map: dict[str, bool] = {r['run_path']: False for r in all_runs}
        active_echarts: dict[str, ui.echart] = {}
        active_text_tables: dict[str, ui.table] = {}   # by run path
        rendered_selection: list[str] = []   # run paths the charts were last drawn for

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
                .q-checkbox__bg {
                    border-radius: 0 !important;
                }
                .q-card {
                    border-radius: 0 !important;
                }
                /* Logged text: a few rows tall, scrolls inside itself, header stays in view */
                .xscope-text {
                    max-height: 320px;
                }
                .xscope-text td {
                    white-space: pre-wrap;
                    font-family: monospace;
                }
                .xscope-text thead tr th {
                    position: sticky;
                    top: 0;
                    z-index: 1;
                    background-color: #ffffff;
                }
            </style>
        ''')

        # Title Bar
        with ui.header(elevated=False).classes('bg-transparent text-slate-800 items-center justify-left h-12 px-4 py-0'):
            ui.button(on_click=lambda: left_drawer.toggle(), icon='menu').props('flat dense color=slate-700')
            ui.label('XSCOPE').classes('font-bold text-sm font-mono tracking-wider text-slate-900')
            ui.space()
            ui.button(on_click=lambda: toggle_diagnostics(), icon='health_and_safety').props('flat dense color=slate-700').tooltip('Diagnostics')
            ui.button(on_click=lambda: right_drawer.toggle(), icon='settings').props('flat dense color=slate-700')

        # Main dynamic container for metric charts
        charts_container = ui.element('div').classes('w-full p-4 grid gap-6 grid-cols-1')
        # Logged text, one table per run, below the charts
        text_container = ui.column().classes('w-full px-4 pb-4 gap-2')

        # Bottom Drawer for Diagnostics (Method 2: ui.footer)
        # Inline flex-direction: NiceGUI's .nicegui-footer rule (row) overrides the flex-col class.
        with ui.footer(value=False).style('background-color: #f8fafc; max-height: 40vh; overflow-y: auto; overflow-x: hidden; flex-direction: column; flex-wrap: nowrap;').classes('border-t border-slate-300 p-0 text-slate-800 flex flex-col gap-0 shadow-lg') as bottom_drawer:
            with ui.row().classes('w-full items-center gap-2 border-b border-slate-200 px-4 py-2'):
                ui.icon('health_and_safety').classes('text-base text-slate-700')
                ui.label('DIAGNOSTICS').classes('font-bold font-mono text-xs tracking-wider text-slate-900')
                ui.space()
                ui.button(on_click=lambda: bottom_drawer.toggle(), icon='close').props('flat dense color=slate-700')

            diagnostics_container = ui.column().classes('w-full p-0 gap-0')

        def refresh_diagnostics():
            diagnostics_container.clear()
            selected_runs = [r for r in all_runs if selected_map.get(r['run_path'])]
            if not selected_runs:
                with diagnostics_container:
                    ui.label('No runs selected. Select runs in the left panel.').classes('font-mono text-xs text-slate-500 italic px-4 py-2')
                return
            results = run_diagnostics_for_runs(selected_runs, data_manager)
            if not results:
                with diagnostics_container:
                    with ui.row().classes('w-full px-4 py-2.5 bg-emerald-50 text-emerald-800 items-center gap-2.5'):
                        ui.icon('check_circle').classes('text-base shrink-0')
                        ui.label('All diagnostics passed. No anomalies detected.').classes('font-mono text-xs font-semibold')
                return

            with diagnostics_container:
                for res in results:
                    if res.severity == 'critical':
                        bg_cls = 'bg-red-50 text-red-800'
                        icon_name = 'error'
                    elif res.severity == 'warning':
                        bg_cls = 'bg-amber-50 text-amber-800'
                        icon_name = 'warning'
                    else:
                        bg_cls = 'bg-blue-50 text-blue-800'
                        icon_name = 'info'

                    with ui.row(wrap=False).classes(f'w-full px-4 py-2.5 items-start gap-2.5 {bg_cls}'):
                        ui.icon(icon_name).classes('text-base shrink-0 mt-0.5')
                        with ui.column().classes('gap-0 flex-1 min-w-0'):
                            title_line = f"[{res.run_name}] {res.title}"
                            if res.epoch is not None:
                                title_line += f" (Step {res.epoch})" if res.metric == 'matrix' else f" (Epoch {res.epoch})"
                            ui.label(title_line).classes('font-bold font-mono text-xs break-words w-full')
                            ui.label(res.message).classes('font-mono text-xs font-medium break-words w-full')
                            if res.suggested_action:
                                ui.label(f"Suggestion: {res.suggested_action}").classes('font-mono text-xs font-medium break-words w-full')

        def toggle_diagnostics():
            bottom_drawer.toggle()
            if bottom_drawer.value:
                refresh_diagnostics()

        def get_all_chart_options(selected_runs: list[dict]) -> list[dict]:
            renderer_val = renderer_toggle.value
            color_override_val = color_override_input.value or None
            font_val = font_select.value
            t_size = int(title_font_size.value or 18)
            x_size = int(x_font_size.value or 12)
            y_size = int(y_font_size.value or 12)
            show_leg = legend_toggle.value
            scale_val = y_scale_toggle.value
            return [
                *build_grouped_scalar_chart_options(
                    selected_runs, font_family=font_val, data_manager=data_manager,
                    renderer=renderer_val, color_override=color_override_val,
                    title_size=t_size, x_axis_size=x_size, y_axis_size=y_size,
                    show_legend=show_leg, scale=scale_val,
                ),
                *build_2d_chart_options(
                    selected_runs, font_family=font_val, equal_aspect=aspect_2d_toggle.value,
                    draw_type=style_2d_toggle.value, data_manager=data_manager,
                    renderer=renderer_val, color_override=color_override_val,
                    title_size=t_size, x_axis_size=x_size, y_axis_size=y_size,
                    show_legend=show_leg,
                ),
                *build_matrix_chart_options(
                    selected_runs, font_family=font_val, data_manager=data_manager,
                    renderer=renderer_val,
                    title_size=t_size, x_axis_size=x_size, y_axis_size=y_size,
                ),
            ]

        def render_all_charts():
            selected_runs = [r for r in all_runs if selected_map.get(r['run_path'])]
            rendered_selection[:] = [r['run_path'] for r in selected_runs]
            charts_container.clear()
            active_echarts.clear()

            renderer_val = renderer_toggle.value
            cols = columns_select.value
            charts_container.classes(replace=f'w-full p-4 grid gap-6 grid-cols-1 md:grid-cols-{cols}')
            with charts_container:
                for options in get_all_chart_options(selected_runs):
                    key = chart_key(options)
                    options.pop('_key', None)
                    layout = options.pop('_layout', None)
                    widget = ui.echart(options, renderer=renderer_val)
                    if layout and layout.get('square'):
                        # Square container + balanced grid margins => square plot area at any width.
                        widget.classes('w-full').style(
                            f"height: auto; aspect-ratio: 1 / 1; max-width: {layout['max_size']}px; margin: 0 auto;"
                        )
                    else:
                        widget.classes('w-full h-[400px]')
                    if key:
                        active_echarts[key] = widget
            render_text_tables(selected_runs)

        def render_text_tables(selected_runs: list[dict], rebuild: bool = True):
            tables = build_text_tables(selected_runs, data_manager)

            # Same tables as before: only the rows change, so a table being read keeps its scroll position
            same_tables = [t['key'] for t in tables] == list(active_text_tables) and all(
                active_text_tables[t['key']].columns == t['columns'] for t in tables)
            if same_tables and not rebuild:
                for table in tables:
                    active_text_tables[table['key']].rows = table['rows']
                return

            text_container.clear()
            active_text_tables.clear()
            with text_container:
                for table in tables:
                    ui.label(table['title']).classes('text-lg text-black')
                    active_text_tables[table['key']] = ui.table(
                        columns=table['columns'], rows=table['rows'], row_key='id', pagination=0,
                    ).props('dense flat square bordered virtual-scroll hide-bottom').classes('w-full xscope-text')

        def update_charts_incremental():
            selected_runs = [r for r in all_runs if selected_map.get(r['run_path'])]
            if not selected_runs:
                return

            all_opts = get_all_chart_options(selected_runs)
            if {chart_key(opt) for opt in all_opts} - {''} != set(active_echarts.keys()):
                render_all_charts()
                return

            for options in all_opts:
                widget = active_echarts.get(chart_key(options))
                if widget is None:
                    continue
                if options.pop('_key', None):
                    # A matrix: its title, colour range and labels all follow the newest record
                    options.pop('_layout', None)
                    widget.options.clear()
                    widget.options.update(options)
                else:
                    widget.options['series'] = options['series']
                    if 'xAxis' in options and 'min' in options['xAxis']:
                        widget.options['xAxis'] = options['xAxis']
                    if 'yAxis' in options and 'min' in options['yAxis']:
                        widget.options['yAxis'] = options['yAxis']
                widget.update()
            render_text_tables(selected_runs, rebuild=False)

        def on_selection_change():
            """A checkbox changed. After Select All / Clear All every checkbox reports in, with nothing new to draw."""
            if [r['run_path'] for r in all_runs if selected_map.get(r['run_path'])] != rendered_selection:
                render_all_charts()

        def select_all():
            for path in selected_map:
                selected_map[path] = True
            render_all_charts()

        def clear_all():
            for path in selected_map:
                selected_map[path] = False
            render_all_charts()

        # Left Pane
        with ui.left_drawer(top_corner=True, bottom_corner=True).style('background-color: #edf2f7').classes('p-3 gap-3') as left_drawer:
            with ui.row().classes('w-full gap-2'):
                ui.button('Select All', icon='select_all', on_click=select_all).props('unelevated square no-caps color=white text-color=slate-800').classes('flex-1')
                ui.button('Clear All', icon='clear_all', on_click=clear_all).props('unelevated square no-caps color=white text-color=slate-800').classes('flex-1')

            # Runs of one group (e.g. a model version) share a folder prefix, so they are already next to each other
            has_groups = any(run.get('group') for run in all_runs)
            shown_group = None
            for run in all_runs:
                group = run.get('group') or 'ungrouped'
                if has_groups and group != shown_group:
                    shown_group = group
                    ui.label(group).classes('font-bold font-mono text-xs tracking-wider text-slate-600 mt-1')

                title_text = get_run_label(run)

                ts_raw = str(run.get('timestamp', ''))
                if len(ts_raw) == 15 and '_' in ts_raw:
                    ts_fmt = f"{ts_raw[:4]}-{ts_raw[4:6]}-{ts_raw[6:8]} {ts_raw[9:11]}:{ts_raw[11:13]}:{ts_raw[13:15]}"
                else:
                    ts_fmt = ts_raw
                commit = run.get('git_commit', 'unknown')
                sub_text = f"{ts_fmt} • {commit}" if ts_fmt else str(commit)

                with ui.card().props('flat square').classes('w-full p-3 shadow-sm gap-1 border border-slate-200'):
                    with ui.row().classes('items-center gap-2 no-wrap w-full'):
                        ui.checkbox(
                            on_change=on_selection_change
                        ).bind_value(selected_map, run['run_path']).props('dense color=dark')
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
                on_change=lambda: render_all_charts(),
            ).props('spread no-caps toggle-color=dark toggle-text-color=white color=white text-color=slate-800 unelevated square').classes('w-full')

            font_select = ui.select(
                options=['sans-serif', 'Times New Roman', 'Arial', 'Courier New'],
                value='sans-serif',
                label='Font Family',
                on_change=lambda: render_all_charts(),
            ).classes('w-full')

            ui.label('Legend').classes('text-xs text-slate-600')
            legend_toggle = ui.toggle(
                options={True: 'Show', False: 'Hide'},
                value=True,
                on_change=lambda: render_all_charts(),
            ).props('spread no-caps toggle-color=dark toggle-text-color=white color=white text-color=slate-800 unelevated square').classes('w-full')

            ui.label('Y-Axis Scale').classes('text-xs text-slate-600')
            y_scale_toggle = ui.toggle(
                options={False: 'Zero', True: 'Focus'},
                value=False,
                on_change=lambda: render_all_charts(),
            ).props('spread no-caps toggle-color=dark toggle-text-color=white color=white text-color=slate-800 unelevated square').classes('w-full')

            ui.label('Font Sizes').classes('text-xs text-slate-600')
            with ui.row().classes('w-full gap-2'):
                title_font_size = ui.number(label='Title', value=18, min=8, max=60, step=1, on_change=lambda: render_all_charts()).props('dense outlined').classes('flex-1')
                x_font_size = ui.number(label='X-Axis', value=12, min=6, max=40, step=1, on_change=lambda: render_all_charts()).props('dense outlined').classes('flex-1')
                y_font_size = ui.number(label='Y-Axis', value=12, min=6, max=40, step=1, on_change=lambda: render_all_charts()).props('dense outlined').classes('flex-1')

            ui.label('Rendering Mode').classes('text-xs text-slate-600')
            renderer_toggle = ui.toggle(
                options={'svg': 'SVG', 'canvas': 'Rasterizer'},
                value='svg',
                on_change=lambda: render_all_charts(),
            ).props('spread no-caps toggle-color=dark toggle-text-color=white color=white text-color=slate-800 unelevated square').classes('w-full')

            ui.label('Graph Color Override').classes('text-xs text-slate-600')
            color_override_input = ui.color_input(
                label='Base Color',
                value='',
                placeholder='Auto (per run)',
                preview=True,
                on_change=lambda: render_all_charts(),
            ).props('clearable').classes('w-full')

            ui.label('2D Chart Style').classes('text-xs text-slate-600')
            style_2d_toggle = ui.toggle(
                options={'dots': 'Dots', 'lines': 'Lines'},
                value='dots',
                on_change=lambda: render_all_charts(),
            ).props('spread no-caps toggle-color=dark toggle-text-color=white color=white text-color=slate-800 unelevated square').classes('w-full')

            ui.label('2D Aspect Ratio').classes('text-xs text-slate-600')
            aspect_2d_toggle = ui.toggle(
                options={True: '1:1', False: 'Auto'},
                value=True,
                on_change=lambda: render_all_charts(),
            ).props('spread no-caps toggle-color=dark toggle-text-color=white color=white text-color=slate-800 unelevated square').classes('w-full')

            ui.label('Live Refresh').classes('text-xs text-slate-600')
            live_refresh_toggle = ui.toggle(
                options={True: 'On', False: 'Off'},
                value=True,
            ).props('spread no-caps toggle-color=dark toggle-text-color=white color=white text-color=slate-800 unelevated square').classes('w-full')


        def on_poll_tick():
            if not live_refresh_toggle.value:
                return
            selected_runs = [r for r in all_runs if selected_map.get(r['run_path'])]
            if not selected_runs:
                return
            if data_manager.poll_changes(selected_runs):
                update_charts_incremental()

        ui.timer(2.0, on_poll_tick)

        with ui.page_scroller(position='bottom-right', x_offset=20, y_offset=20):
            ui.button('Scroll to Top').props('flat dense color=slate-700')



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
