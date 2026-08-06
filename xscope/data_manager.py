import os
import json
from dataclasses import dataclass, field


@dataclass
class FileState:
    offset: int = 0
    mtime: float = 0.0
    size: int = 0
    records: list[dict] = field(default_factory=list)


class RunDataManager:
    """Manages experiment metadata loading, cached incremental file reading, and change polling."""

    def __init__(self, metrics_dir: str = "metrics"):
        self.metrics_dir = metrics_dir
        self.file_states: dict[tuple[str, str], FileState] = {}

    def load_runs_metadata(self) -> list[dict]:
        """Loads experiment run metadata from meta.json and note.txt files in metrics_dir."""
        runs: list[dict] = []
        if not os.path.exists(self.metrics_dir):
            return runs

        for folder in sorted(os.listdir(self.metrics_dir)):
            folder_path = os.path.join(self.metrics_dir, folder)
            meta_path = os.path.join(folder_path, "meta.json")
            if os.path.isfile(meta_path):
                with open(meta_path, "r", encoding="utf-8") as f:
                    try:
                        meta = json.load(f)
                    except json.JSONDecodeError:
                        continue
                meta['run_path'] = folder_path

                note_path = os.path.join(folder_path, "note.txt")
                if os.path.isfile(note_path):
                    with open(note_path, "r", encoding="utf-8") as f:
                        meta['note'] = f.read()
                else:
                    meta['note'] = ""

                runs.append(meta)
        return runs

    def get_records(self, run_path: str, filename: str) -> tuple[bool, list[dict]]:
        """
        Loads or incrementally updates metric records for a run file (e.g., metrics.jsonl).
        Returns a tuple: (has_changed, records_list).
        
        Zero disk read cost if file size and mtime haven't changed.
        Uses seek() to read only newly appended lines when file grows.
        """
        key = (run_path, filename)
        if key not in self.file_states:
            self.file_states[key] = FileState()

        state = self.file_states[key]
        filepath = os.path.join(run_path, filename)

        if not os.path.isfile(filepath):
            if state.records:
                state.records.clear()
                state.offset = 0
                state.size = 0
                state.mtime = 0.0
                return True, []
            return False, []

        try:
            stat = os.stat(filepath)
        except OSError:
            return False, state.records

        # Stat check: fast short-circuit if file hasn't changed
        if stat.st_size == state.size and stat.st_mtime == state.mtime:
            return False, state.records

        # Reset if file was truncated, recreated, or modified in-place without growing
        if stat.st_size <= state.size or stat.st_size < state.offset:
            state.records.clear()
            state.offset = 0

        with open(filepath, "r", encoding="utf-8") as f:
            f.seek(state.offset)
            for line in f:
                line = line.strip()
                if line:
                    try:
                        state.records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
            state.offset = f.tell()

        state.size = stat.st_size
        state.mtime = stat.st_mtime
        return True, state.records

    def poll_changes(self, selected_runs: list[dict]) -> bool:
        """
        Polls tracked files across selected runs to determine if any data has changed.
        Returns True if at least one file had new data appended, False otherwise.
        """
        has_any_change = False
        target_files = ["metrics.jsonl", "2d.jsonl", "matrix.jsonl"]

        for run in selected_runs:
            run_path = run.get('run_path', '')
            if not run_path:
                continue
            for filename in target_files:
                changed, _ = self.get_records(run_path, filename)
                if changed:
                    has_any_change = True

        return has_any_change
