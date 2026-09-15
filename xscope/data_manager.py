import os
import json
import logging
import threading
from collections import defaultdict
from dataclasses import dataclass, field

logger = logging.getLogger("xscope.data_manager")


@dataclass
class FileState:
    offset: int = 0
    mtime: float = 0.0
    size: int = 0
    ino: int = 0
    records: list[dict] = field(default_factory=list)
    max_records: int = 5000  # Keep memory bounded for long runs

    def add_record(self, record: dict):
        self.records.append(record)
        if len(self.records) > self.max_records:
            # Downsample: keep every other old record, keep all recent records
            half = self.max_records // 2
            old = self.records[:-half]
            keep = self.records[-half:]
            self.records = old[::2] + keep


class RunDataManager:
    """Manages experiment metadata loading, cached incremental file reading, and change polling."""

    def __init__(self, metrics_dir: str = "metrics"):
        self.metrics_dir = metrics_dir
        self.file_states: dict[tuple[str, str], FileState] = defaultdict(FileState)
        self.lock = threading.RLock()

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
                        logger.warning("Failed to parse %s", meta_path)
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
        filepath = os.path.join(run_path, filename)
        state = self.file_states[(run_path, filename)] # Return defualt if no data exsits yet.

        # Handle missing data or deleted log files
        if not os.path.isfile(filepath): # File NEVER existed
            if state.records:            # File was loaded previously, but was just DELETED
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
            last_good_pos = state.offset
            
            # FIX: Use readline() instead of `for line in f:`
            # This allows f.tell() to work correctly without raising OSError.
            while True:
                line = f.readline()
                if not line:
                    break  # EOF
                
                stripped_line = line.strip()
                if not stripped_line:
                    last_good_pos = f.tell()
                    continue

                try:
                    state.records.append(json.loads(stripped_line))
                    last_good_pos = f.tell()
                except json.JSONDecodeError:
                    # Incomplete JSON line (likely mid-write). 
                    # Break without updating last_good_pos so we retry next poll.
                    break

            state.offset = last_good_pos

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