import os
import atexit
import json
from typing import Set

class StringSetDB:
    def __init__(self, path: str):
        self.path = path
        self._data: Set[str] = set()
        self._orig_data_size: int = 0

        print(f"using file at {os.path.abspath(self.path)}")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                try:
                    self._data = set(json.load(f))
                    self._orig_data_size = len(self._data)
                    print(f"Read in database with size {len(self._data)}")
                except Exception:
                    raise ValueError(f"Failed to load data from {path}")
        else:
            print(f"Database did not exist; will init")

        atexit.register(self._save)

    def add(self, item: str) -> None:
        assert item not in self._data
        self._data.add(item)

    def __contains__(self, item: str) -> bool:
        return item in self._data

    def _save(self) -> None:
        print(f"maybe dumping database, size {len(self._data)}")
        if len(self._data) < self._orig_data_size:
            raise Exception("data got smaller")
        if len(self._data) == self._orig_data_size:
            print(f'database did not change; will not overwrite')
            return
        with open(self.path, "w", encoding="utf-8") as f:
            print(f'database did change; {self._orig_data_size} -> {len(self._data)}')
            json.dump(sorted(self._data), f, ensure_ascii=False, indent=2)

# todo: maybe instead of this, we could have passed a flag into the other one
class StringSetDBNoOp(StringSetDB):
    def __init__(self, *args):
        pass

    def add(self, item: str) -> None:
        return

    def __contains__(self, item):
        return False

    def _save(self) -> None:
        return
