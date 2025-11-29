import json
import pathlib


SCHEMAS_DIR = pathlib.Path(__file__).parent.resolve()
with open(SCHEMAS_DIR / "dataset_meta_schema.json") as _fid:
    DATASET_META_SCHEMA = json.load(_fid)
with open(SCHEMAS_DIR / "group_meta_schema.json") as _fid:
    GROUP_META_SCHEMA = json.load(_fid)
