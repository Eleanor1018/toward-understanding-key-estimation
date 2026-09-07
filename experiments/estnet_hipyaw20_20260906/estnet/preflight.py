"""Read-only static checks, usable without torch or Isaac Sim installed."""
import argparse
import hashlib
import importlib.util
import json
import platform
from pathlib import Path
from .config import Config

ASSET_HASHES = {
    "g1_29dof_rev_1_0.usd": "cfed730bb043c42708260f25bf3bdf49ac80deed5d5a49384721b8e01059a96e",
    "configuration/g1_29dof_rev_1_0_base.usd": "d9768c942783ae0932f0c3db3558d5283b9a93ba0e561d15d8984f548aa2a65d",
    "configuration/g1_29dof_rev_1_0_physics.usd": "cde4ff0378183e4e2063df1fdcad4060df8247eaf1da777f45e2adea0b6dde22",
    "configuration/g1_29dof_rev_1_0_sensor.usd": "5115364b53bffbb37007af0fd1038336679b66abdae134bc87d0fca9829f9b60",
}


def inspect_asset(asset):
    asset = Path(asset).resolve()
    files = []
    for name, expected in ASSET_HASHES.items():
        path = asset if name == "g1_29dof_rev_1_0.usd" else asset.parent / name
        digest = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
        files.append({"path": str(path), "exists": path.is_file(), "sha256": digest,
                      "expected_sha256": expected, "matches_known_asset": digest == expected})
    return {"files": files, "ready": all(x["matches_known_asset"] for x in files)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset", default="assets/g1_29dof/g1_29dof_rev_1_0.usd")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    cfg = Config()
    cfg.validate()
    report = {"schema": cfg.schema, "python": platform.python_version(),
              "isaaclab_import_available": importlib.util.find_spec("isaaclab") is not None,
              "asset": inspect_asset(args.asset), "configuration": cfg.to_dict(),
              "simulation_validation": "pending: static checks do not establish walking"}
    payload = json.dumps(report, indent=2, ensure_ascii=False)
    print(payload)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    return 0 if report["asset"]["ready"] and report["isaaclab_import_available"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
