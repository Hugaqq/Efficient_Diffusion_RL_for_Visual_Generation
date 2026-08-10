#!/usr/bin/env bash
set -euo pipefail

# Archive only the already accepted frozen A7 matrix.  Training writes to
# tmpfs; this script deliberately has no training or retry behavior.
evidence_root=/dev/shm/v-qiaoqifan/visualrl-v08-a7-final-56507f-6f1533ef
config_root=/mnt/data/v-qiaoqifan/visual_rl_runs/v08_modular_gpu_20260808/configs/a7-final-56507f-6f1533ef
release_root=/mnt/data/v-qiaoqifan/visual_rl_runs/v08_modular_gpu_20260808/release_candidates/code-56507f6e-wheel-6f1533ef
archive_parent=/mnt/data/v-qiaoqifan/visual_rl_runs/v08_modular_gpu_20260808/accepted
archive_target=$archive_parent/a7-final-56507f-6f1533ef
python_bin=${PYTHON_BIN:-/home/v-qiaoqifan/miniconda3/envs/visual-rl-sd35/bin/python}

routes=(
  flow-grpo-sd3
  flow-grpo-wan
  tempflow-sd3
  flash-wan
  world-r1-core-wan
  world-r1-release-surrogate-wan
)

[[ -x "$python_bin" ]] || { echo "missing Python: $python_bin" >&2; exit 66; }
[[ ! -e "$archive_target" ]] || {
  echo "refusing to overwrite accepted archive: $archive_target" >&2
  exit 73
}
for name in acceptance launch-receipts logs runs tools; do
  [[ -d "$evidence_root/$name" && ! -L "$evidence_root/$name" ]] || {
    echo "missing or unsafe evidence directory: $evidence_root/$name" >&2
    exit 66
  }
done
[[ -d "$config_root" && ! -L "$config_root" ]] || {
  echo "missing or unsafe frozen config directory: $config_root" >&2
  exit 66
}
for name in a7-freeze-identity.json a7-reward-artifact-identities.json; do
  [[ -f "$release_root/$name" && ! -L "$release_root/$name" ]] || {
    echo "missing or unsafe freeze record: $release_root/$name" >&2
    exit 66
  }
done

matrix=$evidence_root/acceptance/matrix.json
"$python_bin" - "$matrix" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = json.loads(path.read_text(encoding="utf-8"))
if not (
    payload.get("kind") == "visual_rl_a7_matrix_acceptance"
    and payload.get("accepted") is True
    and payload.get("route_count") == 6
):
    raise SystemExit("matrix.json is not an accepted six-route matrix")
PY

for route in "${routes[@]}"; do
  acceptance=$evidence_root/acceptance/$route.json
  success=$evidence_root/runs/$route/SUCCESS
  "$python_bin" - "$route" "$acceptance" "$success" <<'PY'
import json
import sys
from pathlib import Path

route = sys.argv[1]
acceptance = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
success = json.loads(Path(sys.argv[3]).read_text(encoding="utf-8"))
if not (
    acceptance.get("kind") == "visual_rl_a7_route_acceptance"
    and acceptance.get("route") == route
    and acceptance.get("accepted") is True
    and acceptance.get("committed_steps") == 20
    and success.get("committed_steps") == 20
):
    raise SystemExit(f"{route}: route is not an accepted 20-step terminal run")
PY
done

if find \
  "$evidence_root/acceptance" \
  "$evidence_root/launch-receipts" \
  "$evidence_root/logs" \
  "$evidence_root/runs" \
  "$evidence_root/tools" \
  "$config_root" \
  -type l -print -quit | grep -q .; then
  echo "refusing to archive evidence containing symbolic links" >&2
  exit 65
fi

mkdir -p "$archive_parent"
staging=$(mktemp -d "$archive_parent/.a7-final-56507f-6f1533ef.staging.XXXXXX")
echo "archiving into staging directory: $staging"
for name in acceptance launch-receipts logs runs tools; do
  cp -a "$evidence_root/$name" "$staging/$name"
done
cp -a "$config_root" "$staging/configs"
mkdir "$staging/freeze"
cp -a \
  "$release_root/a7-freeze-identity.json" \
  "$release_root/a7-reward-artifact-identities.json" \
  "$staging/freeze/"

for name in acceptance launch-receipts logs runs tools; do
  diff -qr "$evidence_root/$name" "$staging/$name"
done
diff -qr "$config_root" "$staging/configs"
diff -q \
  "$release_root/a7-freeze-identity.json" \
  "$staging/freeze/a7-freeze-identity.json"
diff -q \
  "$release_root/a7-reward-artifact-identities.json" \
  "$staging/freeze/a7-reward-artifact-identities.json"

(
  cd "$staging"
  find acceptance launch-receipts logs runs tools configs freeze \
    -type f -print0 \
    | LC_ALL=C sort -z \
    | xargs -0 sha256sum > SHA256SUMS
  sha256sum -c SHA256SUMS >/dev/null
)
sync -f "$staging"
mv "$staging" "$archive_target"
(
  cd "$archive_target"
  sha256sum -c SHA256SUMS >/dev/null
)
echo "accepted archive committed: $archive_target"
