#!/usr/bin/env bash
set -euo pipefail

# Convert terminal frozen runs into immutable per-route receipts, then emit the
# six-route matrix receipt.  Live routes are skipped; failed terminal routes
# fail closed through audit_a7_route.py.
source_root=/dev/shm/v-qiaoqifan/visualrl-v08-candidate-56507f6e-source
evidence_root=/dev/shm/v-qiaoqifan/visualrl-v08-a7-final-56507f-6f1533ef
release_root=/mnt/data/v-qiaoqifan/visual_rl_runs/v08_modular_gpu_20260808/release_candidates/code-56507f6e-wheel-6f1533ef
python_bin=${PYTHON_BIN:-/home/v-qiaoqifan/miniconda3/envs/visual-rl-sd35/bin/python}

route_auditor=$evidence_root/tools/audit_a7_route.py
matrix_auditor=$evidence_root/tools/audit_a7_matrix.py
freeze_record=$release_root/a7-freeze-identity.json
reward_identities=$release_root/a7-reward-artifact-identities.json
acceptance_root=$evidence_root/acceptance
routes=(
  flow-grpo-sd3
  flow-grpo-wan
  tempflow-sd3
  flash-wan
  world-r1-core-wan
  world-r1-release-surrogate-wan
)

[[ -d "$source_root" && ! -L "$source_root" ]] || {
  echo "missing or unsafe frozen source root: $source_root" >&2
  exit 66
}
[[ -x "$python_bin" ]] || {
  echo "missing Python interpreter: $python_bin" >&2
  exit 66
}
for path in \
  "$route_auditor" \
  "$matrix_auditor" \
  "$freeze_record" \
  "$reward_identities"; do
  [[ -f "$path" && ! -L "$path" ]] || {
    echo "missing or unsafe acceptance dependency: $path" >&2
    exit 66
  }
done

mkdir -p "$acceptance_root"
pending=0
for route in "${routes[@]}"; do
  exitcode=$evidence_root/logs/$route.exitcode
  if [[ ! -f "$exitcode" ]]; then
    echo "$route: still live or not started"
    pending=1
    continue
  fi

  receipt=$acceptance_root/$route.json
  temporary=$(mktemp "$acceptance_root/.$route.XXXXXX")
  if ! (
    cd "$source_root"
    env PYTHONPATH=. "$python_bin" "$route_auditor" \
      "$evidence_root" \
      "$route" \
      "$freeze_record" \
      "$reward_identities"
  ) >"$temporary"; then
    echo "$route: terminal acceptance failed; diagnostic output retained at $temporary" >&2
    exit 1
  fi
  if [[ -e "$receipt" ]]; then
    cmp "$temporary" "$receipt" || {
      echo "$route: existing acceptance receipt differs" >&2
      exit 65
    }
    rm "$temporary"
  else
    mv "$temporary" "$receipt"
  fi
  echo "$route: accepted"
done

if [[ $pending -ne 0 ]]; then
  echo "matrix remains pending because at least one route is non-terminal" >&2
  exit 75
fi

matrix=$acceptance_root/matrix.json
temporary=$(mktemp "$acceptance_root/.matrix.XXXXXX")
if ! (
  cd "$source_root"
  env PYTHONPATH=. "$python_bin" "$matrix_auditor" \
    "$acceptance_root" \
    "$freeze_record" \
    "$reward_identities"
) >"$temporary"; then
  echo "matrix acceptance failed; diagnostic output retained at $temporary" >&2
  exit 1
fi
if [[ -e "$matrix" ]]; then
  cmp "$temporary" "$matrix" || {
    echo "existing matrix receipt differs" >&2
    exit 65
  }
  rm "$temporary"
else
  mv "$temporary" "$matrix"
fi
echo "six-route matrix accepted: $matrix"
