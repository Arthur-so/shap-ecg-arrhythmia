#!/usr/bin/env bash
# Dispara e acompanha a execução do notebook (kernel) no Kaggle.
#
# Uso:
#   ./run.sh                       # executa com os hiperparâmetros padrão
#   ./run.sh --epochs 5 --lr 1e-3  # executa sobrescrevendo hiperparâmetros
#   ./run.sh smoke                 # teste rápido (poucas épocas + subconjunto)
#   ./run.sh watch                 # só acompanha a execução atual
#   ./run.sh logs                  # baixa e imprime o log da última execução
#   ./run.sh status                # mostra o status atual e sai
#   ./run.sh stop                  # cancela a execução em andamento
#
# Flags de hiperparâmetro (repassadas ao notebook na publicação):
#   --epochs N   --lr X   --batch-size N   --patience N
#   --dropout X  --limit N (0=todos; >0=subconjunto)   --max-per-class N
#
# Intervalo de polling (s): INTERVAL=10 ./run.sh watch
set -uo pipefail
cd "$(dirname "$0")"

KERNEL="arthurso/shap-ecg-arrhythmia-runner"
SLUG="shap-ecg-arrhythmia-runner"
KAGGLE="./.venv/bin/kaggle"
PY="./.venv/bin/python"
STATEDIR="/tmp/kaggle-run-${SLUG}"
INTERVAL="${INTERVAL:-15}"

if [ ! -x "$KAGGLE" ]; then
  echo "ERRO: CLI do Kaggle não encontrada em $KAGGLE (venv 3.12)." >&2
  exit 1
fi
if [ ! -f "$HOME/.kaggle/access_token" ]; then
  echo "ERRO: ~/.kaggle/access_token não encontrado (token KGAT)." >&2
  exit 1
fi
export KAGGLE_API_TOKEN="$(cat "$HOME/.kaggle/access_token")"
mkdir -p "$STATEDIR"

k() { "$KAGGLE" "$@"; }
status_line() { k kernels status "$KERNEL" 2>&1 | head -1; }

# Baixa o output e imprime apenas as linhas de log ainda não exibidas.
print_new_logs() {
  k kernels output "$KERNEL" -p "$STATEDIR" >/dev/null 2>&1 || return 0
  local logfile
  logfile="$(ls -t "$STATEDIR"/*.log 2>/dev/null | head -1)"
  [ -n "${logfile:-}" ] && [ -f "$logfile" ] || return 0
  "$PY" - "$logfile" "$STATEDIR/.printed" <<'PY'
import json, sys, os
logfile, offfile = sys.argv[1], sys.argv[2]
try:
    entries = json.load(open(logfile))
except Exception:
    sys.exit(0)
text = "".join(e.get("data", "") for e in entries if isinstance(e, dict))
prev = 0
if os.path.exists(offfile):
    try: prev = int(open(offfile).read().strip() or 0)
    except Exception: prev = 0
new = text[prev:]
if new:
    sys.stdout.write(new)
    if not new.endswith("\n"): sys.stdout.write("\n")
open(offfile, "w").write(str(len(text)))
PY
}

watch() {
  echo "== acompanhando $KERNEL (Ctrl+C para parar de acompanhar) =="
  echo 0 > "$STATEDIR/.printed"
  while true; do
    local st; st="$(status_line)"
    echo "[$(date +%H:%M:%S)] $st"
    print_new_logs
    if ! echo "$st" | grep -qiE "running|queue"; then
      echo "== execução finalizada =="
      echo "-- log completo --"
      rm -f "$STATEDIR/.printed"; print_new_logs
      echo "-- arquivos de saída em $STATEDIR --"
      ls -1 "$STATEDIR" | grep -v '^\.' || true
      break
    fi
    sleep "$INTERVAL"
  done
}

# Publica o notebook, injetando hiperparâmetros numa cópia temporária
# (o notebook versionado mantém seus valores padrão).
build_and_push() {
  local tmp; tmp="$(mktemp -d)"
  cp notebooks/kaggle_runner.ipynb "$tmp/kaggle_runner.ipynb"
  "$PY" kaggle/inject_params.py "$tmp/kaggle_runner.ipynb" || { rm -rf "$tmp"; exit 1; }
  "$PY" - "$tmp/kernel-metadata.json" <<'PY'
import json, sys
m = json.load(open("kaggle/kernel-metadata.json"))
m["code_file"] = "kaggle_runner.ipynb"   # notebook está na mesma pasta temp
json.dump(m, open(sys.argv[1], "w"), indent=2)
PY
  echo "== publicando e executando o notebook =="
  k kernels push -p "$tmp"
  rm -rf "$tmp"
}

do_run() {
  while [ $# -gt 0 ]; do
    case "$1" in
      --epochs)        export RUNSH_EPOCHS="$2"; shift 2;;
      --lr)            export RUNSH_LR="$2"; shift 2;;
      --batch-size)    export RUNSH_BATCH_SIZE="$2"; shift 2;;
      --patience)      export RUNSH_PATIENCE="$2"; shift 2;;
      --dropout)       export RUNSH_DROPOUT="$2"; shift 2;;
      --limit)         export RUNSH_LIMIT="$2"; shift 2;;
      --max-per-class) export RUNSH_MAX_PER_CLASS="$2"; shift 2;;
      *) echo "flag desconhecida: $1" >&2
         echo "flags: --epochs --lr --batch-size --patience --dropout --limit --max-per-class" >&2
         exit 1;;
    esac
  done
  build_and_push
  echo "== aguardando o Kaggle enfileirar... =="
  sleep 5
  watch
}

case "${1:-run}" in
  watch)  watch ;;
  logs)   rm -f "$STATEDIR/.printed"; print_new_logs ;;
  status) status_line ;;
  stop)   k kernels cancel "$KERNEL" 2>&1 || k kernels stop "$KERNEL" 2>&1 ;;
  smoke)  echo "== teste rápido (2 épocas, 200 registros, 10 amostras/classe) =="
          do_run --epochs 2 --batch-size 16 --limit 200 --max-per-class 10 ;;
  run)    shift; do_run "$@" ;;
  --*)    do_run "$@" ;;
  *) echo "uso: ./run.sh [run [flags]|smoke|watch|logs|status|stop]"; exit 1 ;;
esac
