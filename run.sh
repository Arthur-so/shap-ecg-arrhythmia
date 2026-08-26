#!/usr/bin/env bash
# Dispara e acompanha a execução do notebook (kernel) no Kaggle.
#
# Uso:
#   ./run.sh            # publica o notebook (executa) e acompanha até terminar
#   ./run.sh watch      # só acompanha a execução atual (não dispara nova)
#   ./run.sh logs       # baixa e imprime o log da última execução uma vez
#   ./run.sh status     # mostra o status atual e sai
#   ./run.sh stop        # cancela a execução em andamento
#
# Intervalo de polling (segundos): INTERVAL=10 ./run.sh watch
set -uo pipefail
cd "$(dirname "$0")"

KERNEL="arthurso/shap-ecg-arrhythmia-runner"
SLUG="shap-ecg-arrhythmia-runner"
KAGGLE="./.venv/bin/kaggle"
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
  python3 - "$logfile" "$STATEDIR/.printed" <<'PY'
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
  : > "$STATEDIR/.printed"; echo 0 > "$STATEDIR/.printed"
  while true; do
    local st; st="$(status_line)"
    echo "[$(date +%H:%M:%S)] $st"
    print_new_logs
    if ! echo "$st" | grep -qiE "running|queued|queue"; then
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

case "${1:-run}" in
  run)
    echo "== publicando e executando o notebook =="
    k kernels push -p kaggle/
    echo "== aguardando o Kaggle enfileirar... =="
    sleep 5
    watch
    ;;
  watch)  watch ;;
  logs)   rm -f "$STATEDIR/.printed"; print_new_logs ;;
  status) status_line ;;
  stop)   k kernels cancel "$KERNEL" 2>&1 || k kernels stop "$KERNEL" 2>&1 ;;
  *) echo "uso: ./run.sh [run|watch|logs|status|stop]"; exit 1 ;;
esac
