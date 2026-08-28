#!/usr/bin/env python3
"""Injeta hiperparâmetros na célula PARAMS do notebook runner.

Lê variáveis de ambiente ``RUNSH_*`` (definidas pelo run.sh a partir das flags
de linha de comando) e sobrescreve as atribuições correspondentes na célula de
parâmetros do notebook. Opera sobre uma CÓPIA temporária do notebook, para que
o notebook versionado mantenha sempre seus valores padrão.

Uso: python kaggle/inject_params.py <caminho-do-notebook.ipynb>
"""
import json
import os
import re
import sys

# env var (run.sh) -> variável no notebook
MAP = {
    "RUNSH_EPOCHS": "EPOCHS",
    "RUNSH_LR": "LR",
    "RUNSH_BATCH_SIZE": "BATCH_SIZE",
    "RUNSH_PATIENCE": "PATIENCE",
    "RUNSH_DROPOUT": "DROPOUT",
    "RUNSH_LIMIT": "LIMIT",
    "RUNSH_MAX_PER_CLASS": "MAX_PER_CLASS",
}


def main() -> int:
    if len(sys.argv) != 2:
        print("uso: inject_params.py <notebook.ipynb>", file=sys.stderr)
        return 2
    nb_path = sys.argv[1]

    overrides = {var: os.environ[env] for env, var in MAP.items()
                 if os.environ.get(env, "") != ""}

    with open(nb_path) as f:
        nb = json.load(f)

    applied: dict[str, str] = {}
    found_cell = False
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        if "PARAMS" not in "".join(cell.get("source", [])):
            continue
        found_cell = True
        new_source = []
        for line in cell["source"]:
            m = re.match(r"^([A-Z_]+)\s*=", line)
            if m and m.group(1) in overrides:
                name = m.group(1)
                comment = ""
                if "#" in line:
                    comment = "  #" + line.split("#", 1)[1].rstrip("\n")
                new_source.append(f"{name} = {overrides[name]}{comment}\n")
                applied[name] = overrides[name]
            else:
                new_source.append(line)
        cell["source"] = new_source

    if not found_cell:
        print("[aviso] célula PARAMS não encontrada no notebook", file=sys.stderr)

    with open(nb_path, "w") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)

    print("[inject] parâmetros aplicados:",
          applied if applied else "(nenhum override — usando padrões do notebook)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
