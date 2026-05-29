"""
python tools/import_android_data.py <archivo.json>

Convierte el archivo JSON exportado por la app Android (android_collector/)
al formato .npy que usa train.py.

Estructura esperada del JSON:
{
  "signs": {
    "A": [[[x,y,z], ...21 puntos], ...N muestras],
    "B": [...],
    ...
  }
}

Cada muestra se guarda como data/signs/{label}/sample_{n:04d}.npy (shape 21x3),
continuando la numeracion de las muestras existentes en disco.

Ejemplo de uso:
  python tools/import_android_data.py lsm_data_2025-01-15.json
  python train.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

# Allow running from project root or from tools/
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from config import AppConfig


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    json_path = Path(sys.argv[1])
    if not json_path.exists():
        print(f"ERROR: No se encontro el archivo '{json_path}'")
        sys.exit(1)

    print(f"\nLeyendo {json_path} ...")
    with json_path.open(encoding="utf-8") as f:
        data = json.load(f)

    if "signs" not in data:
        print("ERROR: El JSON no tiene el campo 'signs'. "
              "Asegurate de exportar desde la app LSM Recolector.")
        sys.exit(1)

    version = data.get("version", "?")
    exported_at = data.get("exported_at", "desconocido")
    print(f"Version: {version}  |  Exportado: {exported_at}\n")

    cfg      = AppConfig()
    data_dir = cfg.collector.data_dir
    total    = 0
    skipped  = 0

    for label, samples in data["signs"].items():
        if not samples:
            continue

        dest = data_dir / label
        dest.mkdir(parents=True, exist_ok=True)

        existing = len(list(dest.glob("*.npy")))
        imported = 0

        for i, sample in enumerate(samples):
            try:
                arr = np.array(sample, dtype=np.float32)
            except (ValueError, TypeError) as exc:
                print(f"  WARN  {label} muestra {i}: no se pudo convertir ({exc})")
                skipped += 1
                continue

            if arr.shape != (21, 3):
                print(f"  WARN  {label} muestra {i}: forma {arr.shape} (se esperaba (21,3)) — omitida")
                skipped += 1
                continue

            out_path = dest / f"sample_{existing + imported:04d}.npy"
            np.save(out_path, arr)
            imported += 1

        total += imported
        status = "OK" if existing + imported >= cfg.collector.samples_per_sign else f"{existing + imported}/{cfg.collector.samples_per_sign}"
        print(f"  {label:<10}  {imported:>4} nuevas  |  total en disco: {existing + imported:>4}  [{status}]")

    print(f"\nTotal importado : {total} muestras")
    if skipped:
        print(f"Omitidas        : {skipped} muestras con formato incorrecto")
    print(f"Directorio datos: {data_dir}\n")
    print("Siguiente paso  : python train.py")


if __name__ == "__main__":
    main()
