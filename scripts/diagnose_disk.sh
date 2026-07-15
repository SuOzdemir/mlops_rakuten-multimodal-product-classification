#!/bin/bash
# diagnose_disk.sh
# ==================
# What's actually eating disk: this project's own data/checkpoints, Docker's
# build cache/images, or Docker-managed volumes (postgres/minio/grafana/etc)?
# Read-only -- doesn't delete or move anything, just reports.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "============================================================"
echo "1. Docker's own accounting (images, containers, volumes, build cache)"
echo "============================================================"
docker system df -v 2>/dev/null || docker system df

echo ""
echo "============================================================"
echo "2. Named volumes used by this project (postgres/minio/grafana/prometheus)"
echo "============================================================"
for vol in postgres_data minio_data grafana_data prometheus_data; do
  full_name="mlops-rakuten-product-classification_${vol}"
  size=$(docker system df -v 2>/dev/null | awk -v v="$full_name" '$1==v {print $3}')
  echo "  ${vol}: ${size:-not found (not created yet?)}"
done

echo ""
echo "============================================================"
echo "3. This repo's own on-disk data (raw data, DVC-tracked outputs,"
echo "   model checkpoints, mlruns/ if the old local-artifact layout"
echo "   is still present anywhere)"
echo "============================================================"
for dir in data outputs models mlruns figures; do
  path="$PROJECT_DIR/$dir"
  if [ -d "$path" ]; then
    du -sh "$path" 2>/dev/null | awk -v d="$dir" '{print "  "d": "$1}'
  fi
done

echo ""
echo "  Largest individual files under the repo (top 10, excludes .git):"
find "$PROJECT_DIR" -path "$PROJECT_DIR/.git" -prune -o -type f -print0 2>/dev/null \
  | xargs -0 du -h 2>/dev/null | sort -rh | head -10 | sed 's/^/    /'

echo ""
echo "============================================================"
echo "4. MinIO bucket contents (if minio is running) -- artifacts now live"
echo "   here, not on any single container's own disk"
echo "============================================================"
if docker ps --format '{{.Names}}' | grep -q '^mlops-rakuten-product-classification-minio-1$'; then
  docker run --rm --network mlops-rakuten-product-classification_default \
    --entrypoint sh minio/mc -c "
      mc alias set local http://minio:9000 minioadmin minioadmin >/dev/null 2>&1
      echo '  mlflow-artifacts bucket:'
      mc du local/mlflow-artifacts 2>/dev/null | sed 's/^/    /' || echo '    (empty or unreachable)'
      echo '  dvc-data bucket:'
      mc du local/dvc-data 2>/dev/null | sed 's/^/    /' || echo '    (empty or unreachable)'
    " 2>/dev/null
else
  echo "  minio isn't running (docker compose up -d minio to check)"
fi

echo ""
echo "============================================================"
echo "5. Docker Desktop's own VM disk image (the ceiling everything above"
echo "   shares) -- Settings > Resources > Disk image size, on macOS/Windows"
echo "============================================================"
docker info 2>/dev/null | grep -i "docker root dir\|total memory" || true

echo ""
echo "Done. If (1) build cache is large: 'docker builder prune'."
echo "If (2)/(4) volumes/buckets are large: see README.md's retention-policy"
echo "section (mc ilm add for MinIO auto-expiry) before deleting anything by hand."
