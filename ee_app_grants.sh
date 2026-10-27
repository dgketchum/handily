#!/usr/bin/env bash
# Reader grants for the handily EE app assets (run AFTER publishing the app in the Code Editor).
#   bash ee_app_grants.sh <app-service-account-email>   # restricted app: grant the app's SA reader
#   bash ee_app_grants.sh --public                       # open app: anyone can read (allUsers)
# Every published EE app mints its own service account; find it with
#   gcloud iam service-accounts list --project <app project>   (e.g. ee-dgketchum or ssebop-montana)
# ImageCollection children inherit the collection ACL; tables under the wells folders do NOT, so
# the four tables are granted individually. Idempotent.
set -euo pipefail
[ $# -eq 1 ] || { sed -n 2,5p "$0"; exit 1; }
if [ "$1" = "--public" ]; then WHO=allUsers; else WHO="$1"; fi
ROOT=projects/ssebop-montana/assets/handily/conus
ASSETS="$ROOT $ROOT/dtw $ROOT/wte $ROOT/wells $ROOT/wells_train $ROOT/wells/nv $ROOT/wells/nm $ROOT/wells_train/nv $ROOT/wells_train/nm $ROOT/wells/mt $ROOT/wells_train/mt"
for a in $ASSETS; do earthengine acl ch -u $WHO:R $a 2>&1 | grep -v -E "Warn|warnings.warn" || true; echo "granted $WHO:R on $a"; done
echo; for a in $ROOT/dtw $ROOT/wells/nm; do echo "== $a"; earthengine acl get $a 2>/dev/null | tr -d '\n '; echo; done
