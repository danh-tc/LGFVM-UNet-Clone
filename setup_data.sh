#!/bin/bash
# Download, extract, and prepare the Synapse dataset.
# Usage: bash setup_data.sh
set -e

DATA_DIR="data/Synapse"
LIST_DIR="${DATA_DIR}/lists/lists_Synapse"
ZIP_PATH="data/Synapse.zip"
FILE_ID="1m9ihuBdgxDp0hlJyIlh94tJqzllo0klt"

# 1. Download
echo "==> Downloading Synapse dataset from Google Drive..."
venv/bin/pip install -q gdown
venv/bin/gdown "${FILE_ID}" -O "${ZIP_PATH}"

# 2. Extract
echo "==> Extracting to data/..."
unzip -q -o "${ZIP_PATH}" -d data/
rm "${ZIP_PATH}"

# 3. Rebuild train.txt (the bundled one only has 3 entries)
echo "==> Rebuilding train.txt..."
TEST_CASES="case0001 case0002 case0003 case0004 case0008 case0022 case0025 case0029 case0032 case0035 case0036 case0038"

ls "${DATA_DIR}/train_npz/" | while read f; do
    case=$(echo "$f" | grep -oP "case\d+")
    if ! echo "$TEST_CASES" | grep -qw "$case"; then
        echo "$f"
    fi
done > "${LIST_DIR}/train.txt"

echo "    train slices : $(wc -l < ${LIST_DIR}/train.txt)"
echo "    test slices  : $(wc -l < ${LIST_DIR}/test.txt)"
echo ""
echo "Done! Run: ./venv/bin/python train_synapse.py"
