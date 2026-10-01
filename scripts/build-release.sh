#!/usr/bin/env bash
# Portable release build + packaging for QORA-TTS.
#
# Produces a GENERIC x86-64 binary (no -C target-cpu=native) under
# target/<triple>/release/ — isolated from your personal native build in
# target/release/ (never overwritten by this script).
#
# Best performance = build locally:
#   RUSTFLAGS="-C target-cpu=native" cargo build --release --bin qora-tts
#
# Usage:
#   ./scripts/build-release.sh [triple] [outdir]
#   triple defaults to x86_64-unknown-linux-gnu (host).
#   Needs installed target: rustup target add <triple>
#
# Layout inputs (override via env):
#   QORA_MODEL   main weights      (default ./model.qora-tts)
#   QORA_ENCODER codec encoder     (default ./model/speech_tokenizer-model.safetensors)
set -euo pipefail
cd "$(dirname "$0")/.."

TRIPLE="${1:-x86_64-unknown-linux-gnu}"
OUTDIR="${2:-dist/qora-tts-${TRIPLE}}"
QORA_MODEL="${QORA_MODEL:-./model.qora-tts}"
QORA_ENCODER="${QORA_ENCODER:-./model/speech_tokenizer-model.safetensors}"

echo "==> portable build for ${TRIPLE} (generic x86-64, isolated dir)"
RUSTFLAGS="-C target-cpu=x86-64" \
  cargo build --release --target "${TRIPLE}" --bin qora-tts
BIN="target/${TRIPLE}/release/qora-tts"
[ -x "${BIN}" ] || { echo "binary missing: ${BIN}"; exit 1; }

echo "==> packaging into ${OUTDIR}"
for f in "${QORA_MODEL}" "${QORA_ENCODER}" \
         target/release/config.json target/release/tokenizer.json \
         target/release/merges.txt target/release/vocab.json \
         target/release/tokenizer_config.json; do
  [ -f "$f" ] || { echo "missing input: $f"; exit 1; }
done
rm -rf "${OUTDIR}"
mkdir -p "${OUTDIR}/speech_tokenizer"
cp "${BIN}" "${OUTDIR}/qora-tts"
cp "${QORA_MODEL}" "${OUTDIR}/model.qora-tts"
cp -L "${QORA_ENCODER}" "${OUTDIR}/speech_tokenizer/model.safetensors"  # resolve symlink
cp target/release/config.json target/release/tokenizer.json \
   target/release/merges.txt target/release/vocab.json \
   target/release/tokenizer_config.json "${OUTDIR}/"
cp target/release/speech_tokenizer/*.json "${OUTDIR}/speech_tokenizer/"
(cd "${OUTDIR}" && sha256sum qora-tts model.qora-tts \
  speech_tokenizer/model.safetensors config.json tokenizer.json > sha256sums.txt)
du -sh "${OUTDIR}"
echo "==> self-check (model files present?)"
./"${OUTDIR}"/qora-tts --check 2>&1 | tail -3
echo "OK: ${OUTDIR}"
