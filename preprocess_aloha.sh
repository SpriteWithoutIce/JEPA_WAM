#!/usr/bin/env bash

set -euo pipefail

ALOHA_DATA="${ALOHA_DATA:-/ssd/linyihan/datasets/aloha_dataset}"
ROBOTWIN_ALOHA_SET="${ROBOTWIN_ALOHA_SET:-10}"

ROBOTWIN_ALOHA_ALL_DATASETS=(
  "adjust_bottle_clean_50"
  "adjust_bottle_randomized_500"
  "beat_block_hammer_clean_50"
  "beat_block_hammer_randomized_500"
  "blocks_ranking_rgb_clean_50"
  "blocks_ranking_rgb_randomized_500"
  "blocks_ranking_size_clean_50"
  "blocks_ranking_size_randomized_500"
  "click_alarmclock_clean_50"
  "click_alarmclock_randomized_500"
  "click_bell_clean_50"
  "click_bell_randomized_500"
  "dump_bin_bigbin_clean_50"
  "dump_bin_bigbin_randomized_500"
  "grab_roller_clean_50"
  "grab_roller_randomized_500"
  "handover_block_clean_50"
  "handover_block_randomized_500"
  "handover_mic_clean_50"
  "handover_mic_randomized_500"
  "hanging_mug_clean_50"
  "hanging_mug_randomized_500"
  "lift_pot_clean_50"
  "lift_pot_randomized_500"
  "move_can_pot_clean_50"
  "move_can_pot_randomized_500"
  "move_pillbottle_pad_clean_50"
  "move_pillbottle_pad_randomized_500"
  "move_playingcard_away_clean_50"
  "move_playingcard_away_randomized_500"
)

ROBOTWIN_ALOHA_10_DATASETS=(
  "adjust_bottle_clean_50"
  "adjust_bottle_randomized_500"
  "beat_block_hammer_clean_50"
  "beat_block_hammer_randomized_500"
  "blocks_ranking_rgb_clean_50"
  "blocks_ranking_rgb_randomized_500"
  "blocks_ranking_size_clean_50"
  "blocks_ranking_size_randomized_500"
  "click_alarmclock_clean_50"
  "click_alarmclock_randomized_500"
  "click_bell_clean_50"
  "click_bell_randomized_500"
  "dump_bin_bigbin_clean_50"
  "dump_bin_bigbin_randomized_500"
  "grab_roller_clean_50"
  "grab_roller_randomized_500"
  "handover_block_clean_50"
  "handover_block_randomized_500"
  "handover_mic_clean_50"
  "handover_mic_randomized_500"
)

case "${ROBOTWIN_ALOHA_SET}" in
  10)
    ROBOTWIN_ALOHA_DATASETS=("${ROBOTWIN_ALOHA_10_DATASETS[@]}")
    ;;
  all | 15)
    ROBOTWIN_ALOHA_DATASETS=("${ROBOTWIN_ALOHA_ALL_DATASETS[@]}")
    ;;
  *)
    echo "Invalid ROBOTWIN_ALOHA_SET=${ROBOTWIN_ALOHA_SET}. Use 10 or all." >&2
    exit 2
    ;;
esac

missing=()
incomplete=()

for dataset in "${ROBOTWIN_ALOHA_DATASETS[@]}"; do
  dataset_dir="${ALOHA_DATA}/${dataset}"
  version_dir="${dataset_dir}/1.0.0"

  if [[ -d "${version_dir}" ]]; then
    continue
  fi

  if compgen -G "${dataset_dir}/incomplete.*" > /dev/null; then
    incomplete+=("${dataset}")
  else
    missing+=("${dataset}")
  fi
done

echo "RoboTwin ALOHA TFDS root: ${ALOHA_DATA}"
echo "Dataset set: ${ROBOTWIN_ALOHA_SET}"
echo "Expected datasets: ${#ROBOTWIN_ALOHA_DATASETS[@]}"

if (( ${#missing[@]} > 0 )); then
  printf 'Missing datasets:\n'
  printf '  %s\n' "${missing[@]}"
fi

if (( ${#incomplete[@]} > 0 )); then
  printf 'Incomplete datasets:\n'
  printf '  %s\n' "${incomplete[@]}"
fi

if (( ${#missing[@]} > 0 || ${#incomplete[@]} > 0 )); then
  exit 1
fi

echo "All RoboTwin ALOHA datasets are complete."
