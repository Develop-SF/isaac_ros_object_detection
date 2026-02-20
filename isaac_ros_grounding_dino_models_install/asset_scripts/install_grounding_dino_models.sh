#!/bin/bash
# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Download and convert Grounding DINO models to TensorRT engines
# Models will be stored in the isaac_ros_assets directory
# Setup paths
set -e
if [ -n "$TENSORRT_COMMAND" ]; then
  # If a custom tensorrt is used, ensure it's lib directory is added to the LD_LIBRARY_PATH
  export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:$(readlink -f $(dirname ${TENSORRT_COMMAND})/../../../lib/$(uname -p)-linux-gnu/)"
fi
if [ -z "$ISAAC_ROS_WS" ] && [ -n "$ISAAC_ROS_ASSET_MODEL_PATH" ]; then
  ISAAC_ROS_WS="$(readlink -f $(dirname ${ISAAC_ROS_ASSET_MODEL_PATH})/../../..)"
fi
ASSET_NAME="grounding_dino"
MODELS_DIR="${ISAAC_ROS_WS}/isaac_ros_assets/models/${ASSET_NAME}"
EULA_URL="https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/grounding_dino"
ASSET_DIR="${MODELS_DIR}"
ASSET_INSTALL_PATHS="${ASSET_DIR}/grounding_dino_model.plan"


source "${ISAAC_ROS_ASSET_EULA_SH:-isaac_ros_asset_eula.sh}"

# Create directories if they don't exist
mkdir -p ${MODELS_DIR}

# Download Grounding DINO model
GROUNDING_DINO_URL="https://api.ngc.nvidia.com/v2/models/nvidia/tao/grounding_dino/versions/grounding_dino_swin_tiny_commercial_deployable_v1.0/files/grounding_dino_swin_tiny_commercial_deployable.onnx"
GROUNDING_DINO_ONNX="${MODELS_DIR}/grounding_dino_model.onnx"
GROUNDING_DINO_ENGINE="${MODELS_DIR}/grounding_dino_model.plan"

isaac_ros_common_download_asset --url "${GROUNDING_DINO_URL}" --output-path "${GROUNDING_DINO_ONNX}" --cache-path "${ISAAC_ROS_GROUNDING_DINO_MODEL}"
MODEL_DOWNLOAD_RESULT=$?
if [[ -n ${ISAAC_ROS_ASSETS_TEST} ]]; then
  if [[ ${MODEL_DOWNLOAD_RESULT} -ne 0 ]]; then
    echo "ERROR: Remote model does not match cached model."
    exit 1
  fi
  exit 0
elif [[ ${MODEL_DOWNLOAD_RESULT} -ne 0 ]]; then
  echo "ERROR: Failed to download model."
  exit 1
fi

${TENSORRT_COMMAND:-/usr/src/tensorrt/bin/trtexec} \
    --onnx=${GROUNDING_DINO_ONNX} \
    --saveEngine=${GROUNDING_DINO_ENGINE}
