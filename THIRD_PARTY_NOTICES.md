# Third-party notices

## H3 artifact preparation dependency

`scripts/prepare_h3_artifact.py` invokes, but does not vendor, Hugging Face
Diffusers' `scripts/convert_minimax_h3_to_diffusers.py` from commit
`aaf57d03a3edfbd9de457d2872417a92f149a91d` (Apache License 2.0).
Source: https://github.com/huggingface/diffusers/tree/aaf57d03a3edfbd9de457d2872417a92f149a91d
License: https://github.com/huggingface/diffusers/blob/aaf57d03a3edfbd9de457d2872417a92f149a91d/LICENSE
The tool verifies the external checkout's converter, LICENSE and installed H3
component source hashes before execution. Its key-plan/config APIs provide the
upstream mappings; no upstream transformer/VAE/converter implementation is
copied into this repository. Original MiniMax model weights are not distributed
here and remain subject to their own license. Conversion is not GPU validation.

## World-R1 camera trajectory utilities

`visual_rl/algorithms/conditioning/camera_math.py` contains the camera trajectory
utilities derived from Microsoft World-R1 under the MIT License. The v0.8
`visual_rl/algorithms/conditioning/world_r1_camera.py` component consumes those
utilities; it does not change their license or provenance.

Copyright (c) Microsoft Corporation.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## World-R1 reward service

`services/world_r1_strict/native/general_reward.py`,
`services/world_r1_strict/native/reward_3d.py`, and
`services/world_r1_strict/native/reward_3d_backend.py` are derived from
Microsoft World-R1 and are distributed under the MIT License. The complete
license text is included at
`services/world_r1_strict/licenses/WORLD_R1_LICENSE`.

## World-R1 prompt subset

`data/prompts/world_r1_dynamic_v1.txt` contains the first 20 prompts from
Microsoft World-R1's `dataset/final/dynamic.txt`, preserved in upstream order
under the MIT License. Its exact source and subset digests are recorded in
`data/prompts/world_r1_provenance_v1.json`; the subset is a bounded Phase A
execution fixture, not the complete upstream training dataset.

## Depth Anything 3

`services/world_r1_strict/native/depth_anything_3/` is derived from
Depth Anything 3 by ByteDance Ltd. and/or its affiliates and is distributed
under the Apache License, Version 2.0. The complete license text is included
at `services/world_r1_strict/licenses/DEPTH_ANYTHING_3_LICENSE`.

The DA3-GIANT model weights are not distributed with framecode. They are
licensed separately under CC BY-NC 4.0 and must be supplied as local model
data for non-commercial research use.

## OpenAI CLIP tokenizer vocabulary

`services/world_r1_strict/native/assets/bpe_simple_vocab_16e6.txt.gz` is
copied from OpenAI CLIP and distributed under the MIT License. It is bundled
because the PyPI `hpsv2==1.2.0` wheel omits this runtime tokenizer resource.
The service verifies its size and SHA-256 digest before installing it into the
isolated reward environment, and therefore never downloads it at startup.
The complete license text is included at
`services/world_r1_strict/licenses/OPENAI_CLIP_LICENSE`.
