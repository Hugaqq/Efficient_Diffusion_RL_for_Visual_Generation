# Third-party notices

## World-R1 camera trajectory utilities

`visual_rl/model_adapters/world_r1_camera.py` is derived from the
World-R1 camera trajectory utilities distributed by Microsoft under the
MIT License.

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
