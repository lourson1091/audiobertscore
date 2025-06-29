# AudioBERTScore
**AudioBERTScore: Objective Evaluation of Environmental Sound Synthesis Based on Similarity of Audio Embedding Sequences**  [[Paper]](https://)

![Overview](./overview.png)

---
Abstract:

We propose AudioBERTScore, a novel objective evaluation metric designed for text-to-audio (TTA) generation. Unlike traditional metrics such as mel-cepstral distortion (MCD), which show limited correlation with human perception, AudioBERTScore estimates the perceptual similarity between synthesized and reference audio by comparing their embedding sequences obtained from pretrained audio foundation models. Inspired by BERTScore from NLP, our method computes similarity using both max-norm and a generalized p-norm formulation, effectively capturing the non-local characteristics of environmental sounds. Through extensive experiments, we demonstrate that AudioBERTScore achieves significantly higher correlation with subjective evaluation scores—such as overall quality (OVL) and text relevance (REL)—compared to conventional objective metrics. This approach enables robust, reference-based evaluation without supervised training, offering a practical and semantically-aware alternative for assessing the quality of TTA outputs.

---

## Install
NOTE: Our implementation was developed using Python3.8.18, PyTorch 1.13.1 with CUDA 11.6 support.

To use `audiobertscore`, run the following:

```bash
git clone https://github.com/lourson1091/audiobertscore.git

pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1+cu116 -f https://download.pytorch.org/whl/torch_stable.html

pip install -r requirements.txt
```
 ### Pretrained weight files
Please download the pretrained weight files for AST, ATSTFrame, and BYOL-A from the links below, and place them in the same directory as <code>audiobertscore.py</code>.

- AST: <code>audioset_10_10_0.4593.pth</code>

    Download from:
    https://github.com/YuanGongND/ast?tab=readme-ov-file#pretrained-models

    please download following:

    <code>Full AudioSet, 10 tstride, 10 fstride, with Weight Averaging (0.459 mAP)</code>

- ATST-Frame:<code>atstframe_base.ckpt</code>

    Download from:
    https://github.com/Audio-WestlakeU/audiossl/tree/main/audiossl/methods/atstframe

- BYOL-A:<code>AudioNTT2022-BYOLA-64x96d2048.pth</code>

    Download from:
    https://github.com/nttcslab/byol-a/tree/master/pretrained_weights





---
## Usage
### Parameter description
- sr: Sampling rate used internally for feature extraction. Input waveforms will be resampled to this rate if needed. (default: 16000)

- model_type: Specifies which model to use for feature extraction. 

    Choose from:

    - <code>"ast"</code>: AST

    - <code>"atstframe"</code>: ATSTFrame

    - <code>"byola_v2"</code>: BYOL-A v2

- layer: Transformer layer index (1–13) to extract embeddings from.     Used only for "ast" and "atstframe".

- byola_mode(used only when model_type="byola_v2"): Output mode for BYOL-A embeddings. 

    Options:

    - <code>"local"</code>: Local (framewise) embeddings

    - <code>"global"</code>: Global summary embedding

    - <code>"concat"</code>: Concatenated local and global

- lam: Weighting factor λ in the AudioBERTScore formula to balance precision/recall terms. (e.g., -3.5)

- p: The exponent parameter used in the second term of the AudioBERTScore formula. (e.g., 106.0)

    When p approaches infinity, the p-norm becomes equivalent to the max operation.

- use_gpu: If True, runs on GPU if available. If False, forces CPU execution.

### Example of Usage
Please use <code>test.py</code>. The following initial values yield the best performance.

```python
import numpy as np
from audiobertscore import AudioBERTScore

# Example waveforms.
ref_wav = np.random.rand(10009)
gen_wav = np.random.rand(10003)

scorer = AudioBERTScore(
        sr=16000,
        model_type="ast",
        layer=13, 
        byola_mode="concat",
        lam=-3.5,
        p=106.0,
        use_gpu=True
    )

# Calculate socre.
scores = scorer.score(ref_wav, 16000, gen_wav, 16000)
global_p, global_r, global_f1 = scores[0]
```

## Citation
```tex
(TBA)
```

## Contributors
- Minoru Kishi (Keio University, Japan)
- Ryosuke sakai (Keio University, Japan)
- [Shinnosuke Takamichi](https://sites.google.com/site/shinnosuketakamichi/home) (Keio University & The University of Tokyo, Japan)
- Yusuke Kanamori (The University of Tokyo, Japan)
- [Yuki Okamoto](https://sites.google.com/view/yuki-okamoto/home)(The University of Tokyo, Japan)

