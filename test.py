import numpy as np
from audiobertscore import AudioBERTScore

# ランダムな波形（例: 疑似音声）
ref_wav = np.random.rand(10009)
gen_wav = np.random.rand(10003)

# AudioBERTScore インスタンス作成
scorer = AudioBERTScore(
        sr=16000,
        model_type="ast",
        layer=13, 
        byola_mode="concat",
        lam=-3.5,
        p=106.0,
        use_gpu=True
    )

# 類似度スコアを計算
scores = scorer.score(ref_wav, 16000, gen_wav, 16000)
global_p, global_r, global_f1 = scores[0]

# 結果表示
print(f"Global Precision: {global_p:.3f}")
print(f"Global Recall:    {global_r:.3f}")
print(f"Global F1:        {global_f1:.3f}")