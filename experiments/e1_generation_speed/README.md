# E1(前半): Z-Image-Turbo の生成時間

計画書 §6 の **E1**、マイルストーン **M0** に対応する。

## 何を知りたいのか

| 知りたいこと | なぜ |
|---|---|
| 解像度別の end-to-end 生成時間 | 1ターンの予算は「生成 + 事前チェックで5秒以内」(RULES.md §2.1)。生成が何秒で収まるかで、CLIP・VLM に回せる時間が決まる |
| 工程ごとの内訳 | テキストエンコーダは Qwen3(4B)。ここが支配的なら、解像度を下げても効かない |
| **実際の NFE(transformer の forward 回数)** | M1 で取れる x0 予測の**上限枚数**がこれで決まる。見せたいフレームは6〜7枚が目安 |
| ピーク VRAM | 48GB のうち、CLIP・VLM・ダミー生成用LLM を同居させられる余地(計画書 §4.2) |

## 食い違いを1つ確定させる

計画書 §3.1 は「公式サンプルは `num_inference_steps=9` だが、実際のモデル評価は8回」と
書いている(HF モデルカードの記述)。一方 **diffusers 公式ドキュメントの `ZImagePipeline`
の例は `num_inference_steps=8`** を使っており、記述が食い違っている。

どちらが何枚の x0 予測を生むかは、フレーム設計の土台なので推測で決めない。
このスクリプトは **transformer の forward に hook を付けて実回数を数える**ので、
`--steps 8 9` の結果を見れば確定する。

## 実行方法

```bash
# 既定: 512/640/768/1024 × steps 8/9、各5回(warmup 2回)
.venv/bin/python experiments/e1_generation_speed/measure_speed.py

# 解像度と試行回数を絞る
.venv/bin/python experiments/e1_generation_speed/measure_speed.py --resolutions 512 768 --repeats 5

# 工程別の内訳を測らない(end-to-end だけ)
.venv/bin/python experiments/e1_generation_speed/measure_speed.py --no-breakdown
```

初回はモデル(約20GB)のダウンロードが入る。`~/.cache/huggingface` に入る。

## 出力

`results/<timestamp>/` に3つ。

- `summary.md` — 人が読む表。予算判断にはこれを見る。
- `results.json` — 生データ。計測条件(GPU、ドライバ、ライブラリ版、同居プロセス)込み。
- `samples/` — sanity check 用のサンプル画像(解像度 × steps ごとに1枚)。

## 測り方の約束

- **warmup を捨てる**。初回はCUDAカーネルの準備などで遅く、本番の性能ではない。
- **end-to-end と内訳を分けて測る**。内訳は工程の境界でGPUを同期するため、
  その分 end-to-end が悪化する。**予算の判断には内訳なしの数値を使う**。
- **計測開始時の他プロセスを記録する**。同じGPUで他の処理が動いていれば数値は信用できない。
  このマシンはディスプレイ出力でも1.5GB程度を使っている。
- **プロンプトはゲームと同じ長さの英文**(既定 `A dog chases a cat`)。プロンプト長は
  テキストエンコードの時間に効くため、長い作例で測ると実態とずれる。
- 報告値(「RTX 4090 で1024×1024が約2.3秒」)と**実測値を混ぜない**。1024 を測るのは、
  その報告値と比較して環境が妥当かを見るため。ゲームで使う解像度は512〜768の想定。

## このあと

- 生成時間が5秒の予算に収まらない場合 → 解像度を下げる、または計画書 §3.1 末尾の
  モデル再検討の条件に該当するか判断する。
- 序盤の x0 予測がすでに鮮明すぎる場合(DMD系の蒸留の影響)→ M1 で類似度曲線を見て、
  線形補間か Z-Image-Base を検討する。**先に決め打ちしない。**
