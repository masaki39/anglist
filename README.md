# Sagittal Measure Assist (3D Slicer Extension)

脊柱側面X線で5ランドマークを手動計測し、同じデータを学習用にエクスポートできる拡張（`.npy/.nrrd/.json`）。外部で学習したモデルはONNXにしてSlicerで推論に使う想定。

## Slicerでの使い方
1) 側面X線Volumeを読み込み。  
2) Markups Fiducialを選択/作成し、順に 5 点（L1_ant, L1_post, S1_ant, S1_post, FH）を配置。  
3) 「計測を更新」で PI/PT/SS/LL を確認（左右反転が必要ならチェック）。  
4) エクスポートセクションで出力先とケースID（または自動採番）を指定し「エクスポート」。`.npy`（画像配列）, `.nrrd`（元Volume）, `.json`（IJK座標と角度/メタデータ）を保存。

### エクスポートの中身（`.json`）
- `landmarks_ijk`: 各ランドマークのI/J/K（ピクセル空間）  
- `metadata`: `spacing`, `ijk_to_ras` 行列, `origin_ras`  
- `image_shape`: `.npy` のshape  
- `angles_deg`: PI/PT/SS/LL  
- `flip_x_axis`: 左右反転補正の有無

## ディレクトリ構成
- `SagittalMeasureAssist/` — エントリーポイントとUI分割。  
  - `logic_angles.py`, `logic_export.py`, `ui_measure.py`, `ui_export.py`, `assist_controller.py`  
- `train/` — 外部学習用スクリプト（PyTorch/ONNX）。  
- `CMakeLists.txt` — Slicer拡張のエントリーポイント。

## テスト（純Python）
- 事前準備不要: `uv run python -m pytest`

## 学習パイプライン（外部uv環境）
- 依存インストール（CPU想定）: `uv sync --extra ml`  
- 学習（縦横比を保ちパディングしてリサイズ）:  
  `uv run python train/train.py --data-dir /path/to/exported --save-dir runs --epochs 20`  
  - 入力: `*_image.npy`, `*_landmarks.json`（Slicerエクスポート）  
  - モデル: 軽量UNet、出力5チャネルのヒートマップ  
  - 出力: `runs/best.pt`, `runs/last.pt`  
- ONNXエクスポート:  
  `uv run python train/export_onnx.py --checkpoint runs/best.pt --output runs/best.onnx --height 512 --width 512`  
- ONNX簡易推論（onnxruntime）:  
  `uv run python train/infer_onnx.py --model runs/best.onnx --image sample_image.npy --json sample_landmarks.json`

### モデルロジック（初心者向け）
- 画像を1chに正規化 → 縦横比維持でリサイズ＋余白パディング → 512x512（デフォルト）。  
- 座標も同じスケール＆パディング量で変換し、各点に2Dガウスを置いた5枚のヒートマップを教師信号に。  
- 軽量UNetが5チャネルのヒートマップを出力し、MSEで学習。  
- ONNXに書き出せば、Slicer側でONNX Runtimeを使い、ヒートマップの最大値をMarkupsに置くだけで自動配置に使える。

## Slicer側のONNX推論（自動配置）
- モデル: `train/export_onnx.py` で出力した `.onnx` を指定。  
- 操作: モジュール内「自動推論 (ONNX)」セクションでモデルパスと入力サイズ(学習時と同じ値)を設定→「推論してMarkupsに配置」。  
- 処理: Volumeの1スライス目を正規化・パディングリサイズ→ONNX推論→ヒートマップ最大値を元画像座標へ逆変換→Markupsに5点を自動配置→計測テーブル更新。  
- 注意: モデルの入力サイズは学習時の値に合わせてください（デフォルト512x512）。Slicer環境に`onnxruntime`が無い場合は事前にインストールが必要です。
