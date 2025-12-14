# Sagittal Measure Assist (3D Slicer Extension)

Lateral spine X-ray helper for sagittal alignment measurement and MLデータ作成（計測とエクスポートを1モジュールに統合）。

## モジュール
- `SagittalMeasureAssist`: 5ランドマークを配置して PI/PT/SS/LL を計測し、同じランドマークと角度を `.npy`/`.nrrd`/`.json` でエクスポート。

## 使い方
1) 側面X線Volumeを読み込み。  
2) モジュール内で Markups Fiducial を選択/作成し、順に 5 点（L1_ant, L1_post, S1_ant, S1_post, FH）を配置。  
3) 「計測を更新」で PI/PT/SS/LL を確認（左右反転補正が必要ならチェック）。  
4) エクスポートセクションで出力先とケースID（または自動採番）を設定し「エクスポート」。`.npy`（画像配列）, `.nrrd`（元Volume）, `.json`（IJK座標と角度/メタデータ）を保存。

## ディレクトリ構成
- `SagittalMeasureAssist/` — エントリーポイントとUI分割。  
  - `logic_angles.py` — 角度計算ヘルパー。  
  - `logic_export.py` — エクスポート処理（IJK/JSON/npy/nrrd）。  
  - `ui_measure.py` — 計測UIセクション。  
  - `ui_export.py` — エクスポートUIセクション。  
- `CMakeLists.txt` — Extensionエントリーポイント。

## Roadmap
- ONNX Runtime等による自動ランドマーク提案の追加。  
- エクスポート時の簡易QA（オーバーレイや不足チェック強化）。  
- 参考トレーニングスクリプト（PyTorch）の提供。

## テスト（純Python）
- 事前準備不要で、毎回ワンショット実行:  
  `uv run python -m pytest`
