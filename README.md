# 日本取引所グループ ファンダメンタルズ分析チャレンジ 1位解法

## 入力データについて
訓練とモデルの検証に用いるデータは、規約により再配布できないため、本レポジトリには含めていません。

## 実行環境
docker-composeの利用が前提です。また、一部のコードは、下記の環境変数を参照します。:

- HOST_UID: ホスト側のユーザーのUID
- HOST_GID: ホスト側のユーザーのGID
- REFRESH_TOKEN: J-Quants APIの reflesh roken

## コードの概略
- script/get_data_via_api.ipynb: J-Quants APIによる最新データの取得
- script/preprocess.ipynb: 訓練のための前処理
- script/train.ipynb: モデルの訓練と交差検証
- script/test_predictor.ipynb: public LBの再現
- archive/src/*.py: 投稿されるコード一式
- mk_submission.bash: 投稿用zipファイルの作成

## リンク
- competition page: https://signate.jp/competitions/423/submissions
- tutorial: https://japanexchangegroup.github.io/J-Quants-Tutorial/
- official document of the runtime: https://signate.jp/features/runtime/detail
- api doc: https://jpx-jquants.com/apidoc.html
- この解法の説明資料: https://speakerdeck.com/m_mochizuki/the-provisional-1st-place-solution-of-jpx-fundamentals-analysis-challenge-on-signate 
