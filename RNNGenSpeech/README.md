# RNNGenSpeech
RNNの使用例として、有名なスピーチテキストをお手本に、同じような文章を自動生成してみます。
文章生成では文を単語列として扱うのが一般的ですが、前処理が面倒なのと日本語を対象にしたいので、あえて文字単位で処理します。
作りたいネットワークは例えば以下のようなものです。

<img alt="elman net例" src="./ElmanNet4Inference.png" title="推論に使うElemanネット" size="25%"/>

一番上のInputは一文字分の文字コード、

## 訓練方式
以下の２種類を比較できるようにしてあります。
- 一文字方式：お手本テキストの中の、２０文字を読んで２１文字目を推定するように学習
- 各文字方式：お手本テキストの中の、２０文字を読んで、各文字の次の文字を推定するように学習

どちらの訓練方式でも、推論時は１文字読んで次の文字を推定します。
各文字方式の方が、同じ数の訓練データに対して訓練効率が高いようです。

## ファイル一覧
### pythonコード
- create_RNNtrain_data.py：テキストファイルから一文字方式の訓練データを作成
- create_RNNtrain_dataW.py：テキストファイルから各文字方式の訓練データを作成
- show_dict.py：文字コード辞書の中身を確認
- traindata2str.py：訓練データの中身を確認（両方式とも利用可能）
- output2str.py：一文字方式での訓練で作成されたoutput_result.csvをコード変換して確認

- generate_text.py：Elman型のネットワークで推論（テンプレート）
- generate_text_LSTM.py：LSTM１層のネットワークで推論（テンプレート）
- generate_text_2LSTM.py：LSTM２層のネットワークで推論（テンプレート）

- make_dict.py：文字コード辞書作成ライブラリ

### サンプルテキスト
- ObamaHiroshimaSpeech.txt：オバマ元大統領の広島スピーチ全文
- SteveJobsStanford.txt：ジョブズのスタンフォード大学でのスピーチ全文
- UenoTodaiSpeech+.txt：上野千鶴子の東大入学式祝辞スピーチ全文＋α

### 生成したファイルのサンプル
- ObamaHiroshimaSpeech.pkl：オバマ元大統領スピーチの文字コード辞書
- UenoTodaiSpeech+.pkl：上野千鶴子東大入学式祝辞の文字コード辞書
- OHS_train.csv：オバマ元大統領スピーチの一文字方式の訓練データ
- OHS_test.csv：オバマ元大統領スピーチの一文字方式のテストデータ
- OHSW_train.csv：オバマ元大統領スピーチの各文字方式の訓練データ
- OHSW_test.csv：オバマ元大統領スピーチの各文字方式のテストデータ
- UenoW_train.csv：上野千鶴子東大入学式祝辞の各文字方式の訓練データ
- UenoW_test.csv：上野千鶴子東大入学式祝辞の各文字方式のテストデータ
- generate_text_2LSTM_ueno.py：上野千鶴子東大入学式祝辞学習したLSTM２層のネットワークで推論

### その他
- README.md：このファイル
- ElmanNet4Inference.png：このファイルの画像データ
