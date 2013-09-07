---
layout: post
title: "Super-Bit LSH"
date: 2013-01-26 13:54
comments: true
categories: ハッシュ 機械学習
---

今日は、すこし前に読んだSuper-Bit LSH (SB-LSH)という手法を簡単に紹介します。
これは角類似度[^1]に対するランダム射影LSH[^2]の改良版です。

[^1]: 2つのベクトル $x, y$ に対して $1-{1\over\pi}\cos^{-1}\left({x\cdot y\over\lVert x\lVert\,\lVert y\lVert}\right)\in[0, 1]$ を角類似度(angular similarity)と呼びます。
    Cosine類似度やArccos類似度と呼ばれることもあります。

[^2]: SimHash、Cosine LSH、Arccos LSHなどいろんな呼ばれ方をしています。
    論文中ではSRP-LSH (Sign-random-projection LSH) と呼んでいます。

Jianqiu Ji, Jianmin Li, Shuicheng Yany, Bo Zhang, Qi Tianz.\\
Super-Bit Locality-Sensitive Hashing. NIPS 2012.

ランダム射影LSHの解説は[海野さんのスライド(29-30ページ目)](http://blog.jubat.us/2012/05/17-web.html "第17回 データマイニング+WEB＠東京で発表しました | Jubatus Blog")がわかりやすいです。
ランダム射影LSHを使うと、角類似度の不偏推定量が得られます。

Super-Bit LSHではこれをNビットごとにグループ分けして、各グループ内の射影ベクトルを直交させるというものです。
直交化はたとえばGram-Schmidtの直交化法を使います。
こうすると、推定量の分散がただのランダム射影よりも小さくなるということです。
論文ではこのNビットのグループをSuper Bitと呼んでいます。

論文中の実験を見る限り、特に直角に近い角度の推定が正確になるようです。
これは近傍探索に使うときにはあまり問題にならないですが、別の用途で純粋に角度の推定値が欲しい時や、Active Learningに用いるとき[^3]には役立ちそうです。

[^3]: 重みベクトルとできるだけ直交する教師データを選びたいときなど。

ただ、次元がとても高い場合、ランダムなベクトル同士がほとんど直交する（大数の法則）ので、あまり効果はないかもね、と書かれています。
論文中の実験ではSIFT特徴量(128次元)とBoVW(3125次元)を使っているようで、これくらいだと有効みたいです。
また、自然言語処理などでよく事前に次元数がわからない場合がありますが、そういう場合も使えなさそうです。

ランダム射影LSHはLSHの中でもかなり古典的なアルゴリズムですが、データ非依存という設定でその精度が改良されるのはおそらく初めてだと思います（論文中でもそう書かれています）。
今まではいろんな種類の類似度に対するLSHの開発が盛んでしたが、そろそろ既存のLSHを改良する
