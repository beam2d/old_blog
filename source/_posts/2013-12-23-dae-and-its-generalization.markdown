---
layout: post
title: "Denoising Autoencoderとその一般化"
date: 2013-12-23 22:48
comments: true
categories: [機械学習, 深層学習]
---

[Machine Learning Advenc Calendar 2013](http://qiita.com/advent-calendar/2013/machinelearning)の23日目担当の得居です。
[株式会社Preferred Infrastructure](https://preferred.jp)で[Jubatus](http://jubat.us)を作ったりしています。

今日は深層学習(deep learning)の話です。
深層学習はこの2年ほどで専門外の人にも知れ渡るほどに大流行しました。
データさえ大量にあればテクニック次第で他の手法を圧倒する性能を達成できることから、特に大量のデータを持つ大企業において大々的な参入が相次ぎました。

主に流行っているのは教師あり学習です。
補助として教師なし学習による事前学習(pretraining)も、特に音声認識のタスクにおいては行われているようですが、画像認識を中心に事前学習なしでもテクニック次第で学習できるという見方が強まっています。

一方で教師なしデータからの学習はブレイクスルー待ちといった雰囲気です。
Deep Belief NetworksやDeep Boltzmann Machinesなど「無名隠れ変数 (anonymous latent variables)」を用いた生成モデルが先行して詳しく解析されていますが、その学習は教師あり学習のようにはうまくいっていないのが現状です[^1]。

[^1]: といっても僕自身これらを自分で実装して動かしたわけではなく、論文追っかけてるだけなので誰かの受け売り状態なのですが。。。

今日お話するDenoising Autoencoder (DAE)はそんな教師なし深層学習で用いられる手法・モデルの一つです。
RBM系の深層生成モデルとは別の系統になりますが、DAEをベースにした確率的な深層モデルの理論が数年遅れて最近発展してきているので、今日はその概要をまとめて書きたいと思います。
主にPascal VincentとYoshua Bengioによるこの5年間の研究の流れに沿って、以下の様な順に書きます。

- Autoencoderの定式化
- DAEの定式化とそのオリジナルの解釈
- DAEのスコアマッチングとしての解釈
- DAEの確率モデルとしての一般化
- 一般化DAEのためのWalkbackアルゴリズム
- 隠れ変数を入れたさらに一般のモデルとその特徴

分量の割に時間がかかっていないので、説明が変だったりまどろっこしい部分があるかもしれません。
気づいた点がございましたらぜひ[@beam2d](https://twitter.com/beam2d)までお知らせください。

<!-- more -->

## Autoencoder (AE)

まずAEの定式化を紹介します。
入力層 $x\in\mathbb R^d$ に対して隠れ層 $y=f\_\theta(x)=s(Wx+b)\in\mathbb R^{d_\text{h}}$ と復元層 $x^r=g\_{\theta'}(y)=s_r(W'y+c)\in\mathbb R^d$ を定義します。
ただし $\theta=(W, b)$, $\theta'=(W', c)$ は共に学習すべきパラメータです。
$s, s\_r$ は活性化関数(activation function)と呼ばれます。
$s\_r$ は入力がバイナリ値ならシグモイド関数、一般の実数値なら恒等関数にします。
$W'=W^\top$ という制約(tied weights)を置くと性能が良いことが知られています（解釈の一つが後ほど出てきます）。
関数 $f\_\theta$ はエンコーダ、$g\_{\theta'}$ はデコーダと呼ばれます。

こうして定義された復元層 $x^r$ が入力層 $x$ にできるだけ近くなるようにエンコーダとデコーダのパラメータ $\theta, \theta'$ を学習します。
訓練データ $\mathcal D=\{x\_1, \dots, x\_n\}$ に対して損失関数 $L(x, x^r)$ の平均値を最小化します:

$$\min_{\theta, \theta'} {1\over n}\sum_{i=1}^nL(x_i, g(f(x_i))).$$

損失関数 $L$ としては、入力がバイナリ値ならば交差エントロピー誤差、一般の実数値ならば二乗誤差を用いるのが一般的です。

AE自体は1980年代から知られていたようです(要出典)。
オリジナルのAEには ${d\le d\_\textrm{h}}$ のときに恒等関数が最適になってしまうという問題点があります。
この問題を回避するために $d>d\_\textrm{h}$ としたり（ボトルネック型のAE）、正則化項を加えたものを用いたりしていました。

## Denoising Autoencoder (DAE)

DAE [1]は正則化項とは異なるアプローチで2008年にPascal Vincentらが提案したAEの亜種です。
入力の一部を破壊することで、恒等関数が最適でないような問題に変形します。

入力ベクトルの一部にノイズを加える破壊分布(corruption distribution) $\mathcal C(\tilde X\vert X)$ を考えます。
破壊分布としてはガウスノイズや、成分をランダムに選んで0や1に潰す塩胡椒ノイズ(salt and pepper noise)などが用いられます。

入力 $X=x$ に対して $\mathcal C(\tilde X|X)$ から $\tilde X$ をサンプリングして、そこからエンコーダ・デコーダを使って入力 $X$ を復元します。
入力データを生成する分布を $\mathcal P(X)$ とします。
このときDAEは以下の期待値最適化問題として書かれます。

$$\min_{\theta, \theta'} {\mathbb E}_{X\sim\mathcal P(X), \tilde X\sim\mathcal C(\tilde X\vert X)}L(X, g(f(\tilde X))).$$

これは同時分布 $\mathcal P(X, \tilde X)=\mathcal P(X)\mathcal C(\tilde X\vert X)$ 上の確率的最適化問題なので、確率的勾配降下法 (SGD)などで最適化できます。

関数 $g\circ f$ は破壊された（ノイズの乗った）入力 $\tilde X$ をもとの入力 $X$ に復元するように学習されるので、提案者であるP. VincentはこれをDenoising Autoencoderと名づけました。

### DAEの様々な解釈

DAEの見た目はAEのちょっとした亜種ですが、確率的な操作が関わることで多様な解釈を生み出しています。
例えばVincentの元論文ではDAEに対するいくつかの解釈が述べられています。

- **多様体学習**
  高次元空間 $\mathbb R^d$ 上のデータ分布 $\mathcal P(X)$ は低次元多様体に埋まっているとする仮説があります。
  DAEは多様体から少しずれた位置にある $\tilde X$ を多様体上に引き戻すように学習されるので、この低次元多様体を学習していることに相当します。
  DAEはさらに、多様体の法線方向へのずれに対してロバストな写像を学習すると考えることができます。
  この考え方はContractive Autoencoderへと受け継がれています[2]。
- **生成モデル**
  バイナリ値の入力の場合を考えます。
  $f, g$ を用いて二つの生成モデルを考えます。
  - エンコーディングのモデル $q^0(X, \tilde X, Y)=q^0(X)\mathcal C(\tilde X\vert X)\delta\_{f\_\theta(\tilde X)}(Y)$。ここで $q^0(X)$ は経験分布、$\delta$ は添字を平均とするデルタ分布。
  - デコーディングのモデル $p(X, \tilde X, Y)=p(Y)p(X\vert Y)\mathcal C(\tilde X\vert X)$。ここで $p(Y)$ は $\[0, 1\]^{d\_\text{h}}$ 上の一様分布、$p(X\vert Y)$ は $g\_{\theta'}(Y)$ を平均とするベルヌーイ分布。

  このとき、ごにょごにょ計算すると、DAEの誤差最小化問題は二つのモデルの間の交差エントロピー $\mathbb E\_{q^0(\tilde X)}\[-\log p(\tilde X)\]$ の変分下界最大化と等価であることが示せます。
- **情報理論**
  DAEの最小化問題は他にも入力層 $X$ と隠れ層 $Y$ の相互情報量 $I(X; Y)$ の下界を最大化していることとも等価になりそうです。

またVincentの2011年の論文[3]ではスコアマッチングとの関係が示されています。
**スコアマッチング (Score Matching)** [4]とは分配関数を計算しないで確率モデルのパラメータを推定するための手法の一つです。
確率分布 $p(\xi; \theta)$ のスコア関数を $\psi(\xi; \theta)=\nabla\_{\xi}\log p(\xi; \theta)$ と定義します[^2]。
つまりスコア関数は入力に対する対数密度の勾配です。
これは真の分布 $p\_x(\xi)$ についても考えることができます: $\psi\_x(\xi)=\nabla\_{\xi}\log p\_x(\xi)$。
これら2つのスコア関数の期待二乗距離を最小化するのがスコアマッチングです。

$$\min_{\theta}\frac12\int\lVert\psi(\xi; \theta)-\psi_x(\xi)\lVert^2dp_x(\xi).$$

$\psi\_x$ が計算できないように見えますが、ごにょごにょ計算するとこれをモデルの2階微分を使った式で置き換えることができます。
スコアマッチングは入力に対する対数勾配を取ることで、分配関数を消しているところが特徴です（分解関数は入力に依存しないため、入力で微分すると消えます）。

[^2]: この定義は統計学におけるもともとの「スコア関数』の定義とは異なるそうです。もともとは確率モデルの対数密度勾配を、入力変数ではなくモデルのパラメータについて取ったものがスコア関数でした。

[3]では、入力が実変数で損失関数が二乗誤差の場合のDAEが、スコアマッチングにおいて真の分布 $p\_x$ の代わりに各訓練データを平均とする成分（パーゼン窓）からなる混合ガウス分布を用いたものと等価であることを示しています。
その過程でtied weightsが自然に導出されます。
またDAEの目的関数に対応するエネルギー関数も導出されています。

$$E(x; \theta)=-{1\over\sigma^2}\left(\langle c, x\rangle-\frac12\lVert x\lVert^2+\sum_{j=1}^{d_\text{h}}\text{softplus}(\langle W_j, x\rangle+b_j)\right).$$

ここで $\sigma^2$ はパーゼン窓の分散パラメータ、$\text{softplus}(t)=\log(1+e^t)$、$W\_j$ は重み行列 $W$ の $j$ 行目、$b\_j$ はバイアス $b$ の第 $j$ 成分です。
エネルギー関数はRBMのものによく似ています（係数が少し違います）。

## 一般化DAE

これまでに述べたDAEの特徴的な性質は、破壊された入力からもとの入力を復元するという操作から導き出されるものでした。
復元するための関数を2層のニューラルネットに限定せず、また入力や隠れ変数の種類（バイナリか実数かなど）や破壊分布や損失関数の種類に依らない、一般的な確率モデルとして捉え直したものがYoshua Bengioによって今年のNIPSで提案されています [5]。
さらにそのDAEが定めるマルコフ連鎖によって、真のデータ生成分布 $\mathcal P(X)$ 全体を推定することができると主張しています。

真のデータ生成分布を $\mathcal P(X)$、破壊分布を $\mathcal C(\tilde X\vert X)$、パラメータ $\theta$ を持つDAEのモデルを $P\_\theta(X\vert\tilde X)$ とします。
このモデル $P\_\theta(X\vert\tilde X)$ としては、もとのDAEの定式化でいえば $g(f(\tilde X))$ を平均とするベルヌーイ分布やガウス分布などが考えられます（入力変数の種類によって適切なものを使います）。
このとき一般化DAEでは同時分布 $\mathcal P(X, \tilde X)=\mathcal P(X)\mathcal C(\tilde X\vert X)$ 上の負の期待対数尤度 $\mathcal L(\theta)=-\mathbb E\_{\mathcal P(X, \tilde X)}\log P\_\theta(X\vert\tilde X)$ を最小化します。
実際には有限個の訓練データしか使えないので、正則化項を加えた次式を最小化します。

$$\mathcal L_n(\theta)=\frac1n\sum_{X\sim\mathcal P(X), \tilde X\sim\mathcal C(\tilde X\vert X)}\lambda_n\Omega(\theta, X, \tilde X)-\log P_\theta(X\vert\tilde X).$$

ここで $\Omega$ は正則化項、$n$ は訓練データ数、つまり上式の総和の項数です。
この関数 $\mathcal L\_n(\theta)$ を最小化するパラメータを $\theta_n$ とおきます。
ここで正則化の係数 $\lambda_n$ は $\lambda_n\to0\,(n\to\infty)$ となるように選びます。
よって$\mathcal L\_n\to\mathcal L$ となります。

このモデルをもとに、$X$ と $\tilde X$ を交互に生成するマルコフ連鎖を考えます。

$$X_t\sim P_\theta(X\vert\tilde X_{t-1}),\,\tilde X_t\sim\mathcal C(\tilde X\vert X_t).$$

$\theta=\theta\_n$ の場合を考えます。
これを $X\_t$ に関するマルコフ連鎖と考えたときの遷移作用素は $\tilde X$ を積分消去して $T\_n(X\_t\vert X\_{t-1})=\int P\_{\theta\_n}(X_t\vert\tilde X)\,\mathcal C(\tilde X\vert X)\,d\tilde X$ となります。
$T\_n$ が定めるマルコフ連鎖が漸近分布を持つとは限りませんが、持つ場合にはそれを $\pi\_n$ とおきます。
また $T=\lim\_{n\to\infty}T\_n$ とします。
このとき、[5]では以下の定理が成り立つことを示しています。

**定理1** $P\_{\theta\_n}(X\vert\tilde X)$ が $\mathcal P(X\vert\tilde X)$ のconsistent estimatorで、かつ $T\_n$ が定めるマルコフ連鎖がエルゴード的なら、$n\to\infty$ において $\pi\_n(X)$ は $\mathcal P(X)$ に収束する。

つまり上のマルコフ連鎖が既約（任意の状態間を移動できる）で非周期的ならば $P\_{\theta}(X\vert\tilde X)$ を正しく推定することで真のデータ生成分布が得られるということを言っています。

[5]が重要だと主張しているポイントは、破壊された入力 $\tilde X$ に対する $\mathcal P(X\vert\tilde X)$ を推定する方が $\mathcal P(X)$ を直接推定するより簡単だという点です。
$\mathcal P(X)$ は一般に多峰性の複雑な形になりますが、$\mathcal P(X\vert\tilde X)$ は破壊分布 $\mathcal C(\tilde X\vert X)$ が十分局所的なら、ほとんど一つのモード（峰）だけが支配的になり、推定が容易になると言っています。

### Walkbackアルゴリズム

直前の段落で述べたポイントでは $\tilde X$ が $X$ に十分近ければ推定できるという話でしたが、上のマルコフ連鎖においてギブスサンプリングを回しているうちに $\tilde X\_t$ がもとの $X\_0$ から離れていってしまう可能性があります。
特にデータ多様体から離れた位置ではモデルがあまり学習されないため、一度多様体の近くを離れてしまうと誤ったモード(spurious modes)に吸い込まれていく可能性があります。
[5]ではこれに対処するために、複数回ギブスサンプリングを回して得られた $\tilde X\_t$ についても訓練データに入れる（つまり $P\_\theta(X\_0\vert\tilde X\_t)$ も最大化する）という方法を提案しています。
これを、遠くに行ってしまったサンプルをもとの場所に引き戻す、という意味で**Walkbackアルゴリズム**と呼んでいます。
サンプリングを回す回数は適当に決めます（[5]では幾何分布からサンプリングして決めています）。
WalkbackアルゴリズムはもとのDAEと同じ分布を学習することも示されていますが、もとのDAEよりも経験的に性能が良いそうです。

Walkbackアルゴリズムは、モデルから生成されたサンプルを訓練データに使うという意味で、RBMに対するContrastive Divergence（特にCD-$k$アルゴリズム）と似ています。

## Generative Stochastic Network (GSN)

Yoshua Bengioは一般化DAEをさらに一般化した枠組みを提案しています[6]。
一般化DAEの枠組みでは潜在変数が含まれておらず、すべて $P\_\theta(X\vert\tilde X)$ に込められていました。
このモデルに複雑な構造（例えば深層モデル）を持たせるために、一般化DAEに潜在変数を加えたものがGSNです。

GSNでは、一般化DAEにおけるマルコフ連鎖が潜在変数 $H\_t$ を用いて次のように置き換えられます。

$$H_{t+1}\sim P_{\theta_1}(H\vert H_t, X_t),\,X_{t+1}\sim P_{\theta_2}(X\vert H_{t+1}).$$

ここで $H\_{t+1}$ を生成する式はノイズ変数 $Z\_t$ を使って $H\_{t+1}=f\_{\theta\_1}(X\_t, Z\_t, H\_t)$ と表されるとします。
一般化DAEの場合にはこの $P\_{\theta\_1}(H\vert H\_t, X\_t)$ が破壊分布 $\mathcal C(\tilde X\_t\vert X\_t)$ になっています（つまり $\tilde X\_t$ が潜在変数となります）。

[6]ではGSNにおいても一般化DAEと同様の収束性が成り立つことを示しています。
つまり上のマルコフ連鎖に対しても、一般化DAEと同じような仮定のもとで漸近分布がデータ生成分布 $\mathcal P(X)$ に収束します。

GSNではギブスサンプリングの過程で潜在変数 $H\_t$ を記憶することができる点が一般化DAEとの重要な違いのようです。
論文ではDeep Boltzmann Machineのサンプリングの計算グラフと同じように計算が進むようなGSNを例として作っています（[6]Figure 3）。
これは隠れ層が3層あるモデルで、奇数目の層と偶数目の層を交互にサンプリングします。
入力層をサンプリングしたあとには一般化DAEのマルコフ連鎖同様にノイズを加えます。
また、隠れ層の各ユニットは活性化関数の前後にノイズを入れているようです（確率的ニューロンと呼んでいます[7]）。

Walkbackアルゴリズムにより、複数ステップ後の復元結果を学習に使うことができます。
DBMと異なりGSN全体は（確率的な）ニューラルネットなので、この復元層からの学習には誤差逆伝播法が使えます。
これにより、GSN全体は生成モデルでありながら、深いモデルでも教師あり学習の種々のテクニックが利用できるのが利点です。
例えば畳み込みレイヤーやプーリング、AdaGrad、Dropoutなどがそのまま使えます。

また、GSNは入力に欠損値がある場合にも使え、観測されている入力値から欠損値を補完することができます。
観測されている入力値 $X^{(s)}$ を固定した状態でギブスサンプリングを回せば $\mathcal P(X^{(-s)}\vert X^{(s)})$ からサンプリングできます。
ここで $X^{(-s)}$ は残りの入力変数です。

## まとめ

今日はDenoising Autoencoderの定式化と様々な解釈から2段階の一般化（一般化DAEとGenerative Stochastic Network）を紹介しました。
生成モデルの解析周りは個人的に慣れない分野で、今回の話の中にも咀嚼できていない点が多々ありますが、DAEという一つのモデルからこれだけ多様な話題が出てくるのは面白いなあと思っています。
一般化について、特にGSNについては、具体的な応用が来年には出てくるのかなと想像しています。
特に教師あり学習のテクニックが使えるというのが気になっています。
これがブレイクスルーになるかはわかりませんが、これからも追っていきたいです。

MLACについては他の記事も読ませていただいており、今回の記事執筆自体も自分自身勉強になりました。
MLACの著者の皆様、特に企画をしてくださった@naoya_tさん、ありがとうございました。

## 参考文献

[1] Pascal Vincent, Hugo Larochelle, Yoshua Bengio and Pierre-Antoine Manzagol. [Extracting and Composing Robust Features with Denoising Autoencoders](http://www.iro.umontreal.ca/~vincentp/Publications/vincent_icml_2008.pdf). *Proc. of ICML, 2008*.\\
[2] Salah Rifai, Pascal Vincent, Xavier Muller, Xavier Glorot and Yoshua Bengio. [Contractive Auto-Encoders: Explicit Invariance During Feature Extraction](http://www.icml-2011.org/papers/455_icmlpaper.pdf). *Proc. of ICML, 2011*.\\
[3] Pascal Vincent. [A Connection Between Score Matching and Denoising Autoencoders](http://www.iro.umontreal.ca/~vincentp/Publications/DenoisingScoreMatching_NeuralComp2011.pdf). *Neural Computation, 2011*.\\
[4] Aapo Hyv̈arinen. [Estimation of Non-Normalized Statistical Models by Score Matching](http://machinelearning.wustl.edu/mlpapers/paper_files/Hyvarinen05.pdf). *Journal of Machine Learning Research 6, 2005*.\\
[5] Yoshua Bengio, Li Yao, Guillaume Alain, and Pascal Vincent. [Generalized Denoising Auto-Encoders as Generative Models](http://papers.nips.cc/paper/5023-generalized-denoising-auto-encoders-as-generative-models). *NIPS, 2013*.\\
[6] Yoshua Bengio. [Deep Generative Stochastic Networks Trainable by Backprop](http://arxiv.org/abs/1306.1091). *arXiv:1306.1091, 2013*.\\
[7] Yoshua Bengio. [Estimating or Propagating Gradients Through Stochastic Neurons](http://arxiv.org/abs/1305.2982). *arXiv:1305.2982, 2013*.
