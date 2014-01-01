---
layout: post
title: "バイアス・バリアンス分解"
date: 2013-12-28 16:37
comments: true
categories: 機械学習
---

機械学習の文献を読んでいると学習手法のバイアス・バリアンスという話が良く出てきます。
複雑さの大きい手法は低バイアスだが高バリアンスで過学習しやすい、とか言ったりします。
教科書でよく触れられるのは二乗誤差を用いた回帰問題におけるバイアス・バリアンス分解ですが、分類問題ではどうなるのか知らなかったので調べました。

<!-- more -->

## 回帰におけるバイアス・バリアンス分解

まずは回帰における分解を思い出してみます。
訓練データセット $D=((x_i, t_i))_{i=1}^n$ はデータ生成分布 $\mathcal P$ に従うi.i.d.データ列とします。
学習手法 $f$ は訓練データ $D$ を学習したときに予測器 $f_D$ を出力します。
正解データ $t$ に対して $y$ を予測したときのエラーは二乗誤差 $L(t, y)=(t-y)^2$ で与えられるとします。

今、新たなサンプル $x$ が与えられたとします。
予測器の出力を評価するために対応する出力 $t$ を使いますが、$t$ にはラベルノイズが乗っていると仮定します。
このときの最適な予測を $y^\star=\text{arg}\min_{y}\mathbb E_t L(t, y)$ とします。
二乗誤差を用いる場合、最適な予測は $t$ の期待値 $y^\star=\mathbb E_t t$ となります。

ここで考えたいのは $f$ が解きたい問題をどれくらい正しく学習してくれるかです。
データセット $D$ を学習した予測器 $f_D$ のサンプル $x$ における評価は $\mathbb E_t L(t, f_D(x))$ で与えられます。
ここで $\mathbb E$ は添字の確率変数に関する期待値を表すとします。

$\mathbb E_t L(t, f_D(x))$ は予測誤差の評価としては適切ですが、評価対象は具体的な予測器 $f_D$ であって、学習手法 $f$ 自体の評価とは言えません。
後者を考えたい場合、データセット $D$ を確率変数だと思って、これを積分消去します。
つまり評価指標としては $\mathbb E\_{D, t}L(t, f_D(x))=\mathbb E\_{D, t}(t-f_D(x))^2$ が適切です。

同じように予測器に対してもデータセットの積分消去を考えることができます。
これはデータセット全体に渡る「予測器の期待値」$f\_{\mathcal P}$を与え、手法 $f$ のデータセットに依らない出力となります。
式で書くと $f\_{\mathcal P}(x)=\mathbb E_D f_D(x)$ です。
ここではこれを期待予測と呼ぶことにします。

期待予測と最適予測を使って上記の評価指標をキレイに分解することができます。

$$
\begin{align*}
\mathbb E_{D, t}(t-f_D(x))^2&=\mathbb E_{D, t}((t-f_{\mathcal P}(x))+(f_{\mathcal P}(x)-f_D(x)))^2\\
&=\mathbb E_t(t-f_{\mathcal P}(x))^2+\mathbb E_D(f_{\mathcal P}(x)-f_D(x))^2+2(t-f_{\mathcal P}(x)\mathbb E_D(f_{\mathcal P}(x)-f_D(x))\\
&=\mathbb E_t(t-f_{\mathcal P}(x))^2+\mathbb E_D(f_{\mathcal P}(x)-f_D(x))^2\\
&=\mathbb E_t((t-y^\star)+(y^\star-f_{\mathcal P}(x))^2+\mathbb E_D(f_{\mathcal P}(x)-f_D(x))^2\\
&=\mathbb E_t(t-y^\star)^2+(y^\star-f_{\mathcal P}(x))^2-2(y^\star-f_{\mathcal P}(x))\mathbb E_t(t_y^\star)+\mathbb E_D(f_{\mathcal P}(x)-f_D(x))^2\\
&=\mathbb E_t(t-y^\star)^2+(y^\star-f_{\mathcal P}(x))^2+\mathbb E_D(f_{\mathcal P}(x)-f_D(x))^2\\
&=\mathbb E_t L(t, y^\star)+L(y^\star, f_{\mathcal P}(x))+\mathbb E_D L(f_{\mathcal P}(x), f_D(x)).
\end{align*}
$$

最右辺の各項を左から順に $N(x), B(x), V(x)$ と書きます。
これらはそれぞれ**ノイズ**、**（二乗）バイアス**、**バリアンス**と呼ばれています。

## 参考文献

[1] Pedro Domingos. A Unified Bias-Variance Decomposition and its Applications. *Proc. 17th ICML, 2000*.
