---
layout: post
title: "バイアス・バリアンス分解"
date: 2013-12-28 16:37
comments: true
categories: 機械学習
published: false
---

機械学習の文献を読んでいると学習手法のバイアス・バリアンスという話が良く出てきます。
複雑さの大きい手法は低バイアスだが高バリアンスで過学習しやすい、とか言ったりします。
教科書でよく触れられるのは二乗誤差を用いた回帰問題におけるバイアス・バリアンス分解ですが、分類問題ではどうなるのか知らなかったので調べました。

<!-- more -->

## 回帰におけるバイアス・バリアンス分解

まずは回帰における分解を思い出してみます。
訓練データセット $D=((x_i, t_i))_{i=1}^n$ はデータ生成分布 $\mathcal P$ に従うi.i.d.点列とします。
学習手法 $f$ は訓練データ $D$ を学習したときに予測器 $f_D$ を出力します。
正解データ $t$ のときに $y$ を予測した場合のエラーは二乗誤差 $L(t, y)=(t-y)^2$ で与えられるとします。

考えたいのは $f$ が解きたい問題をどれくらい正しく学習してくれるかです。
ここでは新たなサンプル $x$ と対応する $t$ を使って評価します。
ただし $t$ にはラベルノイズ [^1] が乗っていると仮定し、$t$ に関する期待誤差で評価することにします。
データセット $D$ を学習した予測器 $f_D$ のサンプル $x$ における評価は $\mathbb E\_{t\vert x}L(t, f_D(x))$ で与えられます。
ここで $\mathbb E$ は添字の確率変数に関する期待値を表すとします（ここでは条件付き分布 $p(t\vert x)$ について期待値を取ります）。

[^1]: この用語は主に分類問題で用いられるものですが、ここでは回帰を含む一般の予測問題においても同じ用語を使うことにしました。

最適な予測は $y^\star(x)=\text{arg}\min_{y}\mathbb E\_{t\vert x}L(t, y)$ で与えられます。
二乗誤差を用いる場合、最適な予測は $t$ の期待値 $y^\star(x)=\mathbb E\_{t\vert x}t$ となります。

$\mathbb E\_{t\vert x}L(t, f_D(x))$ は予測誤差の評価としては適切ですが、評価対象は具体的な予測器 $f_D$ であって、学習手法 $f$ 自体の評価とは言えません。
後者を考えたい場合、データセット $D$ を確率変数だと思って、これを積分消去します。
つまり評価指標としては $\mathbb E\_{D, t\vert x}L(t, f_D(x))=\mathbb E\_{D, t\vert x}(t-f_D(x))^2$ が適切です。

同じように予測器に対してもデータセットの積分消去を考えることができます。
これはデータセット全体に渡る「予測器の期待値」$f\_{\mathcal P}$を与え、手法 $f$ のデータセットに依らない出力となります。
式で書くと $f\_{\mathcal P}(x)=\mathbb E_D f_D(x)$ です。
ここではこれを期待予測と呼ぶことにします。

期待予測と最適予測を使って上記の評価指標をキレイに分解することができます。

$$
\begin{align*}
\mathbb E_{D, t\vert x}(t-f_D(x))^2&=\mathbb E_{D, t\vert x}((t-f_{\mathcal P}(x))+(f_{\mathcal P}(x)-f_D(x)))^2\\
&=\mathbb E_{t\vert x}(t-f_{\mathcal P}(x))^2+\mathbb E_D(f_{\mathcal P}(x)-f_D(x))^2+2(t-f_{\mathcal P}(x)\mathbb E_D(f_{\mathcal P}(x)-f_D(x))\\
&=\mathbb E_{t\vert x}(t-f_{\mathcal P}(x))^2+\mathbb E_D(f_{\mathcal P}(x)-f_D(x))^2\\
&=\mathbb E_{t\vert x}((t-y^\star)+(y^\star-f_{\mathcal P}(x))^2+\mathbb E_D(f_{\mathcal P}(x)-f_D(x))^2\\
&=\mathbb E_{t\vert x}(t-y^\star)^2+(y^\star-f_{\mathcal P}(x))^2-2(y^\star-f_{\mathcal P}(x))\mathbb E_{t\vert x}(t-y^\star)+\mathbb E_D(f_{\mathcal P}(x)-f_D(x))^2\\
&=\mathbb E_{t\vert x}(t-y^\star)^2+(y^\star-f_{\mathcal P}(x))^2+\mathbb E_D(f_{\mathcal P}(x)-f_D(x))^2\\
&=\mathbb E_{t\vert x}L(t, y^\star)+L(y^\star, f_{\mathcal P}(x))+\mathbb E_D L(f_{\mathcal P}(x), f_D(x)).
\end{align*}
$$

期待二乗誤差をある確率変数について分解するテクニックを2回使っています。
これらのテクニックが使えるのは $y^\star$ と $f\_{\mathcal P}$ の定義のおかげで、これらの定義が本質的であることを表しています。

分解された最右辺の各項を左から順に $N(x), B(x), V(x)$ とおきます。
これらはそれぞれ**ノイズ**、**（二乗）バイアス**、**バリアンス**と呼ばれています。
3項に分解されたこの形はバイアス・バリアンス・ノイズ分解 (BVND) と呼ばれます。
ノイズ項は学習手法 $f$ によらない誤差の下限で、ラベルノイズの影響を表しています。
ラベルノイズが全くない場合には $N(x)=L(y^\star, y^\star)=(y^\star-y^\star)^2=0$ となり、教科書でよく見るバイアス・バリアンス分解 (BVD) の式になります。

ここで述べた二乗誤差に対するバイアス・バリアンス分解の定式化は1992年に [1] において導入されたようです。
バイアス・バリアンスという概念とそれらの間のトレードオフ自体はそれ以前から経験則としてよく知られていたようです。
[1] からしばらくの間、分類問題への様々な拡張が提案されました。
今日はその中でも2000年にPedro Domingosが提案した拡張を紹介します。

## バイアス・バリアンス分解の一般化

Pedro Domingosが提案した拡張 [2] は分類問題だけでなく一般の誤差関数 $L$ に対する拡張です。
分類問題に適用する場合は0-1誤差 $L(t, y)=1_{t=y}$ を用います。
ここで $1_P$ は命題 $P$ が真の場合に1、そうでない場合に0とします。

記号のおさらいをします。
訓練データセット $D=((x_i, t_i))_{i=1}^n$ はデータ生成分布 $\mathcal P$ に従うi.i.d.点列とします。
学習手法 $f$ はデータセット $D$ を入力すると予測器 $f_D$ を出力します。
新たなサンプル $x$ とラベルノイズの乗った出力 $t$ を用いて $f$ を評価します。
具体的な予測器 $f_D$ の評価は $\mathbb E\_{t\vert x}L(t, f_D(x))$ で与えられます。
これをデータセット $D$ について周辺化することで学習手法 $f$ の評価 $\mathbb E\_{D, t\vert x}L(t, f_D(x))$ が得られます。
最適な予測は $y^\star=\text{arg}\min_y\mathbb E\_{t\vert x}L(t, y)$ です。

回帰（二乗誤差）における期待予測 $f\_{\mathcal P}(x)=\mathbb E_D f_D(x)$ を一般の誤差関数でそのまま用いることはできません。
たとえば0-1誤差の場合、予測は必ず0か1でなくてはなりませんが、一般に期待予測 $f\_{\mathcal P}(x)$ が0, 1の二値を取る保証はありません。
そこで [2] では「すべての $f_D(x)$ から平均的に最も近い予測」を代わりに使っています。
式で示すと $f\_{\mathcal P}^{\text m}(x)=\text{arg}\min_y\mathbb E_D L(f_D(x), y)$ です。
[2] ではこれを**主予測(main prediction)**と呼んでいます。
$L$ が二乗誤差の場合、主予測 $f\_{\mathcal P}^{\text m}(x)$ は期待予測 $f\_{\mathcal P}(x)$ と一致します。
つまり主予測は期待予測の一般化になっています。

以上のもとで二乗誤差の場合と全く同じようにノイズ、バイアス、バリアンス項を定義します。

$$
\begin{align*}
N(x)&=\mathbb E_{t\vert x}L(t, y^\star(x)),\\
B(x)&=L(y^\star(x), f_{\mathcal P}^{\text m}(x)),\\
V(x)&=\mathbb E_D L(f_{\mathcal P}^{\text m}(x), f_D(x)).\\
\end{align*}
$$

このとき $L$ が「適切な」誤差関数であれば以下の分解ができます。

$$\mathbb E_{D, t\vert x}L(t, f_D(x))=c_1(x)N(x)+B(x)+c_2(x)V(x).$$

ここで $c_1, c_2$ は$x$ と誤差関数によって定まる定数です。

## 参考文献

[1] Stuart Geman, Elin Bienenstock and Rene Doursat. [Neural networks and the bias/variance dilemma](http://www.dna.caltech.edu/courses/cns187/references/geman_etal.pdf). *Neural Computation*, *4*, 1-58, 1992.\\
[2] Pedro Domingos. [A Unified Bias-Variance Decomposition and its Applications](http://homes.cs.washington.edu/~pedrod/papers/mlc00a.pdf). *Proc. 17th ICML*, 2000.\\
[3] Pedro Domingos. [A unified bias-variance decomposition](http://homes.cs.washington.edu/~pedrod/bvd.pdf). *Tech. Report*, 2000.
