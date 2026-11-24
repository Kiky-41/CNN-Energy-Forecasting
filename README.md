# CNN-Energy-Forecasting

**Future Trends of Energy Consumption in Indonesia Through CNN-based Modeling**

Yustika, Lindiasari Martha · Ikhsan, Rifki Rahman Nur · Raharjo, Jangkung · Callista, Areta Adelia · Yl, Muhammad Albaihaq Abdiary.  
*IEEE IAICT 2025, 311–317*  
[![DOI](https://img.shields.io/badge/DOI-10.1109%2Fiaict65714.2025.11101693-blue)](https://doi.org/10.1109/iaict65714.2025.11101693)

Long-term electricity consumption forecasting for Indonesia with CNN (Conv1D), compared against GRU and ARIMA.

## Abstract

Accurate electricity load forecasting is essential for balancing supply and demand, optimizing energy distribution, and improving cost efficiency. This study presents a (CNN)based model for long-term electricity load forecasting in Indonesia, leveraging 1D Convolutional Layers (Conv1D) to extract patterns from historical consumption data (1989-2020) across multiple sectors. Using a sliding window approach, the model captures temporal dependencies and performs multi-step forecasting up to 30 years ahead. Evaluation against GRU and ARIMA shows CNN’s superior accuracy, achieving a MAPE of $4.30 \text{ percent}$ and $\mathbf{R}^{\mathbf{2}}$ of $\mathbf{0. 8 7}$, outperforming GRU (MAPE 6.24%) and ARIMA (MAPE 8.60%). CNN’s ability to capture seasonal patterns and long-term trends without recurrent architectures makes it a promising tool for national energy planning. Its flexibility and scalability allow adaptation to various future scenarios, supporting sustainable electricity supply and informed policymaking.



## Proposed method

The paper proposes a one-dimensional convolutional network (CNN) for long-term electricity-capacity forecasting. A five-year window of historical consumption is mapped to the next year, and the trained model is then used recursively: each prediction is appended to the window to forecast the following year, producing the 2021–2050 projection. The paper compares the CNN with a GRU and with ARIMA on the same 1989–2020 series and reports a MAPE of 4.30% for the CNN against 6.24% for the GRU and 8.60% for ARIMA.

### Problem formulation

Let $y_t$ be the annual capacity series. The supervised pair built by the sliding window is

$$\mathbf{x}_t=\left[y_{t-4},y_{t-3},y_{t-2},y_{t-1},y_t\right],\qquad \hat y_{t+1}=f_\theta(\mathbf{x}_t),$$

and the network parameters minimise the mean absolute percentage error

$$\theta^{\ast}=\arg\min_\theta~\frac{1}{n}\sum_{t}\left|\frac{y_{t+1}-\hat y_{t+1}}{y_{t+1}}\right|\times100\text{ percent}.$$

### Notation

| Symbol | Meaning | Unit |
|---|---|---|
| $y_t$ | observed capacity in year $t$ | MW |
| $\hat y_t$ | predicted capacity | MW |
| $\mathbf{x}_t$ | five-year input window | MW |
| $f_\theta$ | network with parameters $\theta$ | – |
| $w_m^{(k)},b^{(k)}$ | filter weights and bias of filter $k$ | – |
| $h_j^{(k)}$ | activation of filter $k$ at position $j$ | – |
| $\mathbf{W}_1,\mathbf{b}_1$ | dense-layer weights and bias | – |
| $\mathbf{w}_2,b_2$ | output-layer weights and bias | – |
| $A_i,P_i$ | actual and predicted value in the error metrics | MW |
| $n$ | number of evaluated years | – |

### Network equations

The convolution layer uses 32 filters of width 2. Each filter slides over the input window and applies a rectified linear unit:

$$h_j^{(k)}=\mathrm{ReLU}\left(\sum_{m=0}^{1}w_m^{(k)}x_{j+m}+b^{(k)}\right),\qquad \mathrm{ReLU}(z)=\max(0,z).$$

The filter learns a short local pattern, for example a two-year rise or fall, and the 32 filters together cover different trend shapes. The output is flattened and passed to a fully connected layer with 50 units, followed by a single linear output:

$$\mathbf{z}=\mathrm{ReLU}\left(\mathbf{W}_1\mathrm{vec}(\mathbf{h})+\mathbf{b}_1\right),\qquad \hat y=\mathbf{w}_2^{\top}\mathbf{z}+b_2.$$

The linear output is required because the target is a continuous capacity value, not a class. Training uses Adam with the MAPE objective, 100 epochs, batch size 8, and a chronological split (training before 2011, validation 2011–2020). Inputs are min–max scaled to $[0,1]$ and predictions are rescaled before scoring.

### Baseline models

The GRU baseline controls information with an update gate $z_t$ and a reset gate $r_t$:

$$z_t=\sigma(\mathbf{W}_z\mathbf{x}_t+\mathbf{U}_z\mathbf{h}_{t-1}),\qquad r_t=\sigma(\mathbf{W}_r\mathbf{x}_t+\mathbf{U}_r\mathbf{h}_{t-1}),$$

$$\tilde{\mathbf{h}}_t=\tanh\left(\mathbf{W}_h\mathbf{x}_t+\mathbf{U}_h(r_t\odot\mathbf{h}_{t-1})\right),\qquad \mathbf{h}_t=(1-z_t)\odot\mathbf{h}_{t-1}+z_t\odot\tilde{\mathbf{h}}_t.$$

The update gate decides how much old memory survives, and the reset gate decides how much of the past is used when forming the candidate state. ARIMA$(p,d,q)$ is the statistical baseline:

$$\phi(L)(1-L)^d y_t=\theta(L)\varepsilon_t,\qquad \varepsilon_t\sim\mathcal{N}(0,\sigma^2),$$

where $\phi(L)$ and $\theta(L)$ are autoregressive and moving-average polynomials and $d$ is the differencing order.

### Accuracy measures

$$\mathrm{MAPE}=\frac{1}{n}\sum_{i=1}^{n}\left|\frac{A_i-P_i}{A_i}\right|\times100\text{ percent},\qquad \mathrm{RMSE}=\sqrt{\frac{1}{n}\sum_{i=1}^{n}(A_i-P_i)^2},$$

$$R^2=1-\frac{\sum_i(A_i-P_i)^2}{\sum_i(A_i-\bar A)^2}.$$

MAPE expresses the average error as a percentage of the actual value, RMSE penalises large deviations, and $R^2$ measures the share of variance explained.

### Implementation note

The notebook stored in this repository, `GRU_and_comparison.ipynb`, implements the GRU baseline (50 hidden units, Dense(1), Adam, MAPE loss, 100 epochs, batch 8). The CNN architecture described above is the model reported in the paper; the CNN training script itself is not included. The reported CNN metrics are therefore paper results, not outputs of this notebook.
### Reported result
Validation over 2011-2020: CNN MAPE 4.30% with $R^2=0.87$, GRU MAPE 6.24% with $R^2=0.77$, ARIMA MAPE 8.60% with $R^2=0.52$. The CNN projection for 2021-2050 is the 30-step recursive forecast.

## Contents

```
.gitignore
GRU_and_comparison.ipynb
results/
  GRU.xlsx
```

## Data

Input data is **not included**. Place files in `data/` (notebooks) or next to the `.m` script (MATLAB):

| File | Columns |
|---|---|
| `Data.xlsx` | Sheet1: `Year`, `Household`, `Industry`, `CP`, `TotalKap` |


## Results

### Paper metrics on 2011-2020 validation split

| Model | MAPE | MAE (MW) | RMSE (MW) | R2 |
|---|---|---|---|---|
| CNN (Conv1D) | 4.30% | 8,317.34 | 10,118.88 | 0.87 |
| GRU | 6.24% | 12,260.50 | 13,532.66 | 0.77 |
| ARIMA | 8.60% | 18,351.00 | 19,375.15 | 0.52 |

The CNN leads on all four measures: MAPE 4.30%, MAE 8,317.34 MW, RMSE 10,118.88 MW and R2 0.87, against 6.24% / 12,260.50 / 13,532.66 / 0.77 for the GRU and 8.60% / 18,351.00 / 19,375.15 / 0.52 for ARIMA. The convolutional layer reads short capacity sequences directly, so the forecast does not depend on a recurrent state carried across the whole series.

### Paired t-tests and one-way ANOVA on absolute errors

| Test | Statistic | p | Significant |
|---|---|---|---|
| Paired t-test, CNN vs GRU | -2.81 | 0.020 | yes |
| Paired t-test, CNN vs ARIMA | -3.05 | 0.014 | yes |
| Paired t-test, GRU vs ARIMA | -1.99 | 0.077 | no |
| One-way ANOVA, three models | F 6.59 | 0.005 | yes |

CNN errors are lower than GRU (t -2.81, p 0.020) and lower than ARIMA (t -3.05, p 0.014), while GRU and ARIMA cannot be separated (t -1.99, p 0.077). The one-way ANOVA over the three models gives F 6.59 with p 0.005, so at least one model differs and the paired tests locate it at the CNN.

### Validation, 2011-2020

| Year | 2011 | 2012 | 2013 | 2014 | 2015 | 2016 | 2017 | 2018 | 2019 | 2020 |
|---|---|---|---|---|---|---|---|---|---|---|
| Actual (MW) | 157,992.66 | 173,990.75 | 187,541.02 | 198,601.78 | 202,845.82 | 216,004.32 | 221,574.66 | 234,617.88 | 245,518.20 | 243,582.73 |
| Predicted (MW) | 151,223.48 | 161,794.08 | 174,424.97 | 188,872.86 | 204,024.33 | 216,915.81 | 229,069.31 | 239,754.41 | 250,503.02 | 262,051.06 |
| Error (%) | 4.28 | 7.01 | 6.99 | 4.90 | 0.58 | 0.42 | 3.38 | 2.19 | 2.03 | 7.58 |

Yearly error moves between 0.42% (2016) and 7.58% (2020) with a mean of 3.94%. The model overshoots from 2017 to 2019 and undershoots 2020, the year capacity stops following the pre-2020 trend, which is the single largest miss in the record.

### Projection, selected years

| Year | 2021 | 2025 | 2030 | 2035 | 2040 | 2045 | 2050 |
|---|---|---|---|---|---|---|---|
| Predicted capacity (MW) | 271,635.78 | 332,818.19 | 432,311.41 | 536,407.31 | 618,076.56 | 664,986.12 | 686,532.88 |

The fitted trend gives 271,636 MW in 2021, 432,311 MW in 2030 and 686,533 MW in 2050, about 2.82 times the 2020 installed capacity. Growth is steepest before 2035 and flattens afterwards.

### Result workbooks

| File | Content |
|---|---|
| [`results/GRU.xlsx`](results/GRU.xlsx) | raw output |

The workbooks hold the raw solver output behind the tables above: per-unit schedules, per-hour costs, constraint checks and run statistics.


## Requirements

Python (TensorFlow). Python notebooks: `numpy`, `pandas`, `scipy`, `matplotlib` (as imported). MATLAB scripts: R2021a+.

## Citation

```bibtex
@inproceedings{Yustika_2025, title={Future Trends of Energy Consumption in Indonesia Through CNN-based Modeling}, url={http://dx.doi.org/10.1109/iaict65714.2025.11101693}, DOI={10.1109/iaict65714.2025.11101693}, booktitle={2025 IEEE International Conference on Industry 4.0, Artificial Intelligence, and Communications Technology (IAICT)}, publisher={IEEE}, author={Yustika, Lindiasari Martha and Ikhsan, Rifki Rahman Nur and Raharjo, Jangkung and Callista, Areta Adelia and Yl, Muhammad Albaihaq Abdiary.}, year={2025}, month=July, pages={311–317} }
```

## License

[![CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](LICENSE)

Licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). **Attribution is required**: any use of this code, results, or figures, including in publications, must cite:

> Lindiasari Martha Yustika; Rifki Rahman Nur Ikhsan; Jangkung Raharjo; Areta Adelia Callista; Muhammad Albaihaq Abdiary YL (2025). Future Trends of Energy Consumption in Indonesia Through CNN-based Modeling. *2025 IEEE International Conference on Industry 4.0, Artificial Intelligence, and Communications Technology (IAICT)*. https://doi.org/10.1109/iaict65714.2025.11101693
