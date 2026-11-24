# CNN-Energy-Forecasting

**Future Trends of Energy Consumption in Indonesia Through CNN-based Modeling**

Yustika, Lindiasari Martha · Ikhsan, Rifki Rahman Nur · Raharjo, Jangkung · Callista, Areta Adelia · Yl, Muhammad Albaihaq Abdiary.  
*IEEE IAICT 2025, 311–317*  
[![DOI](https://img.shields.io/badge/DOI-10.1109%2Fiaict65714.2025.11101693-blue)](https://doi.org/10.1109/iaict65714.2025.11101693)

Long-term electricity consumption forecasting for Indonesia with CNN (Conv1D), compared against GRU and ARIMA.

## Abstract

Accurate electricity load forecasting is essential for balancing supply and demand, optimizing energy distribution, and improving cost efficiency. This study presents a (CNN)based model for long-term electricity load forecasting in Indonesia, leveraging 1D Convolutional Layers (Conv1D) to extract patterns from historical consumption data (1989-2020) across multiple sectors. Using a sliding window approach, the model captures temporal dependencies and performs multi-step forecasting up to 30 years ahead. Evaluation against GRU and ARIMA shows CNN’s superior accuracy, achieving a MAPE of $4.30 \%$ and $\mathbf{R}^{\mathbf{2}}$ of $\mathbf{0. 8 7}$, outperforming GRU (MAPE 6.24%) and ARIMA (MAPE 8.60%). CNN’s ability to capture seasonal patterns and long-term trends without recurrent architectures makes it a promising tool for national energy planning. Its flexibility and scalability allow adaptation to various future scenarios, supporting sustainable electricity supply and informed policymaking.


## Proposed method

The forecaster is a one-dimensional CNN that maps a five-year window of historical electricity consumption to the next year's capacity. Predictions are produced recursively: each output is appended to the input window to forecast the following year, which yields the 2021–2050 projection.

### Problem formulation

Let $y_t$ be the annual series. The sliding-window pair is

$$\mathbf{x}_t=\left[y_{t-4},y_{t-3},y_{t-2},y_{t-1},y_t\right],\qquad \hat y_{t+1}=f_\theta(\mathbf{x}_t)$$

and the fitted parameters minimise the mean absolute percentage error

$$\theta^*=\arg\min_\theta\ \frac{1}{n}\sum_{t}\left|\frac{y_{t+1}-\hat y_{t+1}}{y_{t+1}}\right|\times100\%$$

### Network equations

Convolution layer, 32 filters, kernel size 2:

$$h_j^{(k)}=\operatorname{ReLU}\!\left(\sum_{m=0}^{1}w_m^{(k)}x_{j+m}+b^{(k)}\right),\qquad \operatorname{ReLU}(z)=\max(0,z)$$

Flatten, then a fully connected layer with 50 units and a linear output:

$$\mathbf{z}=\operatorname{ReLU}\!\left(\mathbf{W}_1\operatorname{vec}(\mathbf{h})+\mathbf{b}_1\right),\qquad \hat y=\mathbf{w}_2^{\top}\mathbf{z}+b_2$$

Training uses Adam with the MAPE objective, 100 epochs, batch size 8, and a chronological split (training before 2011, validation 2011–2020).

### Baselines

GRU with update and reset gates:

$$z_t=\sigma(\mathbf{W}_z\mathbf{x}_t+\mathbf{U}_z\mathbf{h}_{t-1}),\qquad r_t=\sigma(\mathbf{W}_r\mathbf{x}_t+\mathbf{U}_r\mathbf{h}_{t-1})$$

$$\tilde{\mathbf{h}}_t=\tanh\!\left(\mathbf{W}_h\mathbf{x}_t+\mathbf{U}_h(r_t\odot\mathbf{h}_{t-1})\right),\qquad \mathbf{h}_t=(1-z_t)\odot\mathbf{h}_{t-1}+z_t\odot\tilde{\mathbf{h}}_t$$

ARIMA$(p,d,q)$:

$$\phi(L)(1-L)^d y_t=\theta(L)\varepsilon_t,\qquad \varepsilon_t\sim\mathcal{N}(0,\sigma^2)$$

### Accuracy measures

$$\mathrm{MAPE}=\frac{1}{n}\sum_{i=1}^{n}\left|\frac{A_i-P_i}{A_i}\right|\times100\%,\qquad \mathrm{RMSE}=\sqrt{\frac{1}{n}\sum_{i=1}^{n}(A_i-P_i)^2}$$

$$R^2=1-\frac{\sum_i(A_i-P_i)^2}{\sum_i(A_i-\bar A)^2}$$

Validation results: CNN MAPE 4.30%, $R^2$ 0.87; GRU 6.24%, 0.77; ARIMA 8.60%, 0.52.

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
