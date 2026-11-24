# CNN-Energy-Forecasting

**Future Trends of Energy Consumption in Indonesia Through CNN-based Modeling**

Yustika, Lindiasari Martha · Ikhsan, Rifki Rahman Nur · Raharjo, Jangkung · Callista, Areta Adelia · Yl, Muhammad Albaihaq Abdiary.  
*IEEE IAICT 2025, 311–317*  
[![DOI](https://img.shields.io/badge/DOI-10.1109%2Fiaict65714.2025.11101693-blue)](https://doi.org/10.1109/iaict65714.2025.11101693)

Long-term electricity consumption forecasting for Indonesia with CNN (Conv1D), compared against GRU and ARIMA.

## Abstract

Accurate electricity load forecasting is essential for balancing supply and demand, optimizing energy distribution, and improving cost efficiency. This study presents a (CNN)based model for long-term electricity load forecasting in Indonesia, leveraging 1D Convolutional Layers (Conv1D) to extract patterns from historical consumption data (1989-2020) across multiple sectors. Using a sliding window approach, the model captures temporal dependencies and performs multi-step forecasting up to 30 years ahead. Evaluation against GRU and ARIMA shows CNN’s superior accuracy, achieving a MAPE of $4.30 \%$ and $\mathbf{R}^{\mathbf{2}}$ of $\mathbf{0. 8 7}$, outperforming GRU (MAPE 6.24%) and ARIMA (MAPE 8.60%). CNN’s ability to capture seasonal patterns and long-term trends without recurrent architectures makes it a promising tool for national energy planning. Its flexibility and scalability allow adaptation to various future scenarios, supporting sustainable electricity supply and informed policymaking.

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
