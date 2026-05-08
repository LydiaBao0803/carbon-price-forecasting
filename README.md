# EU Carbon Allowance (EUA) Price Forecasting Using VMD Decomposition and Deep Learning

## Overview
This project forecasts EU carbon allowance (EUA) futures prices using a hybrid signal decomposition and machine learning pipeline. Variational Mode Decomposition (VMD) is applied to decompose EUA and 11 related market factors (equity indices, commodities, volatility) into frequency-based modes; sample entropy filters out noise components. XGBoost with RFECV selects optimal features, and five forecasting models (LSTM with attention, GRU, SVM, KNN, BPNN) are evaluated at 1-day, 15-day, and 30-day forecast horizons.

## Key Skills Demonstrated
- Signal processing: Variational Mode Decomposition (VMD) and sample entropy for noise filtering
- Feature engineering with frequency decomposition (VMD-based frequency factors)
- XGBoost with recursive feature elimination and cross-validation (RFECV)
- Deep learning time-series models: LSTM with attention mechanism and GRU
- Traditional ML forecasting: SVM (RBF kernel), KNN, BPNN (MLP)
- Sliding window architecture for multi-step horizon forecasting
- Multi-model benchmark comparison (RMSE, MAE, MAPE)
- Interval forecasting with B-spline regression

## Methods & Tools
- **Language**: Python
- **Libraries**: `vmdpy`, `xgboost`, `scikit-learn`, `TensorFlow/Keras`, `pandas`, `NumPy`, `matplotlib`, `scipy`
- **Techniques**: VMD (K=11 modes), sample entropy, MinMaxScaler, sliding window (window=5), RFECV with 4-fold CV, LSTM+Attention, GRU, SVR, KNeighborsRegressor, MLPRegressor, EarlyStopping

## Dataset
EU carbon allowance (EUA) daily price series plus 11 market covariates: CAC40, CRB commodity index, Dow Jones (DJI), oil prices, natural gas prices, EUA volatility (EUA_Vol.), EUR/USD exchange rate, FTSE100, DAX30, NASDAQ100 (NDX100), and realized volatility (EUA_RV). Data stored in `data.csv` / `data_clean.csv`; VMD-decomposed features in `data_VMD.csv` and `data_VMD_freq.csv`.

## Results
- VMD-based frequency features outperformed raw original factors across most forecast horizons
- LSTM with attention mechanism achieved competitive performance at short horizons (1-day)
- Systematic benchmark: RMSE, MAE, MAPE reported for all 5 models × 2 feature strategies × 3 horizons (30 total configurations)
- Final results saved to `point_forcast/point_forcast_final_results.csv`

## File Structure
```
数据及代码/
├── MVMD_reconstruction.py        # VMD decomposition + sample entropy filtering
├── XGB_RFE.py                    # XGBoost feature selection with RFECV
├── point_forecast.py             # Main forecasting: LSTM/GRU/SVM/KNN/BPNN comparison
├── interval_forcast.py           # Interval forecasting
├── interval_forcast bspline.py   # B-spline interval forecasting
├── setupAndEDA.ipynb             # Data setup and exploratory data analysis
├── point_plot.ipynb              # Visualization of forecasting results
├── data.csv                      # Raw data
├── data_clean.csv                # Cleaned data with selected features
├── data_VMD.csv                  # VMD-decomposed signals
├── data_VMD_freq.csv             # Frequency-domain VMD features
├── Decomposed/                   # Decomposed mode files per feature
├── point_forcast/                # Model prediction outputs (CSV per model)
└── interval_forcast/             # Interval forecast outputs
```

## How to Run
1. Install dependencies: `pip install vmdpy xgboost scikit-learn tensorflow pandas numpy matplotlib scipy`
2. Run VMD decomposition: `python MVMD_reconstruction.py`
3. Run feature selection: `python XGB_RFE.py`
4. Run model comparison: `python point_forecast.py`
5. View results in `point_forcast/point_forcast_final_results.csv`

---

# 基于 VMD 分解与深度学习的欧盟碳排放配额（EUA）价格预测

## 项目简介
本项目构建了一套混合信号分解与机器学习预测框架，用于预测欧盟碳排放配额（EUA）期货价格。采用变分模态分解（VMD）对 EUA 及 11 个相关市场因子进行频率分解，样本熵筛除噪声模态；XGBoost 结合 RFECV 进行特征选择；最终对比五种预测模型（含注意力机制的 LSTM、GRU、SVM、KNN、BPNN）在 1 天、15 天、30 天三个预测期上的表现。

## 使用技术
- **语言**：Python
- **主要库**：`vmdpy`、`xgboost`、`scikit-learn`、`TensorFlow/Keras`、`pandas`、`NumPy`、`matplotlib`、`scipy`
- **核心方法**：VMD（K=11 模态）、样本熵、滑动窗口、RFECV 特征选择（4 折交叉验证）、LSTM+注意力机制、GRU、SVR、KNN、MLP、EarlyStopping

## 数据集
EUA 日度价格序列及 11 个市场协变量：CAC40、CRB 商品指数、道琼斯指数（DJI）、原油价格、天然气价格、EUA 波动率、EUR/USD 汇率、富时 100（FTSE100）、DAX30、纳斯达克 100（NDX100）和已实现波动率（EUA_RV）。

## 主要结果
- 基于 VMD 频率分解特征的预测效果优于原始特征（多数模型和预测期上）
- 含注意力机制的 LSTM 在短期（1 天）预测中表现出竞争力
- 完整对比了 30 种配置（5 模型 × 2 特征策略 × 3 预测期）的 RMSE、MAE、MAPE
- 汇总结果存储于 `point_forcast/point_forcast_final_results.csv`

## 文件说明
- `MVMD_reconstruction.py`：VMD 分解与样本熵滤波
- `XGB_RFE.py`：基于 RFECV 的 XGBoost 特征选择
- `point_forecast.py`：多模型预测对比主程序
- `setupAndEDA.ipynb`：数据预处理与探索性分析
- `point_plot.ipynb`：预测结果可视化
- `data_VMD_freq.csv`：VMD 频域特征数据

## 运行方法
1. 安装依赖：`pip install vmdpy xgboost scikit-learn tensorflow pandas numpy scipy`
2. VMD 分解：`python MVMD_reconstruction.py`
3. 特征选择：`python XGB_RFE.py`
4. 模型对比：`python point_forecast.py`
