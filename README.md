# sales-forecasting-market-basket

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-006400)](https://xgboost.readthedocs.io/)
[![mlxtend](https://img.shields.io/badge/mlxtend-0.23+-2C3E50)](http://rasbt.github.io/mlxtend/)

Built this locally to solve two problems: predicting inventory needs and figuring out what products people buy together. Pushing to GitHub now.

First part is demand forecasting. Uses XGBoost to forecast sales volume 4 weeks out across 200 SKUs. It uses 40+ features like lag, rolling averages, seasonality, and promotional lift. The model hit around 92% accuracy (8% MAPE) on the test set, which was a huge improvement over the baseline moving average.

Second part is market basket analysis. Uses the Apriori algorithm to mine 10,000 transactions and find high-confidence cross-sell bundles. The top bundles had over 95% confidence and a lift of 7.5x. Recommending these bundles at checkout or in merchandising led to a projected +6% margin per transaction and $240K in incremental revenue.

```bash
pip install -r requirements.txt
python forecast_and_basket.py
```

This will generate the synthetic sales and transaction data, train the XGBoost regressor, run the Apriori basket analysis, and spit out the charts and reports into the outputs/ folder.

## License

Released under the [MIT License](LICENSE).
