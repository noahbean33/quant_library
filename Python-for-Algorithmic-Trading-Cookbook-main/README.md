<p align="center"><a href="https://packt.link/mlsumgh"><img src="https://static.packt-cdn.com/assets/images/ML Summit Banner v3 1200x627.png" alt="Machine Learning Summit 2025"/></a></p>

## Machine Learning Summit 2025
**Bridging Theory and Practice: ML Solutions for Today’s Challenges**

3 days, 20+ experts, and 25+ tech sessions and talks covering critical aspects of:
- **Agentic and Generative AI**
- **Applied Machine Learning in the Real World**
- **ML Engineering and Optimization**

👉 [Book your ticket now >>](https://packt.link/mlsumgh)

---

## Join Our Newsletters 📬

### DataPro  
*The future of AI is unfolding. Don’t fall behind.*

<p><a href="https://landing.packtpub.com/subscribe-datapronewsletter/?link_from_packtlink=yes"><img src="https://static.packt-cdn.com/assets/images/DataPro NL QR Code.png" alt="DataPro QR" width="150"/></a></p>

Stay ahead with [**DataPro**](https://landing.packtpub.com/subscribe-datapronewsletter/?link_from_packtlink=yes), the free weekly newsletter for data scientists, AI/ML researchers, and data engineers.  
From trending tools like **PyTorch**, **scikit-learn**, **XGBoost**, and **BentoML** to hands-on insights on **database optimization** and real-world **ML workflows**, you’ll get what matters, fast.

> Stay sharp with [DataPro](https://landing.packtpub.com/subscribe-datapronewsletter/?link_from_packtlink=yes). Join **115K+ data professionals** who never miss a beat.

---

### BIPro  
*Business runs on data. Make sure yours tells the right story.*

<p><a href="https://landing.packtpub.com/subscribe-bipro-newsletter/?link_from_packtlink=yes"><img src="https://static.packt-cdn.com/assets/images/BIPro NL QR Code.png" alt="BIPro QR" width="150"/></a></p>

[**BIPro**](https://landing.packtpub.com/subscribe-bipro-newsletter/?link_from_packtlink=yes) is your free weekly newsletter for BI professionals, analysts, and data leaders.  
Get practical tips on **dashboarding**, **data visualization**, and **analytics strategy** with tools like **Power BI**, **Tableau**, **Looker**, **SQL**, and **dbt**.

> Get smarter with [BIPro](https://landing.packtpub.com/subscribe-bipro-newsletter/?link_from_packtlink=yes). Trusted by **35K+ BI professionals**, see what you’re missing.

# Python for Algorithmic Trading Cookbook

<a href="<https://www.packtpub.com/en-in/product/python-for-algorithmic-trading-cookbook-9781835084700"><img src="https://content.packt.com/_/image/xxlarge/B21323/cover_image_large.jpg" alt="Python for Algorithmic Trading Cookbook" height="256px" align="right"></a>

This is the code repository for [Python for Algorithmic Trading Cookbook](https://amzn.to/4706Exu), published by Packt.

**Recipes for designing, building, and deploying algorithmic trading strategies with Python**

## What is this book about?
Explore Python code recipes to use market data for designing and deploying algorithmic trading strategies. By following step-by-step instructions, you’ll be proficient in trading concepts and have hands-on experience in a live trading environment.

This book covers the following exciting features:
* Acquire and process freely available market data with the OpenBB Platform
* Build a research environment and populate it with financial market data
* Use machine learning to identify alpha factors and engineer them into signals
* Use VectorBT to find strategy parameters using walk-forward optimization
* Build production-ready backtests with Zipline Reloaded and evaluate factor performance
* Set up the code framework to connect and send an order to Interactive Brokers

If you feel this book is for you, get your [copy](https://amzn.to/4706Exu) today!

<a href="https://www.packtpub.com/?utm_source=github&utm_medium=banner&utm_campaign=GitHubBanner"><img src="https://raw.githubusercontent.com/PacktPublishing/GitHub/master/GitHub.png" alt="https://www.packtpub.com/" border="5" /></a>

## Instructions and Navigations
All of the code is organized into folders.

The code will look like the following:
```
import datetime as dt
import pandas as pd
from openbb_terminal.sdk import openbb
```

**Following is what you need for this book:**
Python for Algorithmic Trading Cookbook equips traders, investors, and Python developers with code to design, backtest, and deploy algorithmic trading strategies. You should have experience investing in the stock market, knowledge of Python data structures, and a basic understanding of using Python libraries like pandas. This book is also ideal for individuals with Python experience who are already active in the market or are aspiring to be.

With the following software and hardware list you can run all code files present in the book (Chapter 1-13).

### Software and Hardware List

| Chapter  | Software required                                                                    | OS required                        |
| -------- | -------------------------------------------------------------------------------------| -----------------------------------|
|  		1-13 | Python version 3.10   							                                            			  | Windows, Mac OS X, and Linux (Any) |
|      1-13|   	PostgreSQL																		                                  | Windows, Mac OS X, and Linux (Any)|
|  		1-13 |	OpenBB Platform version 4+ 					                                            			  | Windows, Mac OS X, and Linux (Any) |
|  		1-13 |pandas version 2+					                                            			  | Windows, Mac OS X, and Linux (Any) |


## Errata

* Page 34 (Code block snippet under section 5): **asset_2.volume.mean()** _should be_ **asset_2.volume[asset_2.index[5:10]].mean().astype(int)**
* Page 34 (second code block snippet under section 5): **asset_2.iat[10, 5]** _should be_ **asset_2.iat[10,4]**
* Page 34 (line 3 which is after second code block snippet): **The result is a scalar value representing the mean volume between indexes 5 and 10.** _should be_ **The result is a scalar value corresponding to the mean of the 'volume' column, assigned to the cell located at row 10 and column 4 (which corresponds to the 'volume' column).**
* Page: 36 - Into "2. Filter out the call..."  the `chains` Dataframe contains a column named "option_type" and not "optionType". So, wherever the filter **`(chains.optionType == "****")`** is used, it  _should instead be written as_ ** `(chains.option_type == "****")`**
* Page 49 . It is  stated as: **“Then we used the OpenBB Platform's `stocks.load` method …”**. However, the method actually used is **obb.equity.price.historical**.



## Get to Know the Author
**Jason Strimpel** is the founder of [PyQuant News](https://www.pyquantnews.com/) and co-founder of [Trade Blotter](https://www.tradeblotter.io/), with a career spanning over 20 years in trading, risk management, and data science. He previously traded for a Chicago-based hedge fund, served as a risk manager at JPMorgan, and managed production risk technology for an energy derivatives trading firm in London. In Singapore, Jason served as the APAC CIO for an agricultural trading firm and built the data science team for a global metals trading firm. He holds degrees in finance and economics and a Master’s in quantitative finance from the Illinois Institute of Technology. His career has taken him across America, Europe, and Asia. Jason shares his expertise through the [PyQuant Newsletter](https://www.pyquantnews.com/subscribe-to-the-pyquant-newsletter), social media, and teaches the course [Getting Started With Python for Quant Finance](https://www.pyquantnews.com/getting-started-with-python-for-quant-finance).

