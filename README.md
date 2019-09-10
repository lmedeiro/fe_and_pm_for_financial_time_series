<h1> Repo Description </h1>
<p>
    This repo contains supporting files and report for 
    a research project done for Aalto University. All material on 
    this repo is free to be explored for your own use. If there are any 
    questions, please contact me. If you wish to cite or use it for 
    any further purposes, please contact me beforehand. 
</p>

<p> 
    For the report, you may refer to <a href="https://github.com/lmedeiro/fe_and_pm_for_financial_time_series/blob/master/project_report_LFM.pdf"> this pdf </a>. 
    Each jupyter notebook has its own model and purpose, therefore you may visit it and run the code. Models and machine learning related
    procedures are implemented with Pytorch, Skorch, and Scikit-learn. The main models used for this project were: LSTM RNNs, CNNs, and RFs. 
</p>

<p> The main dataset used for this work is <a href="https://github.com/lmedeiro/fe_and_pm_for_financial_time_series/blob/master/nasdaq_qqq_yahoo.pckl"> in this pickle file </a>.
It contains the structure that is known by the programs already. A few other pickle files are provided as well. 
You may test with those if desired. </p>

<p> Since this was a short, time restricted project, there was not much effort on the documentation. 
If you are interested enough, please contact me with questions.
</p>

<h1> Research Abstract </h1>
<h3> Feature Engineering and Predictive Modeling
for Financial Time Series Data </h3>
<p>
This paper presents a series of results achieved by attempting to
predict the closing price of a financial market time series. The data
source was Yahoo Finance, containing 19 years worth of daily stock
information, and an ETF tracking the Nasdaq 100 index (QQQ) was
chosen as the specific dataset to predict. A set of features for financial
time series data was engineered and five (5) models were explored:
Random Forest, LeNet, custom LeNet, ResNet, and a LSTM. The
goal was to use to Deep Neural Network models, while engineering
features that allow reliable prediction with only a single day worth
of information. The best performing models managed to achieve accuracy ranges between 83-85%, showing that the features engineered
have predictive value.

</p>