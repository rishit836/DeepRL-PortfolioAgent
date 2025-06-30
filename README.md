<h1 align="center">📈 LSTM Deep Q-Learning Trading Agent</h1>

<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-red?style=flat-square&logo=pytorch" />
  <img src="https://img.shields.io/badge/Finance-RL-blueviolet?style=flat-square&logo=chartdotjs" />
  <img src="https://img.shields.io/badge/LSTM-DeepLearning-yellow?style=flat-square&logo=python" />
</p>

---

## 🌟 Overview

A **Reinforcement Learning trading agent** built in **Python** using **LSTM** and **Deep Q-Learning**.  
The agent learns to **Buy**, **Sell**, or **Hold** a stock based on technical indicators and portfolio state.

Initially trained on **daily data**, the agent is designed to scale to **intraday data** (e.g. 5-minute intervals) for **higher-frequency decision-making**.

---

## 🧠 Core Idea

> Teach a deep RL agent to behave like a trader by feeding it meaningful time series data.  
> As it interacts with a simulated market, the agent learns to maximize its portfolio value through short-term gains.

---

## 🧩 Model Architecture

- 🧾 **Input**: 
  - Technical Indicators (RSI, MACD, etc.)
  - Price + Portfolio state (cash & shares)
  - 14-day sliding window sequences

- 🔁 **Model**:
  - LSTM layers to handle time sequences
  - Fully connected layer to predict Q-values

- 🎯 **Output**:
  - Q-values for 3 actions:
    - `Buy`
    - `Sell`
    - `Hold`

- 🎓 **Learning**:
  - Deep Q-Learning
  - Experience Replay
  - Epsilon-Greedy Exploration

---

## 📊 Technical Indicators Used

- `SMA_14`
- `EMA_14`
- `RSI_14`
- `MACD`
- `MACD Signal`
- `MACD Histogram`
- `Bullish / Bearish Crossovers`

---

## 📦 Project Structure

```bash
.
├── agent.py         # LSTM model & memory buffer
├── data.py          # Yahoo Finance data + indicators
├── env.py           # Custom stock environment (buy/sell/hold logic)
├── train.py         # Model training loop
├── evaluate.py      # Test model on unseen data
├── ticker_data/     # Optional: saved CSVs per stock
└── README.md        # You're here!
