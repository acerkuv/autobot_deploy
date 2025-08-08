# bot/risk.py
def calculate_position_size(balance_usdt, entry_price, risk_pct=0.01, stop_loss_pct=0.02):
    risk_amount = balance_usdt * risk_pct
    position_size = risk_amount / (entry_price * stop_loss_pct)
    return min(position_size, balance_usdt * 0.1)

def get_tp_sl_levels(entry_price, strategy="moderate"):
    levels = {
        "moderate": {"tp": entry_price * 1.05, "sl": entry_price * 0.97}
    }
    return levels.get(strategy, levels["moderate"])