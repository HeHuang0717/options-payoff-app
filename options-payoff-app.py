
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['xtick.direction'] = 'in'  # 将x周的刻度线方向设置向内
plt.rcParams['ytick.direction'] = 'in'  # 将y轴的刻度方向设置向内
plt.rcParams['font.sans-serif'] = ['Simhei']  # 显示中文
plt.rcParams['axes.unicode_minus'] = False  # 显示负号
plt.rcParams['font.size'] = 10

st.title("📈 奇异期权组合计算")

st.sidebar.header("标的价格范围")
price_min = st.sidebar.number_input("最低价格", value=50.0)
price_max = st.sidebar.number_input("最高价格", value=150.0)
price_step = st.sidebar.number_input("价格步长", value=1.0)
marginratio = st.sidebar.number_input("保证金比例", value=0.2)
stockprice = st.sidebar.number_input("当期股价", value=100)


# 用户输入期权信息
options = []
num_options = st.number_input("添加期权数量", min_value=1, max_value=10, value=2)

for i in range(num_options):
    st.subheader(f"期权 {i + 1}")
    cols = st.columns([1, 1, 1, 1, 1, 1])  # 6 列分别放各项参数

    kind = cols[0].selectbox(f"类型 {i + 1}", ["call", "put"], key=f"kind{i}")
    direction = cols[1].selectbox(f"方向 {i + 1}", ["long", "short"], key=f"dir{i}")
    strike = cols[2].number_input(f"执行价", key=f"strike{i}")
    premium = cols[3].number_input(f"权利金", key=f"premium{i}")
    quantity = cols[4].number_input(f"数量", value=1, step=1, key=f"qty{i}")
    fee = cols[5].number_input(f"手续费", value=1.0, key=f"fee{i}")

    options.append({
        "kind": kind,
        "direction": direction,
        "strike": strike,
        "premium": premium,
        "quantity": quantity,
        "fee": fee
    })

def calculate_payoff(option, price):
    # 如果是看多，内在价值是标的价格减去合约价格，如果是看空，内在价值是合约价格减去标的价格
    intrinsic = max(price - option["strike"], 0) if option["kind"] == "call" else max(option["strike"] - price, 0)
    # 手续费
    fee_cost = option["fee"] * option["quantity"]
    #如果是多头，内在价值减去付出的权利金
    #如果是空头，得到的权益金减去内在价值
    if option["direction"] == "long":
        return option["quantity"] * (intrinsic - option["premium"])
    else:
        return option["quantity"] * (option["premium"] - intrinsic)


# 模拟价格范围
price_range = np.arange(price_min, price_max + price_step, price_step)
total_payoffs = []

for price in price_range:
    total = sum([calculate_payoff(opt, price) for opt in options])
    total_payoffs.append(total)

# 绘图
fig, ax = plt.subplots()
ax.plot(price_range, total_payoffs, label="value")
ax.axhline(0, color="gray", linestyle="--")
ax.set_xlabel("strike")
ax.set_ylabel("option value")
ax.set_title("option map")
ax.grid(True)
st.pyplot(fig)

# 显示总持仓成本 空头得到权利金，多头失去权利金，因为是算成本所以计算相反
total_cost = 0
for opt in options:
    cost = opt["premium"] * opt["quantity"]
    fee = opt["fee"] * opt["quantity"]
    if opt["direction"] == "long":
        total_cost += cost
    else:
        total_cost -= cost

total_cost2 = 0
for opt in options:
    cost = opt["premium"] * opt["quantity"]
    fee = opt["fee"] * opt["quantity"]
    if opt["direction"] == "long":
        total_cost2 = total_cost2 + cost + fee
    else:
        total_cost2 = total_cost2 - cost + fee

st.markdown(f"💰 **总持仓成本不含手续费：{total_cost:.2f} 元**")

st.markdown(f"💰 **总持仓成本含手续费：{total_cost2:.2f} 元**")



def calculate_margin(option, underlying_price, margin_ratio=0.2):
    """
    使用 M = 权利金 + 行权价 × 保证金比例 - 虚值额
    仅适用于卖出期权（short call / short put）

    买入期权是
    """
    if option["direction"] != "short":
        cost = (opt["premium"] + opt["fee"]) * opt["quantity"]
        return cost

    premium = option["premium"]
    strike = option["strike"]
    qty = option["quantity"]

    # 计算虚值额
    if option["kind"] == "call":
        otm = max(underlying_price - strike, 0)
    elif option["kind"] == "put":
        otm = max(strike - underlying_price, 0)
    else:
        otm = 0
    margin_per_contract = premium + strike * margin_ratio - otm
    margin_per_contract = max(margin_per_contract, 0)  # 保证金不能为负
    cost =  margin_per_contract * qty + opt["fee"] * opt["quantity"]
    return cost




def total_margin_capital(options, underlying_price, margin_ratio=0.2):
    return sum(
        calculate_margin(opt, underlying_price, margin_ratio)
        for opt in options
    )

capital_required = total_margin_capital(options, stockprice)
st.write(f"💰 需要冻结的保证金资金总额和手续费（基于当前价格 {stockprice}）：{capital_required:.2f}")


return_rate_data = []
for price in price_range:
    total_profit = sum(calculate_payoff(opt, price) for opt in options)
    total_capital = total_margin_capital(options, underlying_price=stockprice)
    if total_capital == 0:
        return_rate = 0
    else:
        return_rate = total_profit / total_capital
    return_rate_data.append({
        "price": price,
        "return_rate": return_rate * 100  # 转为百分比
    })





import matplotlib.pyplot as plt

prices = [d["price"] for d in return_rate_data]
returns = [d["return_rate"] for d in return_rate_data]

fig, ax = plt.subplots()
ax.plot(prices, returns, label="Earning rate（%）", color="green")
ax.axhline(0, color="gray", linestyle="--")
ax.set_xlabel("Strike")
ax.set_ylabel("Earning rate (%)")
ax.set_title("Earning rate vs Strike Price")
ax.grid(True)
st.pyplot(fig)




# st.write(f"总投入资金（含保证金）：{capital:.2f} 元")
# st.write(f"收益率（在当前价格 {price:.2f}）：{return_rate*100:.2f} %")




# fig, ax = plt.subplots()
# ax.plot(price_range, total_payoffs, label="value")
# ax.axhline(0, color="gray", linestyle="--")
# ax.set_xlabel("strike")
# ax.set_ylabel("option value")
# ax.set_title("option map")
# ax.grid(True)