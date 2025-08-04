
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
        return option["quantity"] * (intrinsic - option["premium"]) - fee_cost
    else:
        return option["quantity"] * (option["premium"] - intrinsic) - fee_cost


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
        total_cost += cost + fee
    else:
        total_cost -= cost - fee

st.markdown(f"💰 **总持仓成本：{total_cost:.2f} 元**")
