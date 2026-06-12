import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import random

# ==========================================================
# SMART-EVOPT : DATA GENERATION
# ==========================================================

NUM_EV = 1000

np.random.seed(42)

data = pd.DataFrame({
    "soc": np.random.uniform(20,100,NUM_EV),
    "soh": np.random.uniform(70,100,NUM_EV),
    "traffic_density": np.random.uniform(0,1,NUM_EV),
    "renewable_energy": np.random.uniform(50,500,NUM_EV),
    "grid_load": np.random.uniform(100,1000,NUM_EV),
    "electricity_price": np.random.uniform(5,20,NUM_EV),
    "charging_demand": np.random.uniform(10,100,NUM_EV)
})

# ==========================================================
# PREPROCESSING
# ==========================================================

scaler = MinMaxScaler()

scaled_data = scaler.fit_transform(data)

X = scaled_data[:,:-1]
y = scaled_data[:,-1]

X = X.reshape((X.shape[0],1,X.shape[1]))

# ==========================================================
# LSTM DEMAND PREDICTION
# ==========================================================

model = Sequential([
    LSTM(64,input_shape=(1,X.shape[2])),
    Dense(32,activation='relu'),
    Dense(1)
])

model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae']
)

model.fit(
    X,
    y,
    epochs=20,
    batch_size=32,
    verbose=1
)

predicted_demand = model.predict(X).flatten()

# ==========================================================
# BATTERY LIFECYCLE MANAGEMENT
# ==========================================================

def battery_degradation(soc,soh):

    degradation = (
        (100-soh)*0.4 +
        (100-soc)*0.2
    )

    return degradation

data["battery_deg"] = data.apply(
    lambda row:
    battery_degradation(
        row["soc"],
        row["soh"]
    ),
    axis=1
)

# ==========================================================
# RENEWABLE ENERGY SCHEDULING
# ==========================================================

def renewable_scheduler(
        renewable,
        demand):

    utilization = min(
        renewable,
        demand
    )

    return utilization

data["renewable_used"] = [
    renewable_scheduler(r,d)
    for r,d in zip(
        data["renewable_energy"],
        data["charging_demand"]
    )
]

# ==========================================================
# TRAFFIC AWARE ROUTING
# ==========================================================

def routing_cost(
        traffic,
        distance):

    return (
        traffic*10 +
        distance*0.5
    )

data["distance"] = np.random.uniform(
    1,
    30,
    NUM_EV
)

data["route_cost"] = [
    routing_cost(t,d)
    for t,d in zip(
        data["traffic_density"],
        data["distance"]
    )
]

# ==========================================================
# DAMEO OPTIMIZATION
# ==========================================================

def dameo_objective(row):

    charging_delay = (
        row["traffic_density"]*20
    )

    battery_cost = (
        row["battery_deg"]
    )

    grid_cost = (
        row["grid_load"]/100
    )

    renewable_bonus = (
        row["renewable_used"]/10
    )

    score = (
        charging_delay +
        battery_cost +
        grid_cost -
        renewable_bonus
    )

    return score

data["dameo_score"] = data.apply(
    dameo_objective,
    axis=1
)

# ==========================================================
# MULTI AGENT SYSTEM
# ==========================================================

class EVAgent:

    def __init__(
            self,
            ev_id,
            soc):

        self.ev_id = ev_id
        self.soc = soc

    def choose_station(self):

        return random.randint(1,10)

agents = [
    EVAgent(i,s)
    for i,s in enumerate(data["soc"])
]

stations = [
    a.choose_station()
    for a in agents
]

data["assigned_station"] = stations

# ==========================================================
# Q LEARNING AGENT
# ==========================================================

states = 10
actions = 5

Q = np.zeros(
    (states,actions)
)

alpha = 0.1
gamma = 0.9
epsilon = 0.1

for episode in range(1000):

    state = np.random.randint(states)

    if np.random.rand() < epsilon:
        action = np.random.randint(actions)
    else:
        action = np.argmax(Q[state])

    reward = np.random.uniform(0,100)

    next_state = np.random.randint(states)

    Q[state,action] += alpha * (
        reward +
        gamma*np.max(Q[next_state])
        - Q[state,action]
    )

# ==========================================================
# PERFORMANCE EVALUATION
# ==========================================================

renewable_utilization = (
    data["renewable_used"].sum()
    /
    data["renewable_energy"].sum()
)*100

avg_delay = (
    data["traffic_density"]*20
).mean()

grid_stability = (
    100 -
    (
        data["grid_load"].std()
        /
        data["grid_load"].mean()
    )*100
)

battery_health = (
    100 -
    data["battery_deg"].mean()
)

print("\nSMART-EVOPT RESULTS\n")

print(
    "Renewable Utilization :",
    round(
        renewable_utilization,
        2
    ),
    "%"
)

print(
    "Average Charging Delay :",
    round(
        avg_delay,
        2
    ),
    "min"
)

print(
    "Grid Stability Score :",
    round(
        grid_stability,
        2
    ),
    "%"
)

print(
    "Battery Health Score :",
    round(
        battery_health,
        2
    ),
    "%"
)

print(
    "Average DAMEO Score :",
    round(
        data["dameo_score"].mean(),
        2
    )
)