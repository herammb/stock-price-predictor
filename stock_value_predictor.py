import streamlit as st
import datetime as dt
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go

st.set_page_config(
    page_title="Make Predictions",
    page_icon="📊",
)

page_bg = """
    <style>
        [data-testid="stAppViewContainer"] {
            background-image: url("High_resolution_wallpaper_background_ID_77700392105.jpg");
            background-size: cover;
        }
    </style>
"""

st.markdown("", unsafe_allow_html=True)

st.title("Make Predictions")

start = "2010-01-01"
end = dt.datetime.now()

selected_stocks = st.text_input("Enter stock ticker", "AAPL")


n_years = 1
period = n_years*365

@st.cache_data   #stores the data so doesn't have to reload
def load_data(ticker):
    data = yf.download(ticker, start, end)
    data.reset_index(inplace=True)
    return data

data_load_state = st.text("Load data...")
data = load_data(selected_stocks)
data_load_state.text("Loading data... Done.")

st.subheader("Raw data")
st.write(data.tail())

#plotting graph for current raw data
def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data["Date"], y=data["Open"], name="opening_price"))
    fig.add_trace(go.Scatter(x=data["Date"], y=data["Close"], name="closing_price"))
    fig.layout.update(title_text="Time Series Data", xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_raw_data()

#forecasting data
df_train = data[["Date", "Close"]]
df_train = df_train.rename(columns={"Date":"ds", "Close":"y"}) 

model = Prophet()
model.fit(df_train)
future = model.make_future_dataframe(periods=period)
forecast = model.predict(future)

st.subheader("Forecast Data")
st.write(forecast.tail())

st.write("Forecast Data")
fig1 = plot_plotly(model, forecast)
st.plotly_chart(fig1)

