import pandas as pd
import plotly.graph_objects as go

data = pd.read_csv("/Users/edwinkcijo/Desktop/just for india/Machine Learning/Covid/Cumulative/WHO-COVID-19-global-data.csv")
data.columns = ("Date","CC","Country","WHO","NCases","Cases","NDeaths","Deaths")

data["Date"] = pd.to_datetime(data["Date"])
land = input("What Country (give code) would you like the data of?\n> ") 

if land == "None":
    timeseries = data.groupby("Date").sum()

    # New Cases
    newcases = go.Figure()
    newcases.add_trace(go.Scatter(x=timeseries.index, y=data["NCases"], fill="tonexty", line_color="red"))
    newcases.update_layout(title="New Cases Daily Worldwide")
    newcases.write_html("Covid/Cumulative/html/ncases.html", auto_open=True)

    # Cumulative Cases
    sumcases = go.Figure()
    sumcases.add_trace(go.Scatter(x=timeseries.index, y=data["Cases"], fill="tonexty", line_color="cyan"))
    sumcases.update_layout(title="Accumulated Total Cases Worldwide")
    sumcases.write_html("Covid/Cumulative/html/cases.html", auto_open=True)

    # Death Cases
    newdeaths = go.Figure()
    newdeaths.add_trace(go.Scatter(x=timeseries.index, y=data["NDeaths"], fill="tonexty", line_color="lime"))
    newdeaths.update_layout(titile="New Deaths Daily Worlwide")
    newdeaths.write_html("Covid/Cumulative/html/ndeaths.html", auto_open=True)
else:
    countrycode = data[data["CC"] == land]
    timeseries = countrycode.groupby("Date").sum()
    name = countrycode["Country"].iloc[0]

    # New Cases
    newcases = go.Figure()
    newcases.add_trace(go.Scatter(x=timeseries.index, y=countrycode["NCases"], fill="tonexty", line_color="red"))
    newcases.update_layout(title=f"New Cases Daily of {name}")
    newcases.write_html("Covid/Cumulative/html/ncases.html", auto_open=True)

    # Cumulative Cases
    sumcases = go.Figure()
    sumcases.add_trace(go.Scatter(x=timeseries.index, y=data["Cases"], fill="tonexty", line_color="cyan"))
    sumcases.update_layout(title=f"Accumulated Total Cases of {name}")
    sumcases.write_html("Covid/Cumulative/html/cases.html", auto_open=True)

    # Death Cases
    newdeaths = go.Figure()
    newdeaths.add_trace(go.Scatter(x=timeseries.index, y=data["NDeaths"], fill="tonexty", line_color="lime"))
    newdeaths.update_layout(title=f"New Deaths Daily of {name}")
    newdeaths.write_html("Covid/Cumulative/html/ndeaths.html", auto_open=True)