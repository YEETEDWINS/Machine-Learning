import pandas as pd
import numpy as np
import plotly as pl
import plotly.express as exp
import plotly.graph_objects as gobj
from plotly.subplots import make_subplots

data = pd.read_csv("/Users/edwinkcijo/Desktop/just for india/Machine Learning/Covid/Plotly/covid_data.csv")
data = data[["Province_State","Country_Region","Confirmed","Recovered","Deaths","Active"]]
data.columns = ("State","Country","Confirmed","Recovered","Deaths","Active")
data["State"].fillna(value="",inplace=True)

# Bubble Plot
conf10 = pd.DataFrame(data.groupby("Country")["Confirmed"].sum().nlargest(10).sort_values(ascending=False))

bubble = exp.scatter(conf10,x=conf10.index,y="Confirmed",size="Confirmed",size_max=120,color=conf10.index,title="Top 10 Covid Spread Countries")
bubble.write_html("Covid/html/Plotly/BubblePlot.html",auto_open=True)

# Bar Chart ["#87CEEB","#00FF00"]
rec10 = pd.DataFrame(data.groupby("Country")["Recovered"].sum().nlargest(10).sort_values(ascending=False))

bar = exp.bar(rec10,x=rec10.index,y="Recovered",height=600,color="Recovered",color_continuous_scale=exp.colors.sequential.Turbo,title="Top 10 Countries beating Covid")
bar.write_html("Covid/html/Plotly/BarChart.html",auto_open=True)

# Bar Graph using GOBJ
topus = data["Country"] == "US"
topus = data[topus].nlargest(10, "Deaths")

bar2 = gobj.Figure(data=[
  gobj.Bar(name="Highest States of Deaths in US", x=topus["Deaths"], y=topus["Deaths"], width=3000,  orientation="h")
  #hoverlabel=topus["State"]
])

bar2.update_layout(title="Most Active Covid Cases in the US", height=1000, )
bar2.write_html("Covid/html/Plotly/BarChart2.html",auto_open=True)