import numpy as np
import pandas as pd
pd.set_option('display.max_columns', 500)
import plotly.plotly as py
import plotly

## need to store somewhere before committing
plotly.tools.set_credentials_file(username=name, api_key=key)

## load sample data, aggregate by party, show on heat map (sample states)
df = pd.read_csv("/home/brett/Untitled1.csv")

df['vote_party_match'] = np.where(df['party'] == "I", 0,
                                  np.where(df['party'] == df['vote_party'], 1, 0))

map_dict = {1: "AZ", 2: "CA", 3: "OR"}
df = df.replace({"zip_code_proxy": map_dict})

df_agg = df.groupby(['zip_code_proxy', 'party'])['vote_party_match'].mean().reset_index()


scl = [[0.0, 'rgb(242,240,247)'],[0.2, 'rgb(218,218,235)'],[0.4, 'rgb(188,189,220)'],\
            [0.6, 'rgb(158,154,200)'],[0.8, 'rgb(117,107,177)'],[1.0, 'rgb(84,39,143)']]

# df_agg['text'] = df_agg['zip_code_proxy'] + '<br>' + df_agg['vote_party_match']

data = [ dict(
        type='choropleth',
        colorscale = scl,
        autocolorscale = False,
        locations = df_agg['zip_code_proxy'],
        z = df_agg['vote_party_match'].astype(float),
        locationmode = 'USA-states',
        text = 'this is the text',
        marker = dict(
            line = dict (
                color = 'rgb(255,255,255)',
                width = 2
            )
        ),
        colorbar = dict(
            title = "Millions USD"
        )
    ) ]

layout = dict(
        title = 'Voting Sample',
        geo = dict(
            scope='usa',
            projection=dict( type='albers usa' ),
            showlakes = True,
            lakecolor = 'rgb(255, 255, 255)',
        ),
    )

fig = dict( data=data, layout=layout )

url = py.plot( fig, filename='d3-cloropleth-map' )
