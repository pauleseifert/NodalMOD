import numpy as np
import pandas as pd
import plotly.graph_objects as go
# pio.kaleido.scope.mathjax = None
# pio.renderers.default = "browser"
from colour import Color
from plotly.subplots import make_subplots


# def graph_poncelet(full_ts, poncelet_ts):
#     #row = 0
#     test = pd.DataFrame(np.sort(full_ts.values, axis=0)[::-1], index=full_ts.index, columns=full_ts.columns)
#     test2 = test/test.iloc[0,:]
#     test3 = test2.sum(axis= 1)/1794
#
#     test4 = pd.DataFrame(np.sort(poncelet_ts.values, axis=0)[::-1], index=poncelet_ts.index, columns=poncelet_ts.columns)
#     test5 = test4 / test4.iloc[0, :]
#     test6 = test5.sum(axis=1) / 1794
#     fig = go.Figure()
#     fig.add_trace(go.Scatter(x=test3.index/8760, y=test3,
#                     mode='lines',
#                     name='long'))
#     fig.add_trace(go.Scatter(x=test6.index/336, y=test6,
#                     mode='lines',
#                     name='shortened'))
#     fig.show()
#
# location = "data/north_sea_energy_islands/csv/"
# full_ts = pd.read_csv(location+"renewables_for_poncelet.csv", index_col=0)
# poncelet_ts = pd.read_csv(location+"poncelet_ts.csv", index_col=0)

def plotly_maps_lines_colorless(P_flow, P_flow_DC,bus, scen, maps_folder):
    P_flow=P_flow.reset_index()
    P_flow_DC = P_flow_DC.reset_index()
    #P_flow['text'] = "AC line number "+P_flow['index'].astype(str) + '<br>Mean loading ' + (round(P_flow[0] *100,1)).astype(str) + ' %'
    #P_flow_DC['text'] = "DC line number " + P_flow_DC['index'].astype(str) + '<br>Mean loading ' + (
        #round(P_flow_DC[0] * 100, 1)).astype(str) + ' %'
    df = pd.concat([P_flow, P_flow_DC], axis=0, ignore_index=True).reset_index(drop=True)
    lons = np.empty(3 * len(df))
    lons[::3] = df["LON_x"]
    lons[1::3] = df["LON_y"]
    lons[2::3] = None
    lats = np.empty(3 * len(df))
    lats[::3] = df["LAT_x"]
    lats[1::3] = df["LAT_y"]
    lats[2::3] = None

    fig = go.Figure()
    fig.add_trace(
        go.Scattermapbox(
            lon=lons,
            lat=lats,
            mode='lines',
            line=dict(
                width=3,
                ),
            opacity=1,
            name="Line"
        )
    )
    fig.add_trace(
        go.Scattermapbox(
        lon=bus['LON'],
        lat=bus['LAT'],
        hoverinfo='text',
        text=bus.index,
        mode='markers',
        marker=dict(
            size=2,
            color='rgb(0, 0, 0)'
            ),
        name = "Bus"
        ))

    fig = plotly_map(fig, 4.7)
    fig.update_layout(
        font=dict(size=30,
                  #family = "Serif"
                  ),
        legend_title_text='Grid element',
    )
    #fig.show()
    fig.write_html(maps_folder + str(scen) + "_line_loading_colorless.html")
    fig.write_image(maps_folder + str(scen) + "_line_loading_colorless.pdf", width=2000, height=1600)
    #fig.show()

def plot_generation(generation_temporal, maps_folder, scen, year):
    scaling_factor=1000
    fig = go.Figure()
    if scen != 1:
        fig.add_trace(go.Scatter(x=generation_temporal[year].electrolyser.index, y=-generation_temporal[year].electrolyser / scaling_factor,name='Electrolysis', fill='tozeroy', stackgroup='two', mode='none',fillcolor="rgba(0, 230, 230, 0.8)"))
    fig.add_trace(go.Scatter(x=generation_temporal[year].C_S.index, y=-generation_temporal[year].C_S / scaling_factor, name='Storage charge',fill='tonexty', stackgroup='two', mode='none', fillcolor="rgba(153, 0, 153, 0.8)"))
    #fig.add_trace(go.Scatter(x = generation_temporal[year].other.index, y = generation_temporal[year].other/scaling_factor, name= 'Other', fill = 'tonexty', stackgroup= 'one', mode= 'none', fillcolor= "rgba(128, 128, 128, 0.8)"))
    fig.add_trace(go.Scatter(x=generation_temporal[year].biomass.index, y=generation_temporal[year].biomass / scaling_factor,name='Biomass', fill='tonexty', stackgroup='one', mode='none', fillcolor="rgba(1, 135, 4,  0.8)"))
    fig.add_trace(go.Scatter(x = generation_temporal[year].nuclear.index, y = generation_temporal[year].nuclear/scaling_factor, name= 'Nuclear', fill = 'tonexty', stackgroup= 'one', mode= 'none', fillcolor= "rgba(255, 0, 0, 0.8)"))
    fig.add_trace(go.Scatter(x = generation_temporal[year].coal.index, y = generation_temporal[year].coal/scaling_factor, name= 'Coal', fill = 'tonexty', stackgroup= 'one', mode= 'none', fillcolor= "rgba(77, 38, 0, 0.8)"))
    fig.add_trace(go.Scatter(x = generation_temporal[year].lignite.index, y = generation_temporal[year].lignite/scaling_factor, name= 'Lignite', fill = 'tonexty', stackgroup= 'one', mode= 'none', fillcolor= "rgba(179, 89, 0, 0.8)"))
    fig.add_trace(go.Scatter(x = generation_temporal[year].oil.index, y = generation_temporal[year].oil/scaling_factor, name= 'Oil', fill = 'tonexty', stackgroup= 'one', mode= 'none', fillcolor= "rgba(0, 0, 0, 0.8)"))
    fig.add_trace(go.Scatter(x = generation_temporal[year].gas.index, y = generation_temporal[year].gas/scaling_factor, name= 'Gas', fill = 'tonexty', stackgroup= 'one', mode= 'none', fillcolor= "rgba(255, 165, 0, 0.8)"))
    fig.add_trace(go.Scatter(x = generation_temporal[year].hydro.index, y = generation_temporal[year].hydro/scaling_factor, name= 'Hydro', fill = 'tonexty', stackgroup= 'one', mode= 'none', fillcolor= "rgba(0, 0, 255, 0.8)"))
    fig.add_trace(go.Scatter(x = generation_temporal[year].wind.index, y = generation_temporal[year].wind/scaling_factor, name= 'Wind', fill = 'tonexty', stackgroup= 'one', mode= 'none', fillcolor= "rgba(198, 210, 190, 0.8)"))
    fig.add_trace(go.Scatter(x = generation_temporal[year].solar.index, y = generation_temporal[year].solar/scaling_factor, name= 'Solar', fill = 'tonexty', stackgroup= 'one', mode= 'none', fillcolor= "rgba(255, 255, 0, 0.8)"))
    fig.add_trace(go.Scatter(x=generation_temporal[year].P_S.index, y=generation_temporal[year].P_S / scaling_factor,name='Storage discharge', fill='tonexty', stackgroup='one', mode='none', fillcolor="rgba(145, 0, 0, 0.8)"))
    fig.add_trace(go.Scatter(x=generation_temporal[year].curtailment.index, y=generation_temporal[year].curtailment/scaling_factor,name='Curtailment', fill='tonexty', stackgroup='one', mode='none', fillcolor="rgba(0, 77, 0, 0.8)"))

    fig.update_layout(
        xaxis= dict(title='Timesteps',dtick=24),
        yaxis=dict(title='Generation [GW]'),
        font = dict(size = 30,
                    #family = "Serif"
                    ),
        legend=dict(x=0, y=-0.15, bgcolor='rgba(255, 255, 255, 0)', bordercolor='rgba(255, 255, 255, 0)',font_size = 22),
        legend_orientation="h",
        plot_bgcolor='rgba(0,0,0,0)',
        yaxis_gridcolor = "rgba(166, 166, 166, 0.5)"
    )
    #fig.write_html(maps_folder + "scen_"+str(scen) +"_year_"+str(year) + "_electricity_gen.html")
    fig.write_image(maps_folder + "scen_"+str(scen) +"_year_"+str(year) + "_electricity_gen.pdf", width=2000, height=800)


def plotly_map(fig, zoom, hoffset=0, voffset = 0):
    fig.update_layout(
        margin={'l': 0, 't': 0, 'b': 0, 'r': 0},
        mapbox={
            'center': {'lon': 9.6750+voffset, 'lat': 59.5236+hoffset},
            'style': "white-bg",
            'zoom': zoom},

        mapbox_layers=[
            {
                "below": 'traces',
                "sourcetype": "raster",
                "sourceattribution": 'Tile © Esri, Esri, DeLorme, NAVTEQ',
                "source": [
                    "https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}"
                ]
            },
        ],
    )
    return fig

def plotly_maps_bubbles(df, scen, maps_folder, name, year, unit, size_scale, title, flexible_bubble_size =True, zoom = 1, hoffset = 0, voffset =0, max_legend = False,min_legend = False,):
    unit_dict = {"TWh": 1000000, "GWh":1000, "GW":1000}
    columns = [year, "LAT", "LON"]
    fig = go.Figure()
    df = df[columns]
    #for idx, row in df.iterrows():
    fig.add_trace(
        go.Scattermapbox(
            lon=df['LON'],
            lat=df['LAT'],
            text=df.index,
            marker=dict(
                size=round(df[year]/(unit_dict[unit]/5000*size_scale),5) if flexible_bubble_size == True else 30,
                color=round(df[year]/unit_dict[unit],5),
                sizemode='area',
                showscale = True,
                colorbar=dict(title=title+" in "+unit, orientation = "h", y = -0.1, xpad = 100),
                cmin = min_legend,
                cmax = max_legend
            ),
            name="Node"+str(df.index),
        ),
    )
    fig = plotly_map(fig, 4.7*zoom, hoffset = hoffset, voffset=voffset)
    fig.update_layout(
        font = dict(size=30)
    )
    #fig.show()


    #fig.write_html(maps_folder + str(scen) + "_"+name+".html")
    fig.write_image(maps_folder + "scen_"+str(scen) +"_year_"+str(year) + "_"+name+".pdf", width=2000, height=1600)

def plotly_maps_lines(P_flow, P_flow_DC,bus, scen, maps_folder):
    colors = list(Color("green").range_to(Color("red"), 8))
    limits = [(0, 10), (10, 20), (20, 30), (30, 40), (40, 50), (50, 60), (60, 70), (70, 75)]
    text = ["0%-10%", "10%-20%", "20%-30%","30%-40%", "40%-50%", "50%-60%", "60%-70%", "70%-75%"]
    #P_flow_DC = P_flow_DC[P_flow_DC["Pmax"]> 0.0]
    #P_flow['text'] = "AC line number "+P_flow['index'].astype(str) + '<br>Mean loading ' + (round(P_flow[0] *100,1)).astype(str) + ' %'
    #P_flow_DC['text'] = "DC line number " + P_flow_DC['index'].astype(str) + '<br>Mean loading ' + (round(P_flow_DC[0] * 100, 1)).astype(str) + ' %'
    df = pd.concat([P_flow, P_flow_DC], axis=0, ignore_index=True).reset_index(drop=True)
    fig = go.Figure()
    limits_groups = {}
    for i in range(len(limits)):
        limits_groups.update({i:df[(df[0] < limits[i][1]/100) & (df[0] >= limits[i][0]/100)]})
        lons = np.empty(3* len(limits_groups[i]))
        lons[::3] = limits_groups[i]["LON_x"]
        lons[1::3] = limits_groups[i]["LON_y"]
        lons[2::3] = None
        lats = np.empty(3 * len(limits_groups[i]))
        lats[::3] = limits_groups[i]["LAT_x"]
        lats[1::3] = limits_groups[i]["LAT_y"]
        lats[2::3] = None
        fig.add_trace(
            go.Scattermapbox(
                lon=lons,
                lat=lats,
                mode='lines',
                hovertemplate=
                '<i>Price</i>: $%{lons:.2f}' +
                '<br><b>X</b>: %{lats}<br>' +
                '<b>%{text}</b>',
                text=["line nr. " +str(row.index)for i, row in limits_groups[i].iterrows()],
                line=dict(
                    width=3,
                    color=str(colors[i])
                    ),
                opacity=1,
                name=text[i],
            )
        )
    fig.add_trace(
        go.Scattermapbox(
        lon=bus['LON'],
        lat=bus['LAT'],
        hoverinfo='text',
        text=["bus nr." + str(row["index"]) for i,row in bus.iterrows()],
        mode='markers',
        marker=dict(
            size=2,
            color='rgb(0, 0, 0)'
            ),
        name = "Bus"
        ))

    fig = plotly_map(fig, 4.7)
    fig.update_layout(
        font=dict(size=30,
                  #family = "Serif"
                  ),
        legend_title_text='Line loading',
    )
    #fig.show()
    #fig.write_html(maps_folder + str(scen) + "_line_loading.html")
    fig.write_image(maps_folder + "scen_"+str(scen) + "_line_loading.pdf", width=2000, height=1600)
    #fig.show()
def plotly_maps_lines_hours(P_flow, P_flow_DC,bus, scen, maps_folder, timesteps):
    colors = list(Color("green").range_to(Color("red"), 10))
    limits = [(int(round(i*timesteps/10,0)), int(round((i+1)*timesteps/10 ,0)))for i in range(10)]
    text = [(str(i*10)+"% - "+str((i+1)*10)+"%")for i in range(10)]
    #P_flow['text'] = "AC line number "+P_flow['index'].astype(str) + '<br>Mean loading ' + (round(P_flow[0] *100,1)).astype(str) + ' %'
    #P_flow_DC['text'] = "DC line number " + P_flow_DC['index'].astype(str) + '<br>Mean loading ' + (
    #    round(P_flow_DC[0] * 100, 1)).astype(str) + ' %'
    df = pd.concat([P_flow, P_flow_DC], axis=0, ignore_index=True).reset_index(drop=True)
    fig = go.Figure()
    limits_groups = {}
    limits_groups.update({0: df[(df["full_load_h"] <= limits[0][1]) & (df["full_load_h"] >= limits[0][0])]})
    limits_groups.update({i: df[(df["full_load_h"] <= limits[i][1]) & (df["full_load_h"] > limits[i][0])] for i in range(1, len(limits))})
    for i in range(len(limits)):
        lons = np.empty(3* len(limits_groups[i]))
        lons[::3] = limits_groups[i]["LON_x"]
        lons[1::3] = limits_groups[i]["LON_y"]
        lons[2::3] = None
        lats = np.empty(3 * len(limits_groups[i]))
        lats[::3] = limits_groups[i]["LAT_x"]
        lats[1::3] = limits_groups[i]["LAT_y"]
        lats[2::3] = None
        fig.add_trace(
            go.Scattermapbox(
                lon=lons,
                lat=lats,
                mode='lines',
                line=dict(
                    width=3,
                    color=str(colors[i])
                    ),
                opacity=1,
                name=text[i],
            )
        )
    fig.add_trace(
        go.Scattermapbox(
        lon=bus['LON'],
        lat=bus['LAT'],
        hoverinfo='text',
        text=bus.index,
        mode='markers',
        marker=dict(
            size=2,
            color='rgb(0, 0, 0)'
            ),
        name = "Bus"
        ))

    fig = plotly_map(fig, 4.7)
    fig.update_layout(
        font=dict(size=30,
                  #family = "Serif"
                  ),
        legend_title_text='Average hours with <br> line congestions',
    )
    #fig.show()
    #fig.write_html(maps_folder + str(scen) + "_line_loading_h.html")
    fig.write_image(maps_folder + "scen_"+str(scen)+ "_line_loading_h.pdf", width=2000, height=1600)
    #fig.show()
def plotly_maps_size_lines(P_flow, P_flow_DC, CAP_lines, bus, scen,year, maps_folder, zoom, offset):
    #colors = list(Color("white").range_to(Color("black"), 8))
    limits = [(0, 10), (10, 20), (20, 30), (30, 40), (40, 50), (50, 60), (60, 70), (70, 75)]
    flexlines = CAP_lines[CAP_lines["EI"].notna()][[year, "from", "LON_x", "LAT_x", "LON_y", "LAT_y"]]
    flexlines_used = flexlines[flexlines[year] !=0]
    flexlines_not_used = flexlines[flexlines[year] ==0]
    flex_busses = list(flexlines['from'].unique())
    cluster_busses = set(flex_busses)-set([521, 522,523])

    df_normal_lines = pd.concat([P_flow[year], P_flow_DC[year][~P_flow_DC[year].index.isin(CAP_lines.index)]], axis=0, ignore_index=True).reset_index(drop=True)

    fig = go.Figure()
    limits_groups = {}
    for i in range(len(limits)):
        limits_groups.update({i:df_normal_lines[(df_normal_lines[0] < limits[i][1]/100) & (df_normal_lines[0] >= limits[i][0]/100)]})
        lons = np.empty(3* len(df_normal_lines))
        lons[::3] = df_normal_lines["LON_x"]
        lons[1::3] = df_normal_lines["LON_y"]
        lons[2::3] = None
        lats = np.empty(3* len(df_normal_lines))
        lats[::3] = df_normal_lines["LAT_x"]
        lats[1::3] = df_normal_lines["LAT_y"]
        lats[2::3] = None
        fig.add_trace(
            go.Scattermapbox(
                lon=lons,
                lat=lats,
                mode='lines',
                line=dict(
                    width=0.5,
                    color="grey",
                    ),
                opacity=0.2,
                #text="test2", #problem with static image export
                #name="other grid_lines",
                showlegend=False
            )
        )
    for i, row in flexlines_used.iterrows(): #memo an mich: das geht nur so, da width eine einzelne Zahl sein muss
        lons2 = np.empty(3)
        lons2[0] = row["LON_x"]
        lons2[1] = row["LON_y"]
        lons2[2] = None
        lats2 = np.empty(3)
        lats2[0] = flexlines.loc[i]["LAT_x"]
        lats2[1] = flexlines.loc[i]["LAT_y"]
        lats2[2] = None
        fig.add_trace(
            go.Scattermapbox(
            lon=lons2,
            lat=lats2,
            #hoverinfo=flexlines.loc[i]["Pmax"],
            #text=flexlines.loc[i],
            mode='lines',
            line=dict(
                width=flexlines.loc[i][year]/700,
                color='rgb(212, 46, 46)'
                ),
            showlegend=False
            ))
    lons = np.empty(3 * len(flexlines_not_used))
    lons[::3] = flexlines_not_used["LON_x"]
    lons[1::3] = flexlines_not_used["LON_y"]
    lons[2::3] = None
    lats = np.empty(3 * len(flexlines_not_used))
    lats[::3] = flexlines_not_used["LAT_x"]
    lats[1::3] = flexlines_not_used["LAT_y"]
    lats[2::3] = None
    # fig.add_trace(
    #     go.Scattermapbox(
    #         lon=lons,
    #         lat=lats,
    #         #text = "test",
    #         mode='lines',
    #         line=dict(
    #             width=0.8,
    #             color="blue",
    #         ),
    #         opacity=0.8,
    #         #name="not used lines",
    #         showlegend=False
    #     )
    # )
    fig.add_trace(
        go.Scattermapbox(
        lon=bus.query("index not in @flex_busses")['LON'],
        lat=bus.query("index not in @flex_busses")['LAT'],
        text=bus.index,
        mode='markers',
        showlegend= False,
        marker=dict(
            size=2,
            color='rgb(0, 0, 0)'
            ),
        name = "Bus"
        ))
    fig.add_trace(
        go.Scattermapbox(
        lon=bus.query("index in @cluster_busses")['LON'],
        lat=bus.query("index in @cluster_busses")['LAT'],
        text=bus.index,
        mode='markers',
        showlegend= False,
        marker=dict(
            size=8,
            color='rgb(0, 0, 0)'
            ),
        name = "Wind cluster"
        ))
    fig.add_trace(
        go.Scattermapbox(
        lon=bus.query("index in [521,522,523]")['LON'],
        lat=bus.query("index in [521, 522, 523]")['LAT'],
        #hoverinfo='text',
        mode='markers',
        showlegend= False,
        textposition='top center',
        marker=dict(
            size=12,
            color='rgb(255, 191, 0)'
            ),
        name = "EI"
        ))
    #create the stupid legend
    fig.add_traces(
        [
            go.Scattermapbox(
                name=row["title"],
                lat =[1],
                lon =[1],
                mode='lines',
                line=dict(
                    width=2,
                    color=row["colors"],
                ),
                showlegend=False)
            for i,row in pd.DataFrame({"colors":["Blue", "Black", "white"], "title":["non executed cable option", "Cable connection", "Grid"]}).iterrows()
        ]
    )
    fig = plotly_map(fig, 4.7*zoom, hoffset = offset)
    fig.update_layout(
        font=dict(size=30,
                  #family = "Serif"
                  ),
        legend_title_text='EI connections',
    )
    #fig.update_traces(mode="lines+markers+text", selector = dict(type='scattermapbox'))

    #fig.show()
    #fig.write_html(maps_folder + str(scen) + "_line_size.html")
    fig.write_image(maps_folder + "scen_"+str(scen) +"_line_size.pdf", width=2000, height=1600)
    #fig.show()

def EI_trade_plot(EI_trade, P_R_raw, p_gen_lost, P_H, maps_folder, scen, y, EI):
    fig = go.Figure()
    #colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3', '#FF6692', '#B6E880', '#FF97FF', '#FECB52']
    #colors = ['rgb(73, 103, 170)', 'rgb(253, 207, 65)', 'rgb(255, 150, 100)', 'rgb(255, 102, 194)', 'rgb(133, 134, 198)',
             #'rgb(72, 184, 231)', 'rgb(80, 214, 145)', 'rgb(210, 162, 124)', 'rgb(185, 194, 200)']
    colors = ["rgb(246, 81, 29)", "rgb(47, 109, 38)", "rgb(25, 123, 189)", "rgb(239, 202, 8)", "rgb(188, 237, 246)",
     "rgb(239, 156, 218)", "rgb(210, 162, 124)", 'rgb(185, 194, 200)']
    counter = 0
    fig.add_trace(go.Bar(x=P_R_raw.index, y=P_R_raw, marker_line_width=0, name='Wind infeed',
                         marker_color=colors[counter]))
    fig.add_trace(go.Bar(x=p_gen_lost.index, y=p_gen_lost, name="Curtailment", opacity=0.7,
                         marker_line_width=0,
                         marker_color="black"))
    if scen in [3,4,5,6]:
        #fig.add_trace(go.Bar(x=P_H.index, y=-P_H["0"], name=r'$\textrm{Electrolyser in P}_\textrm{el}$',marker_line_width=0, opacity=0.7, marker_color= colors[5]))
        fig.add_trace(go.Bar(x=P_H.index, y=-P_H, name="Electrolysis", marker_line_width=0, opacity=0.7, marker_color=colors[5]))
    counter += 1
    if scen in [1,2,3,4,5]:
        for index, row in EI_trade.iterrows():
            fig.add_trace(go.Bar(x=row.index, y=-row, name=index, marker_line_width=0, opacity=0.7, marker_color= colors[counter]))
            counter += 1
    fig.update_layout(
        barmode='relative',
        bargap = 0,
        bargroupgap=0,
        #font_family="Serif",
    )
    fig.update_layout(
        xaxis=dict(title='Timestep',dtick=24),
        yaxis=dict(title='Mass balance [MW]'),
        font=dict(size=35),
        legend=dict(x=0, y=-0.2, bgcolor='rgba(255, 255, 255, 0)', bordercolor='rgba(255, 255, 255, 0)', valign = "top", borderwidth=1),
        legend_title_text="",
        legend_orientation="h",
        plot_bgcolor='rgba(0,0,0,0)',
        yaxis_gridcolor = "rgba(166, 166, 166, 0.5)"
    )
    fig.write_html(maps_folder + str(scen) + "_EI"+str(EI) +"_" + str(y)+ "_import.html")
    fig.write_image(maps_folder + str(scen) +"_EI"+ str(EI) + "_" + str(y) + "_import.pdf", width=2000, height=800)
def plot_bar2_electrolyser(df, title, caption, position_label, maps_folder):

    colors = ["#8CBEB2", "#F2EBBF", "#F3B562", "#F06060"]
    fig = go.Figure(
        data=[
            go.Bar(
                name=EI,
                x=df.index,
                y=df[scen][EI],
                offsetgroup=offsetg,
                base = [df[scen].iloc[:,0:z].sum(axis=1) for z in range(4)][EI_number], # not wrong! the first query(z) must give back 0, z = EI
                #base = [df[scen].loc[df.columns.get_level_values(1)[0]:z].sum(axis=1) for z in df.columns.get_level_values(1).unique()][EI_number],
                showlegend = True if offsetg ==0 else False,
                marker = dict(
                    color = colors[EI_number],
                    line = dict(
                        width = 0.5
                    )
                ),
                text=scen if EI_number == position_label else "",
                textposition= "outside",
                cliponaxis=False,
                #insidetextanchor="start"
                ) for scen, offsetg in zip(df.columns.get_level_values(0).unique(), [0,1,2,3]) for EI, EI_number in zip(df.columns.get_level_values(1).unique(), [0,1,2,3])
            ],
        layout=go.Layout(
            #title="Electrolyser capacity in the scenarios",
            font=dict(size=30,
                      #family="Serif"
                      ),
            yaxis=dict(
                tickformat = ".n",
                title = title,
                gridcolor="rgba(166, 166, 166, 0.5)"
            ),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            #gridcolor="rgba(166, 166, 166, 0.5)",
            #barmode="stack"
        )
    )
    #fig.write_html(maps_folder +caption+".html")
    fig.write_image(maps_folder + caption +".pdf", width=2000, height=800)

def plot_bar_yearstack(df, title, caption, position_label, maps_folder):

    unique_scenarios = df.columns.get_level_values(0).unique()
    unique_EI = df.columns.get_level_values(1).unique()
    unique_years = df.index
    df2 = df.reorder_levels([1,0], axis=1)
    #df2 = df

    colors = ["#8CBEB2", "#F2EBBF", "#F3B562", "#F06060"]
    fig = go.Figure(
        data=[
            go.Bar(
                name=EI,
                x=unique_scenarios,
                y=df2[EI].loc[year],
                offsetgroup=offsetg,
                #base = [data[scen].iloc[:,0:z].sum(axis=1) for z in range(4)][offsetg]
                base = [df2.loc[year].iloc[:,unique_EI[0]:z].sum(axis=0) for z in range(4)][EI_number],
                showlegend = True if offsetg ==0 else False,
                marker = dict(
                    color = colors[EI_number],
                    line = dict(
                        width = 0.5
                    )
                ),
                text= year if EI_number == position_label else "",
                textposition= "outside",
                cliponaxis=False,
                #insidetextanchor="start"
                ) for year, offsetg in zip(unique_years, [0,1,2]) for EI, EI_number in zip(unique_EI, [0,1,2,3])
            ],
        layout=go.Layout(
            #title="Electrolyser capacity in the scenarios",
            font=dict(size=30,
                      #family="Serif"
                      ),
            yaxis=dict(
                tickformat = ".n",
                title = title,
                gridcolor="rgba(166, 166, 166, 0.5)"
            ),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            #gridcolor="rgba(166, 166, 166, 0.5)",
            #barmode="stack"
        )
    )
    #fig.write_html(maps_folder +caption+".html")
    fig.write_image(maps_folder + caption +".pdf", width=2000, height=800)



def kpi_development(data, title, folder):
    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Add traces
    colors = ["#8CBEB2", "#F2EBBF", "#F3B562", "#F06060", "#6495ED"]
#    for i,j in zip(data["curtailment"].columns, [0,1,2,3,4]):
#        fig.add_trace(go.Scatter(x=data["curtailment"].index, y=data["curtailment"][i], name=data["curtailment"][i].name, line=dict(color = colors[j]), legendgroup = "curtailment", legendgrouptitle = dict(text = "Curtailment")),secondary_y=False)
    # line=dict(color = colors[j])
    #colors = ["rgb(201, 255, 247)", "rgb(119, 178, 164)", "rgb(83, 148, 132)", "rgb(60, 105, 94)", "rgb(36, 63, 57)"]
    for i,j in zip(data["conventionals"].columns, [0,1,2,3,4]):
        fig.add_trace(go.Bar(x=data["conventionals"].index, y=data["conventionals"][i], marker_color = colors[j], name=data["conventionals"][i].name), secondary_y=False)
    #
    for i,j in zip(data["electrolyser capacity"].columns, [0,1,2,3,4]):
        fig.add_trace(go.Scatter(x=data["electrolyser capacity"].index, y=data["electrolyser capacity"][i], name=data["electrolyser capacity"][i].name, line=dict(color =colors[j]), showlegend = False), secondary_y=True,)
    fig.update_layout(
        #title=title,
        font=dict(size=30),
        yaxis=dict(
            tickformat=".n",
            nticks = 10,
            title="Conventionals [TWh]",),
        xaxis = dict(
            title = "year",
            dtick = 5),
        paper_bgcolor='rgba(0,0,0,0)',
        #legend=dict(x=0, y=-0.15, bgcolor='rgba(255, 255, 255, 0)', bordercolor='rgba(255, 255, 255, 0)',),
        #legend_orientation="h",
        plot_bgcolor='rgba(0,0,0,0)',
        yaxis_gridcolor = "rgba(166, 166, 166, 0.5)",
        legend=dict(
            #title="Sensitivity Scenario",
            x=1.05, y=1
        )
    )
    #fig.for_each_trace(lambda t: fig.add_annotation(
    #    x=t.x[-1], y=t.y[-1], text=t.name,
    #    font_color=t.line.color,
    #    ax=5, ay=0, xanchor="left", showarrow=False))
    fig.update_yaxes(title_text="Electrolyser capacity [GW]",nticks = 10, tickmode = "auto", secondary_y=True)
    fig.write_image(folder + "KPIs_sensitivity.pdf", width=2000, height=800)
def kpi_development2(data, title, folder):
    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Add traces
    for i in data["curtailment"].columns:
        fig.add_trace(
            go.Scatter(x=data["curtailment"].index, y=data["curtailment"][i], name=data["curtailment"][i].name, line=dict(color = "red")),
            secondary_y=False,# legendgroup = "test", legendgrouptitle = "hallo!"
        )

    for i in data["electrolyser capacity"].columns:
        fig.add_trace(
            go.Scatter(x=data["electrolyser capacity"].index, y=data["electrolyser capacity"][i], name=data["electrolyser capacity"][i].name, line=dict(color = "blue")),
            secondary_y=True,
        )
    for i in data["hydrogen production"].columns:
        fig.add_trace(
            go.Scatter(x=data["hydrogen production"].index, y=data["hydrogen production"][i], name=data["hydrogen production"][i].name, line=dict(color = "green")),
            secondary_y=False,
        )
    for i in data["conventional usage"].columns:
        fig.add_trace(
            go.Scatter(x=data["conventional usage"].index, y=data["conventional usage"][i], name=data["conventional usage"][i].name, line=dict(color = "green")),
            secondary_y=False,
        )
    fig.update_layout(font=dict(size=30),
                      title = title,
                            yaxis=dict(
                                tickformat=".n",
                                title="Curtailment in TWh",

                            ),
                            xaxis = dict(
                                title = "year",
                                dtick = 5
                            ))
    fig.update_yaxes(title_text="Electrolyser capacity in GW", range = [50, 130], secondary_y=True)
    fig.write_image(folder+ "KPI_development.pdf", width=2000, height=800)

def radar_chart(maps_folder, data, data2, title=""):
    fig = go.Figure()
    #data2 = data.reindex(columns=data.columns.get_level_values(1).unique(), level=1)
    theta = list(data.columns.get_level_values(0).unique())
    theta.append("curtailment")
    for case in data2.columns.get_level_values(1).unique():
        fig.add_trace(go.Scatterpolar(
            r=[data2[i][case][2040] for i in data2.columns.get_level_values(0).unique()]+[data2["curtailment"][case][2040]],
            theta=theta,
            #fill='toself',
            text = [data[i][case][2040] for i in data.columns.get_level_values(0).unique()],
            name=case
        ))
    # fig.add_trace(go.Scatterpolar(
    #     r=[data2[i]["Offshore"][2040] for i in data2.columns.get_level_values(0).unique()]+[data2["curtailment"]["Offshore"][2040]],
    #     theta=theta,
    #     #fill='toself',
    #     text=[data[i]["Offshore"][2040] for i in data.columns.get_level_values(0).unique()],
    #     name='Offshore'
    # ))
    # fig.add_trace(go.Scatterpolar(
    #     r=[data2[i]["Offshore&Onshore"][2040] for i in data2.columns.get_level_values(0).unique()]+[data2["curtailment"]["Offshore&Onshore"][2040]],
    #     theta=theta,
    #     #fill='toself',
    #     text=[data[i]["Offshore&Onshore"][2040] for i in data.columns.get_level_values(0).unique()],
    #     name='Offshore&Onshore'
    # ))
    # fig.add_trace(go.Scatterpolar(
    #     r=[data2[i]["Stakeholder"][2040] for i in data2.columns.get_level_values(0).unique()]+[data2["curtailment"]["Stakeholder"][2040]],
    #     theta=theta,
    #     text=[data[i]["Stakeholder"][2040] for i in data.columns.get_level_values(0).unique()],
    #     #fill='toself',
    #     name='Stakeholder'
    # ))
    fig.update_traces(
        #mode="lines+markers+text",
        mode="lines+markers",
    )
    fig.update_layout(
        font=dict(size=30),
        title=title,
        polar=dict(
            radialaxis=dict(
                visible=False,
                range=[0, 1.0],

            ),
            bgcolor='rgba(0,0,0,0)'
            ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5,
            ),
        paper_bgcolor='rgba(0,0,0,0)',
        #bgcolor='rgba(0,0,0,0)',
        showlegend=True
    )

    #fig.show()
    fig.write_image(maps_folder + "polar.pdf", width=2000, height=800)