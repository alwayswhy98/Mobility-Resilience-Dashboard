#######################
# Import libraries
import streamlit as st
import pandas as pd
import altair as alt
import plotly.express as px


#######################
# Page configuration
st.set_page_config(
    page_title="Mobility Resilience Dashboard",
    layout="wide",
    initial_sidebar_state="expanded")

alt.themes.enable("dark")


#######################
# Load data
cleaned_df = pd.read_csv('Cleaned Data.csv')
country_df = pd.read_excel("Country Information.xlsx")
resilience_df = pd.read_csv("resilience.csv")

#######################
# Sidebar
with st.sidebar:
    st.title('Mobility Resilience Dashboard')
    
    country_list = list(cleaned_df["country"].unique())
    country_list.insert(0, "Summary")

    selected_country = st.selectbox('Select a country', country_list)
    df_selected_country = cleaned_df[cleaned_df.country == selected_country]

    year_list = [2020, 2021, 2022, 2023]
    selected_year = st.selectbox('Select a year', year_list)

#######################
# CSS Code
st.markdown(
    """
<style>
[data-testid="stMetricValue"] {
    font-size: 30px;
}
</style>
""",
    unsafe_allow_html=True,
)

#######################
# Plot
# Mobility VS Covid-19 Cases
def mobility_covid_chart(df):
    import plotly.graph_objects as go

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=df["date"], y=df["driving"], name='Mobility'))
    fig.add_trace(go.Scatter(x=df["date"], y=df["New COVID-19 Case"], name='COVID-19 Case', yaxis='y2'))

    fig.update_layout(
        yaxis=dict(title='7-days SMA Mobility'),
        yaxis2=dict(title='COVID-19 Cases', overlaying='y', side='right'),
        xaxis_title='Date',
        title='Mobility VS Covid-19',
        legend=dict(x=0, y=1, traceorder="normal", font=dict(size=8)),
        showlegend=True
    ) 

    return fig


# Raw Mobility Data
def raw_mobility_data_chart(df):
    import plotly.graph_objects as go

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['date'], y=df['driving'], name='driving'))
    fig.add_trace(go.Scatter(x=df['date'], y=df['7-day SMA'], name='7-days SMA'))
    
    fig.update_layout(
        xaxis_title='Date',
        yaxis_title='Driving',
        title='Raw Mobility Data',
        legend=dict(x=0, y=1, traceorder="normal", font=dict(size=8))
    )
    
    fig.update_xaxes(tickfont=dict(size=8))
    
    return fig

def calculate_extreme_event(df):
    start_date_list = []
    end_date_list = []
    cnt = 0
    flag = False

    for idx in range(len(df)):
        if df["q_t"][idx] <= 0.7 and flag == False:
            flag = True
            cnt = cnt + 1
            tmp_start_date = df["date"][idx]
            
        elif df["q_t"][idx] <= 0.7 and flag == True:
            flag = True
            cnt = cnt + 1
            
        elif cnt >= 7:
            tmp_end_date = df["date"][idx]

            if tmp_start_date not in start_date_list and tmp_end_date not in end_date_list:
                start_date_list.append(tmp_start_date)
                end_date_list.append(tmp_end_date)
                flag = False
                cnt = 0
        else:
            flag = False
            cnt = 0

    return start_date_list, end_date_list

def extreme_event_detection_plot(df):
    import plotly.graph_objects as go

    fig = go.Figure()

    # Add line plots
    fig.add_trace(go.Scatter(x=df['date'], y=df['7-day SMA'], name='7-days SMA'))

    # Add horizontal lines
    standard_condition = df["7-day SMA"].values[:14].mean()
    fig.add_trace(go.Scatter(x=df['date'], y=[standard_condition]*len(df), name='Standard Condition', line=dict(color='black')))
    fig.add_trace(go.Scatter(x=df['date'], y=[standard_condition*0.7]*len(df), name='Low Threshold(0.7)', line=dict(color='red', dash='dash')))

    # Add vertical lines
    start_date_list, end_date_list = calculate_extreme_event(df)
    
    if len(start_date_list) != len(end_date_list):
        end_date_list.append(max(df['date']))

    for idx in range(len(start_date_list)):
        fig.add_shape(type="line", x0=start_date_list[idx], y0=df['driving'].min(), x1=start_date_list[idx], y1=df['driving'].max(), line=dict(color="rgba(128, 128, 128, 0.3)", width=1, dash="dash"))
        fig.add_shape(type="line", x0=end_date_list[idx], y0=df['driving'].min(), x1=end_date_list[idx], y1=df['driving'].max(), line=dict(color="rgba(128, 128, 128, 0.3)", width=1, dash="dash"))
        # Add shaded area between vertical lines
        fig.add_shape(type="rect", x0=start_date_list[idx], y0=df['driving'].min(), x1=end_date_list[idx], y1=df['driving'].max(), fillcolor="rgba(128, 128, 128, 0.3)", line=dict(width=0))

    fig.update_layout(
        xaxis_title='Date',
        yaxis_title='Driving',
        title='Extreme Event Detection',
        legend=dict(x=0, y=1, traceorder="normal", font=dict(size=8)),
        showlegend=True
    )

    fig.update_xaxes(tickfont=dict(size=8))

    return fig


def resilience_plot(df):
    import plotly.graph_objects as go

    fig = go.Figure()

    # Add line plots
    fig.add_trace(go.Scatter(x=df['date'], y=df['q_t'], name='Q(t)'))

    # Add horizontal lines
    fig.add_trace(go.Scatter(x=df['date'], y=[1]*len(df), name='Standard Condition', line=dict(color='black')))

    # Add vertical lines
    start_date_list, end_date_list = calculate_extreme_event(df)
    
    if len(start_date_list) != len(end_date_list):
        end_date_list.append(max(df['date']))

    for idx in range(len(start_date_list)):
        start_val_idx = list(df["date"].values).index(start_date_list[idx])
        end_val_idx = list(df["date"].values).index(end_date_list[idx])

        for val_idx in range(start_val_idx, end_val_idx):
            fig.add_shape(type="rect", x0=df["date"][val_idx], y0=0, x1=df["date"][val_idx+1], y1=df['q_t'][val_idx], fillcolor="rgba(128, 128, 128, 0.3)", line=dict(width=0))

    fig.update_layout(
        xaxis_title='Date',
        yaxis_title='Ratio to Standard Condition',
        title='Resilience (Area Under the Curve)',
        legend=dict(x=0, y=1, traceorder="normal", font=dict(size=8)),
        showlegend=True
    )

    fig.update_xaxes(tickfont=dict(size=8))

    return fig


def country_resilience_heatmap(df):
    import plotly.express as px

    fig = px.choropleth(resilience_df, locations='Country', locationmode='country names',
                    color='Resilience Loss', 
                    #range_color=(0, max(resilience_df["Resilience Loss"])), 
                    range_color=(0, 100),
                    title='Resilience Loss per Country Heatmap',
                    color_continuous_scale='Blues')

    fig.update_layout(
        #title=dict(x=0.5, y=0.95, xanchor='center', yanchor='top'),
        font=dict(size=12),
        plot_bgcolor='rgb(255, 255, 255)',
        margin=dict(l=0, r=0, t=30, b=0),
        geo=dict(showframe=False, showcoastlines=False)
    )

    return fig

def make_heatmap(input_df, input_y, input_x, input_color):    
    heatmap = alt.Chart(input_df).mark_rect().encode(
            y=alt.Y(f'{input_y}:O', axis=alt.Axis(title="Year", titleFontSize=18, titlePadding=15, titleFontWeight=900, labelAngle=0)),
            x=alt.X(f'{input_x}:O', axis=alt.Axis(title="", titleFontSize=18, titlePadding=15, titleFontWeight=900)),
            color=alt.Color(f'max({input_color}):Q',
                             legend=None),
            stroke=alt.value('black'),
            strokeWidth=alt.value(0.25),
        ).properties(width=900
        ).configure_axis(
        labelFontSize=12,
        titleFontSize=12
        ) 
    # height=300
    return heatmap


#######################
# Dashboard Main Panel
# Summary
if selected_country == "Summary":
    col = st.columns((2, 5, 2.5), gap='medium')

    with col[0]:
        with st.container(height=70, border= False):
            st.markdown('#### Summary')

        ######################
        # Population in the World
        population_state_name = 'World Population in ' + str(selected_year)
        raw_population = country_df[country_df["year"]==selected_year]["population"].sum()
        prev_population = country_df[country_df["year"]==selected_year-1]["population"].sum()
        format_world_population = format(raw_population, ",")
        total_case_chage = round(raw_population / prev_population * 100 - 100, 2)
        state_delta = str(total_case_chage) + "%"
        st.metric(label=population_state_name, value=format_world_population, delta=state_delta)

        ######################
        # COVID-19 Cases in the World
        covid_case_state_name = 'COVID-19 Cases in ' + str(selected_year)
        raw_total_case = country_df[country_df["year"]==selected_year]["Total Cases"].sum()
        prev_total_case = country_df[country_df["year"]==selected_year-1]["Total Cases"].sum()
        format_total_case = format(raw_total_case, ",")
        if selected_year == 2020:
            state_delta = "NA"
        else:
            total_case_chage = round(raw_total_case / prev_total_case * 100 - 100, 2)
            state_delta = str(total_case_chage) + "%"
        st.metric(label=covid_case_state_name, value=format_total_case, delta=state_delta)

        ######################
        # COVID-19 Death in the World
        third_state_name = 'Total COVID-19 Deaths ' + str(selected_year)
        raw_total_death = country_df[country_df["year"]==selected_year]["Total Deaths"].sum()
        prev_total_death = country_df[country_df["year"]==selected_year-1]["Total Deaths"].sum()
        format_total_death = format(raw_total_death, ",")
        if selected_year == 2020:
            state_delta = "NA"
        else:
            total_death_chage = round(raw_total_death / prev_total_death * 100 - 100, 2)
            state_delta = str(total_death_chage) + "%"
        st.metric(label=third_state_name, value=format_total_death, delta=state_delta)

    with col[1]:
        with st.container(height=70, border= False):
            st.markdown('#### ')

        choropleth = country_resilience_heatmap(resilience_df)
        st.plotly_chart(choropleth, use_container_width=True)

        heatmap = make_heatmap(country_df, 'year', 'Country', 'Total Cases')
        st.altair_chart(heatmap, use_container_width=True)

    with col[2]:
        ######################
        # First wave resilience dataframe
        st.markdown('#### Resilience: COVID-19 First Wave')

        st.dataframe(resilience_df,
                    column_order=("Country", "Resilience", "Resilience Loss"),
                    hide_index=True,
                    width=None,
                    column_config={
                        "Country": st.column_config.TextColumn(
                            "Country",
                        ),
                        "Resilience": st.column_config.ProgressColumn(
                            "Resilience",
                            format="%.2f",
                            min_value=0,
                            max_value=max(resilience_df.Resilience),
                        ),
                        "Resilience Loss": st.column_config.ProgressColumn(
                            "Resilience Loss",
                            format="%.2f",
                            min_value=0,
                            max_value=max(resilience_df["Resilience Loss"]),
                        )}
                    )
        
        ######################
        # About
        with st.expander('About', expanded=True):
            st.write('''
                - Mobility Data: [Apple Mobility Trends Report](https://covid19.apple.com/mobility)
                - Covid Data: [Our World in Data](https://ourworldindata.org/coronavirus)
                - Country Data: [World Population Review](https://worldpopulationreview.com/)
                - :orange[**HDI**]: Human Development Index
                - :orange[**Population, Total COVID-19 Cases, Total COVID-19 Deaths**]: As Of December 31
                - :orange[**GDP, GDP per Capita, HDI**]: On the selected year
                ''')

# Selected country
else:
    col = st.columns((1, 1, 3, 3, 2.5), gap='medium')

    with col[0]:
        ######################
        # Container
        if selected_country == "United Arab Emirates" or selected_country == "United Kingdom":
            container_height = 150
        else:
            container_height = 100

        with st.container(height=container_height, border= False):
            st.markdown('#### ' + selected_country)
            st.markdown('##### Information')
        
        ######################
        # Resilience
        resilience_state_name = 'Resilience'
        resilience = str(round(resilience_df[resilience_df["Country"]==selected_country]["Resilience"].values[0], 1)) + " days"
        st.metric(label=resilience_state_name, value=resilience)

        ######################
        # Population
        df_selected_country = country_df[(country_df["Country"] == selected_country)]

        first_state_name = 'Population'
        raw_population = df_selected_country[df_selected_country["year"]==selected_year]["population"].values[0]
        prev_population = df_selected_country[df_selected_country["year"]==selected_year-1]["population"].values[0]
        population_chage = round(raw_population / prev_population * 100 - 100, 2)
        format_population = str(round(raw_population / 1000000, 2)) + "M"
        first_state_delta = str(population_chage) + "%"
        st.metric(label=first_state_name, value=format_population, delta=first_state_delta)

        ######################
        # GDP
        second_state_name = 'GDP'
        raw_gpd = df_selected_country[df_selected_country["year"]==selected_year]["GDP"].values[0]
        prev_gpd = df_selected_country[df_selected_country["year"]==selected_year-1]["GDP"].values[0]
        
        if selected_year == 2023:
            format_gdp = "NA"
            second_state_delta = "NA"
        else:
            format_gdp = "USD " + str(round(raw_gpd / 1000000000000, 2)) + "T"
            gpd_chage = round(raw_gpd / prev_gpd * 100 - 100, 2)
            second_state_delta = str(gpd_chage) + "%"
        st.metric(label=second_state_name, value=format_gdp, delta=second_state_delta)

        ######################
        # COVID-19 Cases
        third_state_name = 'Total COVID-19 Cases'
        raw_total_case = df_selected_country[df_selected_country["year"]==selected_year]["Total Cases"].values[0]
        prev_total_case = df_selected_country[df_selected_country["year"]==selected_year-1]["Total Cases"].values[0]
        format_total_case = format(raw_total_case, ",")
        if selected_year == 2020:
            third_state_delta = "NA"
        else:
            total_case_chage = round(raw_total_case / prev_total_case * 100 - 100, 2)
            third_state_delta = str(total_case_chage) + "%"
        st.metric(label=third_state_name, value=format_total_case, delta=third_state_delta)


    with col[1]:
        ######################
        # Container
        if selected_country == "United Arab Emirates" or selected_country == "United Kingdom":
            container_height = 150
        else:
            container_height = 100

        with st.container(height=container_height, border=False):
            st.markdown('#### ')

        df_selected_country = country_df[(country_df["Country"] == selected_country)]

        ######################
        # Resilience Loss
        resilience_loss_state_name = 'Resilience Loss'
        resilience_loss = str(round(resilience_df[resilience_df["Country"]==selected_country]["Resilience Loss"].values[0], 1)) + " days"
        st.metric(label=resilience_loss_state_name, value=resilience_loss)

        ######################
        # Human Development Index
        first_state_name = 'HDI'
        raw_hdi = df_selected_country[df_selected_country["year"]==selected_year]["Human Development Index"].values[0]
        prev_hdi = df_selected_country[df_selected_country["year"]==selected_year-1]["Human Development Index"].values[0]
        if selected_year == 2023:
            format_hdi = "NA"
            first_state_delta = "NA"
        else:
            format_hdi = raw_hdi
            hdi_chage = round(raw_hdi / prev_hdi * 100 - 100, 2)
            first_state_delta = str(hdi_chage) + "%"
        st.metric(label=first_state_name, value=format_hdi, delta=first_state_delta)

        ######################
        # GDP per Capita
        second_state_name = 'GDP per Capita'
        raw_gpd_cap = df_selected_country[df_selected_country["year"]==selected_year]["GDP per Capita"].values[0]
        prev_gpd_cap = df_selected_country[df_selected_country["year"]==selected_year-1]["GDP per Capita"].values[0]
        if selected_year == 2023:
            format_gdp_cap = "NA"
            second_state_delta = "NA"
        else:
            format_gdp_cap = "USD " + str(round(raw_gpd_cap / 1000, 2)) + "k"
            gpd_cap_chage = round(raw_gpd_cap / prev_gpd_cap * 100 - 100, 2)
            second_state_delta = str(gpd_cap_chage) + "%"
        st.metric(label=second_state_name, value=format_gdp_cap, delta=second_state_delta)

        ######################
        # COVID-19 Deaths
        third_state_name = 'Total COVID-19 Deaths'
        raw_total_death = df_selected_country[df_selected_country["year"]==selected_year]["Total Deaths"].values[0]
        prev_total_death = df_selected_country[df_selected_country["year"]==selected_year-1]["Total Deaths"].values[0]
        format_total_case = format(raw_total_death, ",")
        if selected_year == 2020:
            third_state_delta = "NA"
        else:
            total_death_chage = round(raw_total_death / prev_total_death * 100 - 100, 2)
            third_state_delta = str(total_death_chage) + "%"
        st.metric(label=third_state_name, value=format_total_case, delta=third_state_delta)


    with col[2]:
        df_selected_country = cleaned_df[cleaned_df.country == selected_country].reset_index(drop=True)

        ######################
        # Mobility vs COVID-19
        mobility_covid_fig = mobility_covid_chart(df_selected_country)
        st.plotly_chart(mobility_covid_fig, use_container_width=True)

        ######################
        # Extreme Event Detection
        extreme_event_fig = extreme_event_detection_plot(df_selected_country)
        st.plotly_chart(extreme_event_fig, use_container_width=True)


    with col[3]:
        df_selected_country = cleaned_df[cleaned_df.country == selected_country].reset_index(drop=True)

        ######################
        # Raw Mobility Data
        raw_mobility_data_fig = raw_mobility_data_chart(df_selected_country)
        st.plotly_chart(raw_mobility_data_fig, use_container_width=True)

        ######################
        # Resilience Calculation
        resilience_fig = resilience_plot(df_selected_country)
        st.plotly_chart(resilience_fig, use_container_width=True)


    with col[4]:
        ######################
        # First wave resilience dataframe
        st.markdown('#### Resilience: COVID-19 First Wave')

        st.dataframe(resilience_df,
                    column_order=("Country", "Resilience", "Resilience Loss"),
                    hide_index=True,
                    width=None,
                    column_config={
                        "Country": st.column_config.TextColumn(
                            "Country",
                        ),
                        "Resilience": st.column_config.ProgressColumn(
                            "Resilience",
                            format="%.2f",
                            min_value=0,
                            max_value=max(resilience_df.Resilience),
                        ),
                        "Resilience Loss": st.column_config.ProgressColumn(
                            "Resilience Loss",
                            format="%.2f",
                            min_value=0,
                            max_value=max(resilience_df["Resilience Loss"]),
                        )}
                    )
        
        ######################
        # About
        with st.expander('About', expanded=True):
            st.write('''
                - Mobility Data: [Apple Mobility Trends Report](https://covid19.apple.com/mobility)
                - Covid Data: [Our World in Data](https://ourworldindata.org/coronavirus)
                - Country Data: [World Population Review](https://worldpopulationreview.com/)
                - :orange[**HDI**]: Human Development Index
                - :orange[**Population, Total COVID-19 Cases, Total COVID-19 Deaths**]: As Of December 31
                - :orange[**GDP, GDP per Capita, HDI**]: On the selected year
                ''')