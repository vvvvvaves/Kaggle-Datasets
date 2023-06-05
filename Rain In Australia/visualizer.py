import os
import pickle
import pandas as pd
import plotly.express as px

import predictor
from preprocessor import *
from predictor import Predictor
import plotly.graph_objects as go

standard_roll_cols = ['MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine',
                      'WindGustSpeed', 'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am',
                      'Humidity3pm', 'Pressure9am', 'Pressure3pm', 'Cloud9am',
                      'Cloud3pm', 'Temp9am', 'Temp3pm', 'RainToday']
standard_roll_strategies = ['mean'] * (len(standard_roll_cols) - 1) + ['sum']
standard_roll_period = 7


class Visualizer:
    def __init__(self):
        self.preprocessor = Preprocessor(numerical_impute_strategy='mean',
                                         categorical_impute_strategy='mode',
                                         roll_cols=standard_roll_cols,
                                         roll_period=standard_roll_period,
                                         roll_strategies=standard_roll_strategies)
        self.predictor = Predictor()
        self.data = self.preprocess_data()
        with open('feature_importances.pickle', 'rb') as handle:
            self.feature_importances = pickle.load(handle)

    def preprocess_data(self):
        if os.path.exists('visualizer_train_data.pickle') and os.path.exists('locations.pickle'):
            with open('visualizer_train_data.pickle', 'rb') as handle:
                preprocessed = pickle.load(handle)

            with open('locations.pickle', 'rb') as handle:
                self.preprocessor.locations = pickle.load(handle)

        else:
            train_data, test_data = self.preprocessor.load_and_split()
            preprocessed = self.preprocessor.preprocess(_data=train_data,
                                                        visualize=True,
                                                        train=True)

            with open('visualizer_train_data.pickle', 'wb') as handle:
                pickle.dump(preprocessed, handle, protocol=pickle.HIGHEST_PROTOCOL)

            with open('visualizer_raw_test_data.pickle', 'wb') as handle:
                pickle.dump(test_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

            with open('locations.pickle', 'wb') as handle:
                pickle.dump(self.preprocessor.locations, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return preprocessed

    def australia_map(self):
        total_rains = self.data.groupby('Location')['RainTomorrow'].sum()
        total_rains.index.name = 'index'
        locations = self.preprocessor.locations.copy(deep=True)
        locations = locations.merge(total_rains.rename('Total_rains').astype(int).reset_index(),
                                    left_on='Location_reduced',
                                    right_on='index').drop('index', axis=1)
        fig = px.scatter_mapbox(locations,
                                lon=locations['Longitude'],
                                lat=locations['Latitude'],
                                zoom=3,
                                # color=,
                                size=locations['Total_rains'].values,
                                width=450 * 1.8 - 10,
                                height=300 * 1.8 + 50)
        Australia_latitude, Australia_longitude = -24.7761086, 134.755
        fig.update_layout(
            mapbox_style="white-bg",
            mapbox_layers=[
                {
                    "below": 'traces',
                    "sourcetype": "raster",
                    "sourceattribution": "United States Geological Survey",
                    "source": [
                        "https://basemap.nationalmap.gov/arcgis/rest/services/USGSImageryOnly/MapServer/tile/{z}/{y}/{x}"
                    ]
                }
            ])
        fig.update_layout(mapbox={'center': go.layout.mapbox.Center(lat=Australia_latitude, lon=Australia_longitude)})
        fig.update_layout(margin={"r": 20, "t": 20, "l": 20, "b": 20})
        return fig

    def winds_location(self, location, rain=1):
        # 1. Map is centered around chosen location.
        # 2. The size of the circle is the amount of rains in that location.
        # 3. The color of each circle represents correlation between rainTomorrow at the chosen location and each location's RainToday.
        locations = self.preprocessor.locations.copy(deep=True)

        selected_columns = [column for column in self.data.columns if 'RainToday_' in column]
        total_rains = self.data.groupby('Location')['RainTomorrow'].sum()
        total_rains.index.name = 'index'
        locations = locations.merge(total_rains.rename('Total_rains').astype(int).reset_index(),
                                    left_on='Location_reduced',
                                    right_on='index').drop('index', axis=1)
        relative_raintoday = self.data.loc[
            (self.data.RainTomorrow == 1) & (self.data.Location == location), selected_columns].sum(axis=0)
        relative_raintoday.index = relative_raintoday.index.to_series().apply(lambda txt: txt[10:])
        locations = locations.merge(relative_raintoday.rename('relative_RainToday').astype(int).reset_index(),
                                    left_on='Location_reduced', right_on='index').drop('index', axis=1)

        fig = px.scatter_mapbox(locations,
                                lon=locations['Longitude'],
                                lat=locations['Latitude'],
                                zoom=7,
                                color=locations['relative_RainToday'],
                                size=locations['Total_rains'].values,
                                width=450 * 1.5,
                                height=300 * 1.5 + 50)
        location_latitude = locations.loc[locations.Location_reduced == location, 'Latitude'].to_list()[0]
        location_longitude = locations.loc[locations.Location_reduced == location, 'Longitude'].to_list()[0]
        fig.update_layout(mapbox_style="open-street-map",
                          mapbox={'center': go.layout.mapbox.Center(lat=location_latitude, lon=location_longitude)})
        fig.update_layout(margin={"r": 20, "t": 20, "l": 20, "b": 20})

        winds = self.data.loc[(self.data.RainTomorrow == rain) & (
                    self.data.Location == location), 'WindGustDir'].value_counts().sort_values(
            ascending=False).reset_index()
        winds.columns = ['Wind Gust Directions', 'Amount of rain']
        bar = px.bar(winds, x='Wind Gust Directions', y='Amount of rain', orientation='v')
        bar.update_layout(margin={"r": 20, "t": 20, "l": 20, "b": 20})
        return bar, fig

    def violinplot(self, column, overlay, box, points):
        violin_data = self.data[[column, 'RainTomorrow']].dropna().sample(n=500, random_state=42)
        if overlay:
            x = None
        else:
            x = 'RainTomorrow'

        if points:
            fig = px.violin(violin_data, y=column, box=box, x=x,
                            color='RainTomorrow', points='all',
                            violinmode='overlay')
        else:
            fig = px.violin(violin_data, y=column, x=x,
                            box=box, color='RainTomorrow', violinmode='overlay')
        fig.update_layout(margin={"r": 20, "t": 20, "l": 20, "b": 20})
        return fig

    def rain_months(self, location):
        if location == 'All':
            months = self.data.loc[self.data.RainTomorrow == 1, 'Month'].value_counts().sort_index(ascending=True).reset_index()
        else:
            months = self.data.loc[(self.data.RainTomorrow == 1) & (self.data.Location == location), 'Month'].value_counts().sort_index(ascending=True).reset_index()
        months.columns = ['Month', 'Amount of rain']
        bar = px.bar(months, x='Month', y='Amount of rain', orientation='v')
        bar.update_layout(margin={"r": 20, "t": 30, "l": 20, "b": 20})
        return bar

    def raintoday_roll(self, location):
        if location == 'All':
            filter_ = True
        else:
            filter_ = self.data.Location == location
        rain = self.data[f'RainToday_{self.preprocessor.roll_period}days'].loc[(self.data.RainTomorrow == 1) & filter_].value_counts(normalize=True).sort_index().reset_index()
        no_rain = self.data[f'RainToday_{self.preprocessor.roll_period}days'].loc[(self.data.RainTomorrow == 0) & filter_].value_counts(normalize=True).sort_index().reset_index()
        rain['RainTomorrow'] = 'Yes'
        no_rain['RainTomorrow'] = 'No'
        bar_ = pd.concat([rain, no_rain], axis=0)
        bar_.columns = ['Weekdays', 'number of days (normalized)', 'RainTomorrow']
        bar = px.bar(bar_, x="Weekdays", y="number of days (normalized)",
                     color="RainTomorrow", orientation='v', barmode="group",
                     title='Number of days raining before target date.')
        bar.update_layout(margin={"r": 20, "t": 30, "l": 20, "b": 20})
        return bar

    def windspeed(self, location):
        rain = self.data.loc[(self.data.RainTomorrow == 1) & (self.data.Location == location)][['WindGustSpeed', 'Month']].groupby('Month').median().reset_index()
        no_rain = self.data.loc[(self.data.RainTomorrow == 0) & (self.data.Location == location)][['WindGustSpeed', 'Month']].groupby('Month').median().reset_index()
        rain['RainTomorrow'] = 'Yes'
        no_rain['RainTomorrow'] = 'No'
        linechart_ = pd.concat([rain, no_rain], axis=0)
        linechart_.columns = ['Month', 'WindGustSpeed', 'RainTomorrow']
        line = px.line(linechart_, x='Month', y='WindGustSpeed', color='RainTomorrow')
        line.update_layout(margin={"r": 20, "t": 20, "l": 20, "b": 20})
        line.update_layout({'height': 275})
        return line

    def confusion_heatmap(self, conf_m):
        fig = px.imshow(conf_m, text_auto=True)
        fig.update_layout(margin={"r": 20, "t": 20, "l": 20, "b": 20})
        return fig

    def metrics_barplot(self, threshold):
        auc_score, conf_m, accuracy, recall, precision, f1_score = self.predictor.model_presentation(threshold=threshold)
        bar = px.bar(x=[auc_score, accuracy, precision, recall, f1_score], y=["ROC AUC", "Accuracy", "Precision", "Recall", "F1-score"], orientation='h')
        bar.update_layout(margin={"r": 20, "t": 40, "l": 20, "b": 20})
        bar.update_layout(xaxis_title="", yaxis_title="")
        return bar

    def shap_barplot(self):
        importances = self.feature_importances.sort_values('importance', ascending=False).iloc[:30]
        bar = px.bar(importances, x='importance', y='feature', orientation='h')
        bar.update_layout(margin={"r": 80, "t": 20, "l": 20, "b": 20})
        bar.update_layout(xaxis_title="", yaxis_title="")
        return bar