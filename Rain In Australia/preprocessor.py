import pandas as pd
import numpy as np
import re
from geopy.geocoders import Nominatim
from category_encoders.target_encoder import TargetEncoder


class Preprocessor:
    def __init__(self, numerical_impute_strategy, categorical_impute_strategy, roll_cols, roll_strategies, roll_period):
        if len(roll_cols) != len(roll_strategies):
            raise ValueError('Value Error: len(roll_cols) != len(roll_strategies).')
        self.locations = None
        self.numerical_impute_strategy = numerical_impute_strategy
        self.categorical_impute_strategy = categorical_impute_strategy
        self.roll_cols = roll_cols
        self.roll_strategies = roll_strategies
        self.roll_period = roll_period

    def preprocess(self, _data, visualize=False):
        data = _data.copy(deep=True)

        for column in ['RainToday', 'RainTomorrow']:
            data.loc[data[column] == 'Yes', column] = 1
            data.loc[data[column] == 'No', column] = 0

        data['Year'] = pd.DatetimeIndex(data['Date']).year
        data['Month'] = pd.DatetimeIndex(data['Date']).month

        if not visualize:
            self.impute(data)

        data['Location'] = data['Location'].apply(lambda loc: ' '.join(re.findall('[A-Z][^A-Z]+|[A-Z]+', loc)))
        data.loc[data.Location == 'Portland', 'Location'] = 'Portland, Victoria'
        data.loc[data.Location == 'Dartmoor', 'Location'] = 'Dartmoor, Victoria'
        data.loc[data.Location == 'Perth', 'Location'] = 'Perth, Western Australia'
        data.loc[data.Location == 'Richmond', 'Location'] = 'Richmond, New South Wales'
        data['Location'] = data.Location + ', Australia'

        if self.locations is None:
            geolocator = Nominatim(user_agent="rain-in-australia-app")
            locations = {'Location_reduced': [], 'Location': [], 'Address': [], 'Latitude': [], 'Longitude': []}
            for location in data.Location.unique().tolist() + ['Australia']:
                location_enc = geolocator.geocode(location, language='en')
                if location_enc is None:
                    raise ValueError(f'Location not found: {location}')
                locations['Location_reduced'] += [location.split(', ')[0]]
                locations['Location'] += [location]
                locations['Address'] += [location_enc.address]
                locations['Latitude'] += [location_enc.latitude]
                locations['Longitude'] += [location_enc.longitude]
            self.locations = pd.DataFrame(locations)
        data = data.merge(self.locations[['Location', 'Latitude', 'Longitude']], left_on='Location',
                          right_on='Location')

        if visualize:
            data['Location'] = data['Location'].apply(lambda loc: loc.split(', ')[0])

        data = self.RainToday_Locations(data)

        if not visualize:
            data = self.target_encoding(data)

        for i, col in enumerate(self.roll_cols):
            data[f'{col}_{self.roll_period}days'] = Preprocessor.rolling_features_for_all_locations(data[col],
                                                                                                    data.Location,
                                                                                                    period=self.roll_period,
                                                                                                    shift=1,
                                                                                                    strategy=
                                                                                                    self.roll_strategies[
                                                                                                        i])

        return data

    def impute(self, data):
        columns = data.columns[2:-2]
        dtypes = data.dtypes[2:-2]
        numerical = [column for idx, column in enumerate(columns) if dtypes[idx] == float]
        categorical = [column for idx, column in enumerate(columns) if dtypes[idx] == object]
        if self.numerical_impute_strategy == 'mean':
            data[numerical] = data.groupby(['Month', 'Location'])[numerical].transform(lambda x: x.fillna(x.mean()))
        elif self.numerical_impute_strategy == 'median':
            data[numerical] = data.groupby(['Month', 'Location'])[numerical].transform(lambda x: x.fillna(x.median()))
        else:
            raise ValueError('Wrong numerical impute strategy.')

        if self.categorical_impute_strategy == 'mode':
            data[categorical] = data.groupby(['Month', 'Location'])[categorical].transform(lambda x: x.fillna(x.mode()))
        else:
            raise ValueError('Wrong categorical impute strategy.')

    def RainToday_Locations(self, data):
        grouped = data.groupby('Date')[['Location', 'RainToday']].apply(lambda r: r.set_index('Location').T)
        grouped = grouped.reset_index().drop('level_1', axis=1)
        if ',' in grouped.columns[1]:
            grouped.columns = ['Date'] + ['RainToday_' + col[:col.index(',')] for col in grouped.columns[1:]]
        else:
            grouped.columns = ['Date'] + ['RainToday_' + col for col in grouped.columns[1:]]
        return data.merge(grouped, left_on='Date', right_on='Date')

    def target_encoding(self, _data):
        data = _data.copy(deep=True)
        data['Month_Location'] = data['Month']
        location_cols = [column for column in data.columns if 'RainToday_' in column] + ['WindGustDir', 'WindDir9am',
                                                                                         'WindDir3pm', 'Month_Location']

        for location in data.Location.unique():
            location_encoder = TargetEncoder(cols=location_cols, handle_missing=0)
            location_df = data.loc[data.Location == location]
            data.loc[data.Location == location] = location_encoder.fit_transform(location_df, location_df.RainTomorrow)

        location_encoder = TargetEncoder(cols=['Location', 'Month'])
        encoded = location_encoder.fit_transform(data, data.RainTomorrow)

        for c in ['WindGustDir', 'WindDir9am', 'WindDir3pm']:
            encoded[c] = encoded[c].astype(float)
        return encoded

    @staticmethod
    def rolling_features_for_all_locations(series_, locations, period, shift, strategy):

        def rolling_features_for_location(series_, period, shift, strategy):
            first_n = []
            for i in range(period):
                if i == 0:
                    first_n += [series_[i]]
                else:
                    if strategy == 'mean':
                        first_n += [series_[:i + 1 - shift].mean()]
                    elif strategy == 'median':
                        first_n += [series_[:i + 1 - shift].median()]
                    elif strategy == 'sum':
                        first_n += [series_[:i + 1 - shift].sum()]
                    else:
                        raise ValueError('Wrong strategy.')

            if strategy == 'mean':
                new_series = series_.shift(shift).rolling(period).mean()
            elif strategy == 'median':
                new_series = series_.shift(shift).rolling(period).median()
            elif strategy == 'sum':
                new_series = series_.shift(shift).rolling(period).sum()
            else:
                raise ValueError('Wrong strategy.')

            new_series[:period] = first_n
            return new_series

        all_locations = []
        for location in locations.unique():
            one_location, index = series_[locations == location].reset_index(drop=True, inplace=False), series_[
                locations == location].index.to_series()
            new_series_for_location = rolling_features_for_location(one_location, period, shift, strategy)
            new_series_for_location.index = index
            all_locations += [new_series_for_location]
        return pd.concat(all_locations, axis=0)

    @staticmethod
    def load_and_split():
        dataset = pd.read_csv('data/weatherAUS.csv')
        dataset.sort_values(['Date', 'Location'], inplace=True)
        dataset.reset_index(drop=True, inplace=True)
        train, test = dataset.iloc[:109103], dataset.iloc[109103:]
        return train, test
