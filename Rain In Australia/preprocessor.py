import pandas as pd
import numpy as np
import re
from geopy.geocoders import Nominatim
from category_encoders.target_encoder import TargetEncoder

standard_roll_cols = ['MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine',
                      'WindGustSpeed', 'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am',
                      'Humidity3pm', 'Pressure9am', 'Pressure3pm', 'Cloud9am',
                      'Cloud3pm', 'Temp9am', 'Temp3pm', 'RainToday']
standard_roll_strategies = ['mean'] * (len(standard_roll_cols) - 1) + ['sum']
standard_roll_period = 7


class OutlierDetection:
    def __init__(self, threshold=3):
        self.threshold = threshold
        self.numerical = None
        self.info = None

    def fit(self, X):
        self.numerical = [column for column in X.columns if X.dtypes[column] == float]
        self.info = {}
        for col in self.numerical:
            series_ = X[col].dropna()
            mean = np.mean(series_)
            std = np.std(series_)
            self.info[col] = {'mean': mean, 'std': std}
        return self

    def transform(self, X):
        outliers = {}
        for col in self.numerical:
            series_ = X[col].dropna()
            outliers_ = series_.apply(lambda x: (x - self.info[col]['mean']) / self.info[col]['std'] > self.threshold)
            outliers[col] = series_[outliers_].index.values
        return outliers

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

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
        self.detector = None
        self.num_impute_data = None
        self.cat_impute_data = None
        self.encoders = {}

    def preprocess(self, _data, visualize=False, train=True):
        data = _data.copy(deep=True)

        if train:
            self.detector = OutlierDetection()
            self.detector.fit(data)

        outliers = self.detector.transform(data)
        outlier_indeces = set()
        for k in outliers.keys():
            outlier_indeces.update(outliers[k])

        data['is_outlier'] = 0
        data.loc[list(outlier_indeces), 'is_outlier'] = 1

        for column in ['RainToday', 'RainTomorrow']:
            data.loc[data[column] == 'Yes', column] = 1
            data.loc[data[column] == 'No', column] = 0

        data['Year'] = pd.DatetimeIndex(data['Date']).year
        data['Month'] = pd.DatetimeIndex(data['Date']).month

        data['Location'] = data['Location'].apply(lambda loc: ' '.join(re.findall('[A-Z][^A-Z]+|[A-Z]+', loc)))
        data.loc[data.Location == 'Portland', 'Location'] = 'Portland, Victoria'
        data.loc[data.Location == 'Dartmoor', 'Location'] = 'Dartmoor, Victoria'
        data.loc[data.Location == 'Perth', 'Location'] = 'Perth, Western Australia'
        data.loc[data.Location == 'Richmond', 'Location'] = 'Richmond, New South Wales'
        data['Location'] = data.Location + ', Australia'

        if train:
            geolocator = Nominatim(user_agent="rain-in-australia-app")
            locations = {'Location_reduced': [], 'Location': [], 'Address': [], 'Latitude': [], 'Longitude': []}
            for location in data.Location.unique().tolist():
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

        data['Location'] = data['Location'].apply(lambda loc: loc.split(', ')[0])

        if not visualize:
            self.impute(data, train=train)

        data['RainToday'] = data['RainToday'].astype(float)

        data = self.RainToday_Locations(data)

        for i, col in enumerate(self.roll_cols):
            data[f'{col}_{self.roll_period}days'] = Preprocessor.rolling_features_for_all_locations(data[col],
                                                                                                    data.Location,
                                                                                                    period=self.roll_period,
                                                                                                    shift=1,
                                                                                                    strategy=
                                                                                                    self.roll_strategies[
                                                                                                        i],
                                                                                                    corr=False)
            data[f'{col}_{self.roll_period}days_corr'] = Preprocessor.rolling_features_for_all_locations(data[col],
                                                                                                         data.Location,
                                                                                                         period=4,
                                                                                                         shift=0,
                                                                                                         strategy=
                                                                                                         self.roll_strategies[
                                                                                                             i],
                                                                                                         corr=True)

        #         data = self.generate_correlations(data)

        if not visualize:
            data = self.target_encoding(data, train=train)

        acc_col = [column for column in data.columns if
                   'RainToday_' in column and 'days' not in column or column == 'Month_Location']
        data['Accumulated_probabilities'] = data.loc[:, acc_col].sum(axis=1)

        data.drop('Date', axis=1, inplace=True)
        return data

    def generate_correlations(self, _data):
        data = _data.copy(deep=True)
        corrs = [column for column in data.columns if '_corr' in column]
        grouped_ = data.groupby(['Date'])[['Location'] + corrs].apply(lambda r: r.set_index('Location').T)
        grouped_.index.names = ['Date', 'Features']
        grouped_ = grouped_.unstack(level='Features')
        grouped_.columns = ['_'.join(tupl) for tupl in grouped_.columns.values]
        grouped_.reset_index(inplace=True)
        data = data.merge(grouped_, left_on='Date', right_on='Date')
        return data

    def impute(self, data, train):
        columns = data.columns[2:-2]
        dtypes = data.dtypes[2:-2]
        numerical = [column for idx, column in enumerate(columns) if dtypes[idx] == float]
        categorical = [column for idx, column in enumerate(columns) if dtypes[idx] == object]
        if self.numerical_impute_strategy == 'mean' and train:
            self.num_impute_data = data.groupby(['Month', 'Location'])[numerical].mean().reset_index()
        elif self.numerical_impute_strategy == 'median' and train:
            self.num_impute_data = data.groupby(['Month', 'Location'])[numerical].median().reset_index()
        elif train:
            raise ValueError('Wrong numerical impute strategy.')

        if self.categorical_impute_strategy == 'mode' and train:
            self.cat_impute_data = data.groupby(['Month', 'Location'])[categorical].apply(
                lambda x: x.mode()).reset_index()
            self.cat_impute_data = self.cat_impute_data.loc[self.cat_impute_data['level_2'] == 0]
            self.cat_impute_data.drop('level_2', axis=1, inplace=True)
        elif train:
            raise ValueError('Wrong categorical impute strategy.')

        for location in data.Location.unique():
            for month in data.Month.unique():
                locf, monthf = (data.Location == location), (data.Month == month)
                for ccol in categorical:
                    nanf = (data[ccol].isna())
                    ilocf, imonthf = (self.cat_impute_data['Month'] == month), (
                                self.cat_impute_data['Location'] == location)
                    data.loc[locf & monthf & nanf, ccol] = \
                    self.cat_impute_data.loc[ilocf & imonthf, ccol].reset_index(drop=True)[0]

                for ncol in numerical:
                    nanf = (data[ncol].isna())
                    ilocf, imonthf = (self.num_impute_data['Month'] == month), (
                                self.num_impute_data['Location'] == location)
                    data.loc[locf & monthf & nanf, ncol] = \
                    self.num_impute_data.loc[ilocf & imonthf, ncol].reset_index(drop=True)[0]

        return data

    def RainToday_Locations(self, data):
        grouped = data.groupby('Date')[['Location', 'RainToday']].apply(lambda r: r.set_index('Location').T)
        grouped = grouped.reset_index().drop('level_1', axis=1)
        if ',' in grouped.columns[1]:
            grouped.columns = ['Date'] + ['RainToday_' + col[:col.index(',')] for col in grouped.columns[1:]]
        else:
            grouped.columns = ['Date'] + ['RainToday_' + col for col in grouped.columns[1:]]
        return data.merge(grouped, left_on='Date', right_on='Date')

    def target_encoding(self, _data, train):
        data = _data.copy(deep=True)
        data['Month_Location'] = data['Month']
        location_cols = [column for column in data.columns if 'RainToday_' in column] + ['WindGustDir', 'WindDir9am',
                                                                                         'WindDir3pm', 'Month_Location']

        for location in data.Location.unique():
            location_df = data.loc[data.Location == location]

            if train:
                self.encoders[location] = TargetEncoder(cols=location_cols, handle_missing=0)
                data.loc[data.Location == location] = self.encoders[location].fit_transform(location_df,
                                                                                            location_df.RainTomorrow)
            else:
                data.loc[data.Location == location] = self.encoders[location].transform(location_df,
                                                                                        location_df.RainTomorrow)

        if train:
            self.encoders['in_general'] = TargetEncoder(cols=['Location', 'Month'])
            encoded = self.encoders['in_general'].fit_transform(data, data.RainTomorrow)
        else:
            encoded = self.encoders['in_general'].transform(data, data.RainTomorrow)

        for c in ['WindGustDir', 'WindDir9am', 'WindDir3pm']:
            encoded[c] = encoded[c].astype(float)
        return encoded

    @staticmethod
    def rolling_features_for_all_locations(series_, locations, period, shift, strategy, corr=False):

        def rolling_features_for_location(series_, period, shift, strategy, corr=False):
            if not corr:
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
            else:
                return series_.shift(shift).rolling(period).corr(other=series_.index.to_series())

        all_locations = []
        for location in locations.unique():
            one_location, index = series_[locations == location].reset_index(drop=True, inplace=False), series_[
                locations == location].index.to_series()
            new_series_for_location = rolling_features_for_location(one_location, period, shift, strategy, corr=corr)
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
