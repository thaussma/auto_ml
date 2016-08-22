import datetime

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction import DictVectorizer


class FeatureEngineer(BaseEstimator, TransformerMixin):

    # Future: consider a dates_since_min_date feature. This has a high risk of over-fitting though, so I'd prefer not to include it easily.
    def __init__(self, date_cols, return_sparse=False):
        self.return_sparse = return_sparse
        self.date_cols = date_cols


    def fit(self, X, y=None):

        # if we're returning sparse data, we will need to fit a DictVectorizer

        if self.return_sparse:
            predicted_vals = self._iterate_X_and_get_date_features(X)
            self.dv = DictVectorizer(sparse=True)
            self.dv.fit(predicted_vals)

        return self

    def _iterate_X_and_get_date_features(self, X):
        list_of_feature_dicts = []

        for idx, x_row in enumerate(X):

            for date_col in self.date_cols:
                date_val = x_row.pop(date_col, False)

                # make sure this property exists for this x_row
                if date_val:

                    # make sure that value is actually a datetime object
                    if type(date_val) not in (datetime.date, datetime.datetime):
                        print('This value is not a date:')
                        print(date_val)
                        try:
                            date_val = datetime.datetime(date_val)
                        except:
                            pass

                    if self.return_sparse is False:
                        x_row = self.extract_features(date_val, date_col, feature_dict=x_row)
                        X[idx] = x_row
                    else:
                        # if we're returning sparse output in a FeatureUnion, get a dict that ONLY has the relevant date features
                        # we will later feed this through a DictVectorizer to turn the dicts into sparse matrices
                        features_in_dict = self.extract_features(date_val, date_col, feature_dict={})
                        list_of_feature_dicts.append(features_in_dict)


        if self.return_sparse is False:
            return X
        else:
            return list_of_feature_dicts


    def extract_features(self, date_val, date_col, feature_dict):

        feature_dict[date_col + '_day_of_week'] = str(date_val.weekday())
        feature_dict[date_col + '_hour'] = date_val.hour

        minutes_into_day = date_val.hour * 60 + date_val.minute

        if feature_dict[date_col + '_day_of_week'] in (5,6):
            feature_dict[date_col + '_is_weekend'] = True
        elif feature_dict[date_col + '_day_of_week'] == 4 and feature_dict[date_col + '_hour'] > 16:
            feature_dict[date_col + '_is_weekend'] = True
        else:
            feature_dict[date_col + '_is_weekend'] = False

            # Grab rush hour times for the weekdays.
            # We are intentionally not grabbing them for the weekends, since weekend behavior is likely very different than weekday behavior.
            if minutes_into_day < 120:
                feature_dict[date_col + '_is_late_night'] = True
            elif minutes_into_day < 11.5 * 60:
                feature_dict[date_col + '_is_off_peak'] = True
            elif minutes_into_day < 13.5 * 60:
                feature_dict[date_col + '_is_lunch_rush_hour'] = True
            elif minutes_into_day < 17.5 * 60:
                feature_dict[date_col + '_is_off_peak'] = True
            elif minutes_into_day < 20 * 60:
                feature_dict[date_col + '_is_dinner_rush_hour'] = True
            elif minutes_into_day < 22.5 * 60:
                feature_dict[date_col + '_is_off_peak'] = True
            else:
                feature_dict[date_col + '_is_late_night'] = True

        return feature_dict


    def transform(self, X, y=None):

        features = self._iterate_X_and_get_date_features(X)


        if self.return_sparse is False:
            return features
        else:
            return self.dv.transform(features)


