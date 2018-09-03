import calendar
import datetime
from datetime import timedelta
import time
import warnings

import pandas as pd
from sklearn import preprocessing
from sklearn.base import BaseEstimator, TransformerMixin
from encoder import SafeLabelEncoder

warnings.filterwarnings("ignore")

TXN_DATE_IN_STR = "transaction_date_in_string"
DAY_OF_MONTH = "day_of_month"
FEATURES_CAT = 'FEATURES_CAT'
FEATURES_NUM = 'FEATURES_NUM'
FEATURES_ENCODED = 'FEATURES_ENCODED'

def price_level(usd_amount=0):
    if usd_amount <= 300:
        return 'Cheap'
    elif usd_amount > 300 and usd_amount <= 1000:
        return 'Mid'
    #     elif usd_amount > 1000 and usd_amount <= 200:
    #         return 'Quite Expensive'
    else:
        return 'Expensive'

#Convert from str to datetime
def to_date(datestr):
    """Convert from date string to date time"""
    struct = time.strptime(datestr, "%Y-%m-%d %H:%M:%S")
    date = datetime.date(struct.tm_year,struct.tm_mon,struct.tm_mday)
    return date

#Get difference between d2 and d1 in days.
def days_between(d1, d2):
    d1 = to_date(d1)
    d2 = to_date(d2)
    return abs((d2 - d1).days)

def to_day(datestr):
    """Converts date string to day, e.g: '2018-05-02 00:00:00' to be '2' """
    return to_date(datestr).day


def week_of_month(datestr):
    """Determines the week (number) of the month, e.g: '2018-05-02 00:00:00' to be '1' (First week)"""
    date = to_date(datestr)
    #Calendar object. 6 = Start on Sunday, 0 = Start on Monday
    cal_object = calendar.Calendar(6)
    month_calendar_dates = cal_object.itermonthdates(date.year,date.month)

    day_of_week = 1
    week_number = 1

    for day in month_calendar_dates:
        #add a week and reset day of week
        if day_of_week > 7:
            week_number += 1
            day_of_week = 1

        if date == day:
            break
        else:
            day_of_week += 1

    return str(week_number)

def to_weekday(datestr):
    struct = time.strptime(datestr, "%Y-%m-%d %H:%M:%S")
    date = datetime.date(struct.tm_year,struct.tm_mon,struct.tm_mday)
    return date.isoweekday()

def daterange(date1_str, date2_str):
    start_date = to_date(date1_str + ' 00:00:00')
    end_date = to_date(date2_str + ' 00:00:00')
    if start_date > end_date:
        raise ValueError('start date cannot be after the end date.')
    for n in range(int((end_date - start_date).days) + 1):
        yield str(start_date + timedelta(n))


class PreProcessing(BaseEstimator, TransformerMixin):
    """Custom Pre-Processing estimator for Best Retry
    """

    def __init__(self):
        pass

    def transform(self, df):
        """Regular transform() that is a help for training, validation & testing datasets
           (NOTE: The operations performed here are the ones that we did prior to this cell)
        """
        # Consolidated feature processing

        df['week_of_month'] = df[TXN_DATE_IN_STR].apply(week_of_month)
        df['day_of_week'] = df[TXN_DATE_IN_STR].apply(to_weekday)

        df_encoded  = pd.DataFrame(columns=self.features_list)

        for k, v in self.encoders.items():
            if df[k].dtype == 'float64':
                df[k] = df[k].fillna(-1).astype(int)

            df_encoded[k] = v.transform(df[k].astype(str).str.lower().str.replace(' ',''))

        #Num processing
        df_num = df[self.FEATURES_NUM].astype(float)
        df_num = self.scaler.transform(df_num)

        df_encoded[self.FEATURES_NUM] = df_num

        return df_encoded.as_matrix()

    def fit(self, df, y=None, features_dict={}, **fit_params):

        print('features_dict: ', features_dict)
        self.FEATURES_CAT = features_dict[FEATURES_CAT]
        self.FEATURES_NUM = features_dict[FEATURES_NUM]
        self.FEATURES_ENCODED = features_dict[FEATURES_ENCODED]
        self.FEATURES = self.FEATURES_CAT + self.FEATURES_ENCODED

        df['week_of_month'] = df['transaction_date_in_string'].apply(week_of_month)
        df['day_of_week'] =  df['transaction_date_in_string'].apply(to_weekday)


        # one hot for categorical feature ###
        #         self.features_list = list(pd.get_dummies(df[FEATURES].astype(str), prefix=FEATURES).columns.values) + FEATURES_NUM

        self.features_list = self.FEATURES + self.FEATURES_NUM
        print("self.features_list: ", self.features_list)
        feature_encoders = {}
        for f in self.FEATURES:
            if df[f].dtype == 'float64':
                df[f] = df[f].fillna(-1).astype(int)

            encoder = SafeLabelEncoder().fit(df[f].astype(str).str.lower().str.replace(' ',''))
            feature_encoders[f] = encoder

        self.encoders = feature_encoders

        #Fit a scaler
        df_num = df[self.FEATURES_NUM].astype(float)
        self.scaler = preprocessing.StandardScaler().fit(df_num)
        return self