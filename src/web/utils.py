import calendar
import datetime
from datetime import timedelta
import time
import warnings

import pandas as pd
from sklearn import preprocessing
from sklearn.base import BaseEstimator, TransformerMixin
# from src.web.encoder import SafeLabelEncoder
from src.web.encoder import EnhancedTargetEncoder
from src.web.encoder import EnhancedLeaveOneOutEncoder
import category_encoders as ce
import logging
from calendar import monthrange
# warnings.filterwarnings("ignore")

TXN_DATE_IN_STR = "transaction_date_in_string"
DAY_OF_MONTH = "day_of_month"
MONTH = 'month'
FEATURES_CAT = 'FEATURES_CAT'
FEATURES_NUM = 'FEATURES_NUM'
FEATURES_NUM_ENCODED = 'FEATURES_NUM_ENCODED'
FEATURES_ENCODED = 'FEATURES_ENCODED'

WEEK_OF_MONTH = 'week_of_month'
DAY_OF_WEEK = 'day_of_week'
IS_EXPIRED = 'is_expired'
CC_EXPIRE_DATE = 'cc_expiration_date'
FAILED_DECLINE_TYPE = 'failed_decline_type'
FAILED_RESPONSE_MESSAGE = 'failed_response_message'
IS_WEEKEND = 'is_weekend'
DAYS_BETWEEN = 'days_between'

BIN = 'bin'
MIN = 'Min'
MAX = 'Max'
MEAN = 'Mean'
STD_DEV = 'StdDev'
RENEW_ATT_NUM = 'renew_att_num'
FAILED_RESPONSE_CODE = 'failed_response_code'
FAILED_ATTEMPT_DATE = 'failed_attempt_date'
FAILED_DAY_OF_MONTH = 'failed_day_of_month'


def to_date(datestr):
    """Convert from date string to date time"""
    try:
        if len(datestr) <= 10:
            struct = time.strptime(datestr, "%Y-%m-%d")
        else:
            struct = time.strptime(datestr, "%Y-%m-%d %H:%M:%S")
        date = datetime.date(struct.tm_year, struct.tm_mon, struct.tm_mday)
    except Exception as ex:
        raise ValueError("date '{}' does not match format 'yyyy-mm-dd'".format(datestr))
    return date


# Get difference between d2 and d1 in days.
def days_between(d1, d2):
    d1 = to_date(d1)
    d2 = to_date(d2)
    return abs((d2 - d1).days)


#Get difference between d2 and d1 in days.
def days_between_ds(df):
    d1 = to_date(df[FAILED_ATTEMPT_DATE])
    d2 = to_date(df[TXN_DATE_IN_STR])
    return abs((d2 - d1).days)


def to_day(datestr):
    """Converts date string to day, e.g: '2018-05-02 00:00:00' to be '2' """
    return to_date(datestr).day


def to_month(datestr):
    return to_date(datestr).month


def week_of_month(datestr):
    """Determines the week (number) of the month, e.g: '2018-05-02 00:00:00' to be '1' (First week)"""
    date = to_date(datestr)
    # Calendar object. 6 = Start on Sunday, 0 = Start on Monday
    cal_object = calendar.Calendar(6)
    month_calendar_dates = cal_object.itermonthdates(date.year, date.month)

    day_of_week = 1
    week_number = 1

    for day in month_calendar_dates:
        # add a week and reset day of week
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
    date = datetime.date(struct.tm_year, struct.tm_mon, struct.tm_mday)
    return date.isoweekday()


def daterange(date1_str, date2_str):
    start_date = to_date(date1_str + ' 00:00:00')
    end_date = to_date(date2_str + ' 00:00:00')
    if start_date > end_date:
        raise ValueError('start date cannot be after the end date.')
    for n in range(int((end_date - start_date).days) + 1):
        yield str(start_date + timedelta(n))


def isnumeric(dtype: str):
    return dtype.startswith('float') or dtype.startswith('int')


def to_date_cc_expire_date(datestr):
    datestr = str(datestr).replace('/','')
    struct = time.strptime(datestr, "%m%y")
    last_date = monthrange(struct.tm_year,struct.tm_mon)[1]
    date = datetime.date(struct.tm_year,struct.tm_mon, last_date)
    return date


def is_expired(row):
    try:
        cc_expired_date = to_date_cc_expire_date(row[CC_EXPIRE_DATE])
        txn_date = to_date(row[TXN_DATE_IN_STR])
        return txn_date >= cc_expired_date
    except Exception as ex:
        return False


def is_weekend(day):
    """Determines whether the given day is weekend
    Args:
        day (str) : should be a lower case day_of_month. e.g: sunday, saturday
    Return True (bool) if the day is either sunday or saturday
    """
    if day in ['saturday', 'sunday']:
        return True
    return False


def cc_month(cc_expiration_date):
    cc_expiration_date = cc_expiration_date.replace('/', '')
    expire_month = None
    if len(cc_expiration_date) == 3:
        expire_month = int(cc_expiration_date[:1])
    elif len(cc_expiration_date) == 4:    
        expire_month = int(cc_expiration_date[:2])
                     
    return expire_month


class DeclineTypeUtil:
    """A util to group decline type based on provided response message"""

    DECLINE_TEXT = 'DECLINE_TEXT'
    DECLINE_TYPE = 'DECLINE_TYPE'
    BASE = 'Base'

    def __init__(self, df_decline_type):
        self._df_decline_type = df_decline_type
        self._msg_group = {'declined': 'decline',
                           'do_not_honor': 'do not honor',
                           'txn_refused': 'refuse',
                           'attempt_lower_amount': 'lower amount',
                           'Insufficient Funds': 'insufficient',
                           'not_allowed': 'not allowed',
                           'correct_cc_retry': 'correct card',
                           'invalid_cc': 'invalid card',
                           'lost_stolen': 'lost or stolen',
                           'invalid_account': 'invalid account',
                           'do_not_try': 'do not try',
                           'expired_card': 'expired',
                           'pickup_card': 'pick',
                           'blocked_first_used': 'blocked',
                           'invalid_txn': 'invalid trans',
                           'restricted_card': 'restricted',
                           'not_permitted': 'not permitted',
                           'expired card': 'expired card',
                           'unable to determine format': 'determine format',
                           'system error': 'error',
                           '' : ''
                           }

    def __group_response_msg(self, msg):
        """Groups decline type based on the given message"""
        other = self.BASE

        if isinstance(msg, str) == False:
            return other

        msg_lower = msg.lower()
        for key, val in self._msg_group.items():
            if val in msg_lower:
                return key

        return other

    def decline_type(self, response_msg):
        '''Converts to decline_type based on the given response_msg'''
        dec_type = self._df_decline_type[self._df_decline_type[self.DECLINE_TEXT] == response_msg][self.DECLINE_TYPE]
        if dec_type.empty or dec_type.iloc[0] == self.BASE:
            return self.__group_response_msg(response_msg)
        else:
            return dec_type.iloc[0]

