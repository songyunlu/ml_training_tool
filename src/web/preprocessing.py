from src.web.utils import *
import numpy as np
import pandas as pd
from SortedSet.sorted_set import SortedSet

PAY_AMOUNT_USD = "payment_amount_usd"

MEAN_DIFF = "mean_diff"
MEDIAN_DIFF = "median_diff"
MAX_95_DIFF = "max_95_diff"
MAX_99_DIFF = "max_99_diff"
STD_DIFF = "std_diff"
FEATURES_NUM_CALCULATED_KEY = "FEATURES_NUM_CALCULATED"
FEATURES_FLOAT_KEY = "FEATURES_FLOAT"
ADDITIONAL_FIELDS_KEY = 'ADDITIONAL_FIELDS'

TXN_DATE_IN_STR = "transaction_date_in_string"
DAY_OF_MONTH = "day_of_month"
MONTH = 'month'
FEATURES_CAT_KEY = 'FEATURES_CAT'
FEATURES_NUM_KEY = 'FEATURES_NUM'
FEATURES_NUM_ENCODED_KEY = 'FEATURES_NUM_ENCODED'
FEATURES_NUM_BIN_PROFILE_KEY = 'FEATURES_NUM_BIN_PROFILE'
FEATURES_ENCODED_KEY = 'FEATURES_ENCODED'
FEATURES_GROUPED = 'FEATURES_GROUPED'

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
MEDIAN = 'Median'
STD_DEV = 'StdDev'
RENEW_ATT_NUM = 'renew_att_num'
FAILED_RESPONSE_CODE = 'failed_response_code'
FAILED_ATTEMPT_DATE = 'failed_attempt_date'
FAILED_DAY_OF_MONTH = 'failed_day_of_month'

first_cal_response_message_fields = ['first_cal_response_message_1', 'first_cal_response_message_2', 'first_cal_response_message_3']
first_cal_response_code_fields = ['first_cal_response_code_1', 'first_cal_response_code_2', 'first_cal_response_code_3']
previous_cal_response_code_fields = ['previous_cal_response_code_1', 'previous_cal_response_code_2', 'previous_cal_response_code_3']

BASE = 'Base'

def processor_mid_changed(row):
    failed_payment_service_id = row['failed_payment_service_id']
    payment_service_id = row['payment_service_id']
    failed_merchant_number = row['failed_merchant_number']
    merchant_number = row['merchant_number']

    if failed_payment_service_id == payment_service_id:
        if merchant_number == failed_merchant_number:
            return "processor_and_mid_unchanged"
        else:
            return "processor_unchanged_mid_changed"
    else:
        if merchant_number == failed_merchant_number:
            return "processor_changed_mid_unchanged"
        else:
            return "processor_changed_mid_changed"


def expired_years_diff(row):
    datestr = row['cc_expiration_date']
    datestr = str(datestr).replace('/', '')
    struct = time.strptime(datestr, "%m%y")
    original_year = struct.tm_year
    txn_year = time.strptime(row['transaction_date_in_string'], '%Y-%m-%d %H:%M:%S').tm_year
    return abs(txn_year - original_year)


def years_over(row):
    expiration_years_diff = 0
    try:
        date_increment = int(row['date_increment'])
        expiration_years_diff = expired_years_diff(row)

        if date_increment < 1:
            return -expiration_years_diff

        total_increment = date_increment
        while total_increment < expiration_years_diff:
            total_increment += date_increment

        if total_increment == date_increment:
            credict_card_month = cc_month(row['cc_expiration_date'])
            txn_month = to_month(row['transaction_date_in_string'])
            if credict_card_month < txn_month:
                total_increment += date_increment
        return total_increment - expiration_years_diff
    except:
        return -expiration_years_diff


class GroupUtil:
    def __init__(self, group_dict):
        self.group_dict = group_dict

    def get_bin_profile_value(self, row, per_date_dict, per_day_of_month_dict):
        val = self.group_dict.get(per_date_dict, {}).get((str(row['bin']), row['month'], row['day_of_month']), None)
        if not val:
            val = self.group_dict.get(per_day_of_month_dict, {}).get((str(row['bin']), row['day_of_month']), None)
        return val

    def get_bank_profile_value(self, row, per_date_dict, per_day_of_month_dict):
        val = self.group_dict.get(per_date_dict, {}).get((str(row['bank_name']), row['card_category'], row['month'], row['day_of_month']), None)
        if not val:
            val = self.group_dict.get(per_day_of_month_dict, {}).get((str(row['bank_name']), row['card_category'], row['day_of_month']), None)
        return val

    def date_max_diff(self, row):
        result = -1
        val = self.get_bin_profile_value(row, 'max_per_date_month_dict', 'max_per_day_of_month_dict')
        if isinstance(val, float) or isinstance(val, int):
            result = val - float(row[PAY_AMOUNT_USD])
        return result

    def date_max_99_diff(self, row):
        result = -1
        val = self.get_bin_profile_value(row, 'max_99_per_date_month_dict', 'max_99_per_day_of_month_dict')
        if isinstance(val, float) or isinstance(val, int):
            result = val - float(row[PAY_AMOUNT_USD])
        return result

    def date_max_bank_card_diff(self, row):
        val = self.get_bank_profile_value(row, 'max_per_bank_card_date_month_dict', 'max_per_bank_card_day_of_month_dict')
        if isinstance(val, float) or isinstance(val, int):
            return val - float(row[PAY_AMOUNT_USD])
        else:
            return self.date_max_diff(row)

    def date_max_99_bank_card_diff(self, row):
        val = self.get_bank_profile_value(row, 'max_99_per_bank_card_date_month_dict', 'max_99_per_bank_card_day_of_month_dict')
        if isinstance(val, float) or isinstance(val, int):
            return val - float(row[PAY_AMOUNT_USD])
        else:
            return self.date_max_99_diff(row)

    def bin_date_max_comparison(self, row):
        result = 0.99
        try:
            val = self.get_bin_profile_value(row, 'max_per_date_month_dict', 'max_per_day_of_month_dict')
            if isinstance(val, float) or isinstance(val, int):
                result = float(row[PAY_AMOUNT_USD]) / val
        except:
            pass
        return result

    def bin_date_mean_comparison(self, row):
        result = 2.0
        try:
            val = self.get_bin_profile_value(row, 'mean_per_date_month_dict', 'mean_per_day_of_month_dict')
            if isinstance(val, float) or isinstance(val, int):
                result = float(row[PAY_AMOUNT_USD]) / val
        except:
            pass
        return result

    def success_bin_count_per_day_of_month(self, row):
        val = self.get_bin_profile_value(row, 'success_per_date_month_dict', 'success_per_day_of_month_dict')
        if isinstance(val, float) or isinstance(val, int):
            return int(val)
        return 0

    def success_bank_count_per_day_of_month(self, row):
        val = self.get_bank_profile_value(row, 'success_per_bank_card_date_month_dict', 'success_per_bank_card_day_of_month_dict')
        if isinstance(val, float) or isinstance(val, int):
            return int(val)
        else:
            return self.success_bin_count_per_day_of_month(row)

    def date_median_diff(self, row):
        result = 0
        val = self.get_bin_profile_value(row, 'median_per_date_dict', 'median_per_day_of_month_dict')
        if isinstance(val, float) or isinstance(val, int):
            result = val - float(row[PAY_AMOUNT_USD])
        return result

    def date_mean_diff(self, row):
        result = 0
        val = self.get_bin_profile_value(row, 'mean_per_date_dict', 'mean_per_day_of_month_dict')
        if isinstance(val, float) or isinstance(val, int):
            result = val - float(row[PAY_AMOUNT_USD])
        return result

def get_hour(txn_hour_min_segment):
    """Return transaction_hour from txn_hour with minute segment input"""
    try:
        return int(txn_hour_min_segment.split(':')[0])
    except:
        return -1

class PreProcessing(BaseEstimator, TransformerMixin):
    """Custom Pre-Processing estimator for Best Retry
    """
    default_txn_hour_group = [0, 2, 6, 10, 14, 18, 22, 25]

    def __init__(self):
        self.df_decline_type = None
        self.bin_profile = None
        self.group_util = None

        self.features_cat = None
        self.features_encoded = None
        self.features_cat_and_encoded = None
        self.features_num = None
        self.features_num_encoded = None
        self.features_bin_profile = None
        self.features_num_calculated = None
        self.features_float = None
        self.features_all = None
        self.encoder = None
        self.scaler = None
        self.fillna_val = None
        self.feat_grouped = None
        self.features_grouped_encoded = None
        self.additional_fields = None
        self.use_cat_encoder = True
        self.txn_hour_group = []

        '''first_level_models is a dict consist of model_name as key and model_file as value'''
        self.first_level_models: dict = {}
        # self.first_level_models_field_names = []
        pass

    def convert_str_to_sorted_set(self, s):
        """This can be used to convert failed response code fields that are
        consolidated in given s to be a sorted unique values set"""
        x = str(s).replace(' ', '').replace('.0', '')
        response_codes = SortedSet(x.split(',')) - SortedSet([''])
        unique_codes = response_codes - self.generic_decline_codes
        if not unique_codes:
            unique_codes = response_codes
        return ",".join(unique_codes)

    def convert_sin(self, value, max_value, start=0):
        return np.sin((value - start) * (2. * np.pi / max_value))

    def convert_cos(self, value, max_value, start=0):
        return np.cos((value - start) * (2. * np.pi / max_value))

    def convert_date_sin(self, txn_date_str):
        date = to_date(txn_date_str)
        last_date = calendar.monthrange(date.year, date.month)[1]
        return self.convert_sin(date.day, last_date, 1)

    def convert_date_cos(self, txn_date_str):
        date = to_date(txn_date_str)
        last_date = calendar.monthrange(date.year, date.month)[1]
        return self.convert_cos(date.day, last_date, 1)

    def convert_day_of_week_sin(self, day_of_week):
        return self.convert_sin(day_of_week, 7, 1)

    def convert_day_of_week_cos(self, day_of_week):
        return self.convert_cos(day_of_week, 7, 1)

    def convert_txn_hour_sin(self, txn_hour):
        return self.convert_sin(txn_hour, 24, 0)

    def convert_txn_hour_cos(self, txn_hour):
        return self.convert_cos(txn_hour, 24, 0)

    def handle_feat_cyclical(self, df):
        if 'date_sin' in self.features_num_encoded and 'date_cos' in self.features_num_encoded:
            df['date_sin'] = df[TXN_DATE_IN_STR].apply(self.convert_date_sin)
            df['date_cos'] = df[TXN_DATE_IN_STR].apply(self.convert_date_cos)

        if 'day_of_week_sin' in self.features_num_encoded and 'day_of_week_cos' in self.features_num_encoded:
            df['day_of_week_sin'] = df['day_of_week'].apply(self.convert_day_of_week_sin)
            df['day_of_week_cos'] = df['day_of_week'].apply(self.convert_day_of_week_cos)

        if 'txn_hour_sin' in self.features_num_encoded and 'txn_hour_cos' in self.features_num_encoded:
            df['txn_hour_sin'] = df['transaction_hour'].apply(self.convert_txn_hour_sin)
            df['txn_hour_cos'] = df['transaction_hour'].apply(self.convert_txn_hour_cos)

        return df

    def handle_feat_float(self, df):
        for feat in self.features_float:
            if feat in df.columns:
                df[feat] = df[feat].fillna('').astype(str).str.replace('.0', '', regex=False)
        return df

    def encode_feat_grouped(self, df):
        for feat_group in self.feat_grouped:
            group_name = "-".join(feat_group)
            df[group_name] = df[feat_group].fillna('').astype(str).apply(lambda x: '-'.join(x), axis=1)

    def handle_feat_encoded(self, df):
        if WEEK_OF_MONTH in self.features_encoded:
            df[WEEK_OF_MONTH] = df[TXN_DATE_IN_STR].apply(week_of_month)

        if DAY_OF_WEEK in self.features_encoded:
            df[DAY_OF_WEEK] = df[TXN_DATE_IN_STR].apply(to_weekday)

        if IS_EXPIRED in self.features_cat_and_encoded and 'cc_expiration_date' in df.columns and IS_EXPIRED not in df.columns:
            df[IS_EXPIRED] = df.apply(is_expired, axis=1)

        if DAY_OF_WEEK not in self.features_encoded:
            df[DAY_OF_WEEK] = df[TXN_DATE_IN_STR].apply(to_weekday)

        df[IS_WEEKEND] = df[DAY_OF_WEEK].apply(is_weekend)

        if 'min_segment' in self.features_encoded and 'min_segment' not in df.columns:
            df['min_segment'] = df[TXN_DATE_IN_STR].apply(to_min_segment)

        if 'txn_hour_min_segment' in self.features_encoded and 'txn_hour_min_segment' not in df.columns:
            df['txn_hour_min_segment'] = df[TXN_DATE_IN_STR].apply(hour_min_segment)

        if FAILED_ATTEMPT_DATE in df.columns:
            try:
                df[DAYS_BETWEEN] = df.apply(days_between_ds, axis=1)
            except ValueError:
                df[DAYS_BETWEEN] = None

        if 'days_between_from_first_cal' in self.features_cat_and_encoded and 'days_between_from_first_cal' not in df.columns:
            try:
                df["days_between_from_first_cal"] = df.apply(init_days_between_ds, axis=1)
            except ValueError:
                df["days_between_from_first_cal"] = df[DAYS_BETWEEN]

        if 'failed_decline_type_from_first_cal' in self.features_cat_and_encoded and 'failed_decline_type_from_first_cal' not in df.columns:
            try:
                failed_decline_type_from_first_cal = BASE
                response_messages = ','.join(df[x].at[0] or '' for x in first_cal_response_message_fields if x in df.columns).replace(',', '')
                if response_messages:
                    failed_decline_type_from_first_cal = decline_type_util_v2.decline_type(response_messages)

                if failed_decline_type_from_first_cal != BASE:
                    df['failed_decline_type_from_first_cal'] = failed_decline_type_from_first_cal
                else:
                    df['failed_decline_type_from_first_cal'] = df['failed_decline_type']
            except ValueError:
                df["failed_decline_type_from_first_cal"] = df['failed_decline_type']

        if 'failed_response_codes_from_first_cal' in self.features_cat_and_encoded and 'failed_response_codes_from_first_cal' not in df.columns:
            try:
                response_codes = ','.join(df[x].at[0] or '' for x in first_cal_response_code_fields if x in df.columns)
                if response_codes:
                    df['failed_response_codes_from_first_cal'] = response_codes
                    df['failed_response_codes_from_first_cal'] = df.failed_response_codes_from_first_cal.apply(self.convert_str_to_sorted_set)

            except ValueError:
                df["failed_response_codes_from_first_cal"] = df['failed_response_code']

        if 'failed_response_codes_from_previous_cal' in self.features_cat_and_encoded and 'failed_response_codes_from_previous_cal' not in df.columns:
            try:
                response_codes = ','.join(df[x].at[0] or '' for x in previous_cal_response_code_fields if x in df.columns)
                if response_codes:
                    df['failed_response_codes_from_previous_cal'] = response_codes
                    df['failed_response_codes_from_previous_cal'] = df.failed_response_codes_from_previous_cal.apply(self.convert_str_to_sorted_set)

            except ValueError:
                df["failed_response_codes_from_previous_cal"] = df['failed_response_code']

        if 'num_of_days' in self.features_cat_and_encoded:
            df['num_of_days'] = df[TXN_DATE_IN_STR].apply(num_of_days)

        if FAILED_DAY_OF_MONTH in self.features_cat_and_encoded:
            df[FAILED_DAY_OF_MONTH] = df[FAILED_ATTEMPT_DATE].apply(to_day)

        if MONTH in self.features_cat_and_encoded:
            df[MONTH] = df[TXN_DATE_IN_STR].apply(to_month)

        if "cc_expiration_date" in df.columns:
            df['cc_expiration_date'] = df['cc_expiration_date'].apply(str)

        if 'expiration_date_changed' in self.features_cat_and_encoded and 'expiration_date_changed' not in df.columns:
            failed_cc = df['failed_cc_expiration_date'].replace('/', '')
            current_cc = df['cc_expiration_date'].replace('/', '')
            df['expiration_date_changed'] = (failed_cc != current_cc)

        if 'processor_mid_changed' in self.features_cat_and_encoded and 'processor_mid_changed' not in df.columns:
            df['processor_mid_changed'] = df.apply(processor_mid_changed, axis=1)

        if "cc_month" in self.features_cat_and_encoded and "cc_month" not in df.columns:
            df['cc_month'] = df["cc_expiration_date"].apply(cc_month)

        if "card_info_is_empty" in self.features_cat_and_encoded and "card_info_is_empty" not in df.columns:
            try:
                df['card_info_is_empty'] = (df['card_brand'] == '') & (df['card_category'] == '')
            except:
                df['card_info_is_empty'] = None

        segment_num_group = [-1, 1, 2, 3, 4, 5, 6, 7, 8, 15, 20, 25, 30, 40, 50, 70, 100, 150]
        duration_group = [0, 3, 6, 9, 13, 17, 20, 25, 27, 33, 39, 43, 62, 70, 80, 88, 94, 100, 118, 125, 130, 146, 155, 176, 184, 200, 213, 230, 263, 300, 363, 368, 373,729, 733, 1000, 2000]
        amount_group = [-1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 170, 190, 210, 250, 300, 400, 500, 1000, 1500, 2000, 5000, 10000, 20000]


        if "segment_num_group" in self.features_cat_and_encoded and "segment_num_group" not in df.columns:
            try:
                df['segment_num_group'] = pd.cut(df['segment_num'], segment_num_group).astype(str).str.replace('.0', '', regex=False)
            except:
                df['segment_num_group'] = None

        if "payment_amount_group" in self.features_cat_and_encoded and "payment_amount_group" not in df.columns:
            try:
                df['payment_amount_group'] = pd.cut(df['payment_amount_usd'], amount_group).astype(str).str.replace('.0', '', regex=False)
            except:
                df['payment_amount_group'] = 'nan'

        if "is_first_renewal" in self.features_cat_and_encoded and "is_first_renewal" not in df.columns:
            try:
                df['is_first_renewal'] = (df['segment_num'] < 2) & (df['segment_num'] >= 0)
            except:
                df['is_first_renewal'] = None

        if "duration" in df.columns:
            df.loc[(df['duration'] == 28) | (df['duration'] == 29) | (df['duration'] == 31), 'duration'] = 30
            df.loc[(df['duration'] == 366), 'duration'] = 365
            df.loc[(df['duration'] == 731), 'duration'] = 730

        if "sub_duration_group" in self.features_cat_and_encoded and "sub_duration_group" not in df.columns:
            try:
                df['sub_duration_group'] = pd.cut(df['duration'], duration_group).astype(str).str.replace('.0', '', regex=False)
            except:
                df['sub_duration_group'] = None

        if "sub_age_group" in self.features_cat_and_encoded and "sub_age_group" not in df.columns:
            try:
                df['sub_age_group'] = pd.cut(df['sub_age'], duration_group).astype(str).str.replace('.0', '', regex=False)
            except:
                df['sub_age_group'] = None

        if ("transaction_hour" in self.features_cat_and_encoded or "txn_hour_group" in self.features_cat_and_encoded) \
                and "transaction_hour" not in df.columns:
            try:
                df['transaction_hour'] = df['txn_hour_min_segment'].apply(get_hour)
            except:
                df['transaction_hour'] = -1

        if "txn_hour_group" in self.features_cat_and_encoded and "txn_hour_group" not in df.columns:
            try:
                df['txn_hour_group'] = pd.cut(df['transaction_hour'], self.txn_hour_group).astype(str).str.replace('.0', '', regex=False)
            except:
                df['txn_hour_group'] = None

        if "issuer_country" in self.features_cat_and_encoded and "billing_country" in df.columns:
            df["issuer_country"] = df["issuer_country"].replace('', np.nan).fillna(df["billing_country"])

        if "bank_code" in self.features_cat_and_encoded and "bank_code" in df.columns:
            df.loc[(df.bank_code.str.lower() != "non3ds") & (df.bank_code.str.lower() != "rb"), 'bank_code'] = 'other'

        if "date_increment" in self.features_cat_and_encoded:
            df["date_increment"] = df["date_increment"].replace('', np.nan).fillna('NONE')
            df.loc[df.date_increment == 'nan', 'date_increment'] = 'NONE'

        if "bank_name" in df.columns:
            df['bank_name'] = df["bank_name"].astype(str).apply(lambda x: x.lower().replace(' ', '').replace("nationalassociation", "n.a").replace(",", ""))

        if "card_category" in df.columns:
            try:
                df.loc[(df.card_brand.str.lower().str.startswith('american', na=False)), 'card_category'] = 'american_express'
                df.loc[(df.card_brand.str.lower().str.startswith('american', na=False)), 'funding_source'] = 'american_express'
            except Exception as ex:
                print(ex)

            df['card_category'] = df["card_category"].astype(str).apply(lambda x: x.lower().replace(' ', '').replace(",", ""))

        if "bank_name" in df.columns:
            try:
                df.loc[(df.card_brand.str.lower().str.startswith('american', na=False)), 'bank_name'] = 'american_express'
                df.loc[(df.card_brand.str.lower().str.startswith('discover', na=False)), 'bank_name'] = 'discover'
            except Exception as ex:
                print(ex)

        if "card_class" in df.columns:
            try:
                df.loc[(df.card_brand.str.lower().str.startswith('american', na=False)), 'card_class'] = 'american_express'
                df.loc[(df.card_brand.str.lower().str.startswith('discover', na=False)), 'card_class'] = 'discover'
            except Exception as ex:
                print(ex)

        print("# Finish handle_feat_encoded.")
        return df

    def handle_stack_models(self, df):
        # df_input = df.copy()
        for model_name, model_file in self.first_level_models.items():
            feat_name = f'predict_proba_{model_name}'
            predict_proba = model_file.predict_proba(df)
            df[feat_name] = list(map(lambda x: x[1], predict_proba.tolist()))
        return df

    def handle_feat_num_encoded(self, df):
        if BIN in df.columns:
            df[BIN] = pd.to_numeric(df[BIN], errors='coerce')
            df[BIN] = df[BIN].astype(str).str.replace('.0', '', regex=False)


            if "Max_99" in self.features_num_encoded and "Max_99" not in df.columns and self.features_bin_profile is not None:
                df[BIN] = df[BIN].astype(str).str.replace('.0', '', regex=False)
                df = pd.merge(df, self.bin_profile[[BIN] + self.features_bin_profile], left_on=BIN, right_on=BIN,
                              how='left')

            if MEAN_DIFF in self.features_num_calculated and MEAN_DIFF not in df.columns:
                df[MEAN_DIFF] = df[MEAN] - df[PAY_AMOUNT_USD]

            if MEDIAN_DIFF in self.features_num_calculated and MEDIAN_DIFF not in df.columns:
                df[MEDIAN_DIFF] = df[MEDIAN] - df[PAY_AMOUNT_USD]

            if 'max_diff' in self.features_num_calculated and 'max_diff' not in df.columns:
                df['max_diff'] = df['Max'] - df[PAY_AMOUNT_USD]

            if MAX_99_DIFF in self.features_num_calculated and MAX_99_DIFF not in df.columns:
                df[MAX_99_DIFF] = df['Max_99'] - df[PAY_AMOUNT_USD]

            if STD_DIFF in self.features_num_calculated and STD_DIFF not in df.columns:
                if MEAN_DIFF not in self.features_num_calculated:
                    df[MEAN_DIFF] = df[MEAN] - df[PAY_AMOUNT_USD]

                df[STD_DIFF] = df[STD_DEV] - abs(df[MEAN_DIFF])

        if "expired_years_diff" in self.features_num_encoded:
            df["expired_years_diff"] = df.apply(expired_years_diff, axis=1)

        if "years_over" in self.features_num_encoded and 'date_increment' in df.columns:
            df["years_over"] = df.apply(years_over, axis=1)

        if "date_inc_bin" in self.features_num_encoded:
            df["date_inc_bin"] = df.apply(self.eco_bin_util.date_inc_bin, axis=1)

        if "add_expiry_years_bin" in self.features_num_encoded:
            if "added_expiry_years" not in df.columns:
                df["added_expiry_years"] = df["expired_years_diff"] + df["years_over"]
                df["added_expiry_years"] = df["added_expiry_years"].fillna('').astype(str).str.replace('.0', '',
                                                                                                       regex=False)
            df["add_expiry_years_bin"] = df.apply(self.eco_bin_util.added_years_bin, axis=1)

        #         if "payment_mid_bin" in self.features_num_encoded:
        #             df["payment_mid_bin"] = df.apply(self.payment_mid_bin_util.payment_mid_bin, axis=1)

        if "txn_amount_bin_max_per_date_diff" in self.features_num_encoded:
            df["txn_amount_bin_max_per_date_diff"] = df.apply(self.group_util.date_max_diff, axis=1)

        if "txn_amount_bin_max_99_per_date_diff" in self.features_num_encoded:
            df["txn_amount_bin_max_99_per_date_diff"] = df.apply(self.group_util.date_max_99_diff, axis=1)

        if "txn_amount_bank_card_max_per_date_diff" in self.features_num_encoded:
            df["txn_amount_bank_card_max_per_date_diff"] = df.apply(self.group_util.date_max_bank_card_diff, axis=1)

        if "txn_amount_bank_card_max_99_per_date_diff" in self.features_num_encoded:
            df["txn_amount_bank_card_max_99_per_date_diff"] = df.apply(self.group_util.date_max_99_bank_card_diff, axis=1)

        if "bin_date_max_comparison" in self.features_num_encoded:
            df["bin_date_max_comparison"] = df.apply(self.group_util.bin_date_max_comparison, axis=1)

        if "bin_date_mean_comparison" in self.features_num_encoded:
            df["bin_date_mean_comparison"] = df.apply(self.group_util.bin_date_mean_comparison, axis=1)

        if 'success_bin_count_per_day_of_month' in self.features_num_encoded:
            df["success_bin_count_per_day_of_month"] = df.apply(self.group_util.success_bin_count_per_day_of_month, axis=1)

        if 'success_bank_card_count_per_day_of_month' in self.features_num_encoded:
            df["success_bank_card_count_per_day_of_month"] = df.apply(self.group_util.success_bank_count_per_day_of_month, axis=1)

        if "date_median_diff" in self.features_num_encoded:
            df["date_median_diff"] = df.apply(self.group_util.date_median_diff, axis=1)

        if "date_max" in self.features_num_encoded:
            df["date_max"] = df.apply(self.group_util.date_max, axis=1)

        if "date_mean_diff" in self.features_num_encoded:
            df["date_mean_diff"] = df.apply(self.group_util.date_mean_diff, axis=1)


        df = self.handle_feat_cyclical(df)

        return df

    def convert_mid(self, row):
        mid = row['merchant_number']
        if row['payment_service_id'] is not None and (
                row['payment_service_id'].startswith('netgiro-') or row['payment_service_id'].startswith('drwp-')):
            mid = mid.split('-')[0]
            mid = mid + "-" + row['payment_currency'].upper() + "-pacific"

        return mid

    def convert_failed_mid(self, row):
        mid = row['failed_merchant_number']
        if row['failed_payment_service_id'] is not None and (
                row['failed_payment_service_id'].startswith('netgiro-') or row['failed_payment_service_id'].startswith(
            'drwp-')):
            mid = mid.split('-')[0]

        return mid

    def handle_mid(self, df):
        if "merchant_number" in self.features_cat_and_encoded and "payment_service_id" in self.features_cat_and_encoded:
            df["merchant_number"] = df.apply(self.convert_mid, axis=1)

        if "failed_merchant_number" in self.features_cat_and_encoded and "failed_payment_service_id" in self.features_cat_and_encoded:
            df["failed_merchant_number"] = df.apply(self.convert_failed_mid, axis=1)

        return df

    def handle_processor(self, df):
        if 'payment_service_id' in df.columns:
            df['payment_service_id'] = df["payment_service_id"].astype(str).str.replace('drwp-', 'netgiro-', regex=False)

    def encode(self, df):
        # Consolidated feature processing
        df_encoded_all = pd.DataFrame(columns=self.features_all)
        df = self.handle_stack_models(df)

        df = self.handle_mid(df)
        df = self.handle_feat_float(df)
        df = self.handle_feat_encoded(df)
        df = self.handle_feat_num_encoded(df)
        df = df.reset_index()
        df[self.features_cat_and_encoded] = df[self.features_cat_and_encoded].fillna('')
        df[self.features_cat_and_encoded] = df[self.features_cat_and_encoded].astype(str).apply(
            lambda x: x.str.lower().replace(' ', '', regex=True)
                .replace("nodatafound',value:'n/a", "", regex=False)
                .replace("nodatafound", "", regex=False)
                .replace("nodatafound'value:'n/a", "",regex=False))

        self.handle_processor(df)
        self.encode_feat_grouped(df)

        time_start = time.time()

        if self.use_cat_encoder:
            df_encoded_cat = self.encoder.transform(df[self.features_cat + self.features_grouped_encoded])
        else:
            df_encoded_cat = df[self.features_cat + self.features_grouped_encoded]

        transform_time = time.time() - time_start
        print("# transform_time:", transform_time)
        df_encoded_all[self.features_cat + self.features_grouped_encoded] = df_encoded_cat

        # Num processing
        df_num = df[self.features_num + self.features_num_encoded + self.features_num_calculated].astype(float)
        self.fillna_val = df[self.features_num + self.features_num_encoded + self.features_num_calculated ].astype(float).mean()
        df_num = df_num.fillna(self.fillna_val)
        # if not df_num.empty:
        #     df_num = self.scaler.transform(df_num.fillna(self.fillna_val))
        df_encoded_all[self.features_num + self.features_num_encoded + self.features_num_calculated] = df_num

        return df_encoded_all

    def transform(self, df):
        """Regular transform() that is a help for training, validation & testing datasets
           (NOTE: The operations performed here are the ones that we did prior to this cell)
        """
        df_encoded_all = self.encode(df)
        print(df_encoded_all.head())

        return df_encoded_all.values

    def fit(self, df, y=None, features_dict={}, **fit_params):
        """fit is called only when training, this should not be called when predicting"""
        self.generic_decline_codes = features_dict.get('generic_decline_codes', SortedSet([]))
        # self.first_level_models = features_dict.get('first_level_models', {})
        # self.first_level_models_field_names = [f'predict_proba_{k}' for k in self.first_level_models.keys()]
        df = self.handle_stack_models(df)

        self.features_num_encoded = features_dict[FEATURES_NUM_ENCODED_KEY]
        if FEATURES_NUM_BIN_PROFILE_KEY in features_dict:
            self.features_bin_profile = features_dict[FEATURES_NUM_BIN_PROFILE_KEY]
        if self.features_num_encoded:
            if self.bin_profile is None:
                self.bin_profile = features_dict.get('df_bin_profile', None)

        if self.group_util is None:
            self.group_util = GroupUtil(features_dict.get('group_dict', {}))

        self.use_cat_encoder = features_dict.get('use_cat_encoder', True)
        self.features_cat = features_dict[FEATURES_CAT_KEY]
        self.features_num = features_dict[FEATURES_NUM_KEY]
        self.features_float = features_dict[FEATURES_FLOAT_KEY]
        self.features_num_calculated = features_dict[FEATURES_NUM_CALCULATED_KEY]
        self.features_encoded = [e for e in features_dict[FEATURES_ENCODED_KEY] if
                                 e not in (self.features_num_encoded + self.features_num_calculated)]
        print("self.features_encoded: {}".format(self.features_encoded))
        self.features_cat_and_encoded = self.features_cat + self.features_encoded

        self.feat_grouped = features_dict[FEATURES_GROUPED]
        self.features_grouped_encoded = ["-".join(feat_group) for feat_group in self.feat_grouped]
        self.additional_fields = features_dict[ADDITIONAL_FIELDS_KEY]

        self.txn_hour_group = features_dict.get('txn_hour_group', self.default_txn_hour_group)

        df = self.handle_feat_float(df)
        df = self.handle_feat_encoded(df)
        df = self.handle_feat_num_encoded(df)


        df = df.reset_index()
        print("self.features_all: ", self.features_all)
        print('In fit, self.features_cat: {}'.format(self.features_cat))
        df[self.features_cat_and_encoded] = df[self.features_cat_and_encoded].astype(str).apply(
            lambda x: x.str.lower().replace(' ', '', regex=True).replace("nodatafound',value:'n/a", "nan",
                                                                         regex=False).replace("nodatafound", "nan",
                                                                                              regex=False))

        self.encode_feat_grouped(df)
        self.features_all = self.features_cat + self.features_grouped_encoded + self.features_num + self.features_num_encoded + \
                            self.features_num_calculated

        time_start = time.time()
        #         te = EnhancedTargetEncoder(cols=self.features_cat_and_encoded, handle_unknown='impute', min_samples_leaf=25, impute_missing=True)
        te = EnhancedLeaveOneOutEncoder(cols=self.features_cat + self.features_grouped_encoded,
                                        handle_unknown='impute', impute_missing=True)
        print(self.features_cat)

        if self.use_cat_encoder:
            self.encoder = te.fit(df[self.features_cat + self.features_grouped_encoded], y)
            fit_time = time.time() - time_start
            print("# fit_time:", fit_time)
            print('In fit, self.encoder: ')
            print(self.encoder)
        else:
            print("# not using cat encoder")

        # Fit a scaler
        df_num = df[self.features_num + self.features_num_encoded + self.features_num_calculated].astype(float)
        self.fillna_val = df_num.mean()  # .median()
        # df_num = df_num.fillna(self.fillna_val)
        # if not df_num.empty:
        #     self.scaler = preprocessing.StandardScaler().fit(df_num.fillna(self.fillna_val))

        return self


from sklearn.pipeline import Pipeline


class EnhancedPipeline(Pipeline):

    def fit_predict(self, X, y=None, **fit_params):
        print('In EnhancedPipeline fit_predict ...')

        last_step = self._final_estimator
        Xt, fit_params = self._fit(X, y, **fit_params)
        if hasattr(last_step, 'fit_transform'):
            return last_step.fit_predict(Xt, y, **fit_params)
        else:
            return last_step.fit(Xt, y, **fit_params).predict(Xt)
        
    def predict_proba(self, X):
        """Apply transforms, and predict_proba of the final estimator"""
        Xt = X
        for name, transform in self.steps[:-1]:
            if transform is not None:
                Xt = transform.transform(Xt)
        return self.steps[-1][-1].predict_proba(Xt, thread_count=1)


def make_pipeline(*steps, **kwargs):
    """Construct a Pipeline from the given estimators.

    This is a shorthand for the Pipeline constructor; it does not require, and
    does not permit, naming the estimators. Instead, their names will be set
    to the lowercase of their types automatically.

    Parameters
    ----------
    *steps : list of estimators,

    memory : None, str or object with the joblib.Memory interface, optional
        Used to cache the fitted transformers of the pipeline. By default,
        no caching is performed. If a string is given, it is the path to
        the caching directory. Enabling caching triggers a clone of
        the transformers before fitting. Therefore, the transformer
        instance given to the pipeline cannot be inspected
        directly. Use the attribute ``named_steps`` or ``steps`` to
        inspect estimators within the pipeline. Caching the
        transformers is advantageous when fitting is time consuming.

    Pipeline(memory=None,
             steps=[('standardscaler',
                     StandardScaler(copy=True, with_mean=True, with_std=True)),
                    ('gaussiannb', GaussianNB(priors=None))])

    Returns
    -------
    p : Pipeline
    """
    memory = kwargs.pop('memory', None)
    if kwargs:
        raise TypeError('Unknown keyword arguments: "{}"'
                        .format(list(kwargs.keys())[0]))
    from sklearn.pipeline import _name_estimators
    print('Best Retry preprocessing pipeline ... ')
    return EnhancedPipeline(_name_estimators(steps), memory=memory)
