from src.web.utils import *
import numpy as np

PAY_AMOUNT_USD = "payment_amount_usd"

MEAN_DIFF = "mean_diff"
MEDIAN_DIFF = "median_diff"
MAX_95_DIFF = "max_95_diff"
MAX_99_DIFF = "max_99_diff"
STD_DIFF = "std_diff"
FEATURES_NUM_CALCULATED_KEY = "FEATURES_NUM_CALCULATED"
FEATURES_FLOAT_KEY = "FEATURES_FLOAT"

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


class DeclineTypeUtil:
    """A util to group decline type based on provided response message"""

    DECLINE_TEXT = 'DECLINE_TEXT'
    DECLINE_TYPE = 'DECLINE_TYPE'
    BASE = 'Base'

    def __init__(self, df_decline_type):
        self._df_decline_type = df_decline_type
        self._msg_group = {
            'do_not_honor': 'do not honor',
            'attempt_lower_amount': 'lower amount',
            'Insufficient Funds': ['insufficient', 'balance'],
            'correct_cc_retry': 'correct card',
            'invalid_cc': 'invalid card',
            'lost_stolen': 'lost or stolen',
            'invalid_account': 'invalid account',
            'do_not_try_merchant_review': 'do not try again/merchant review',
            'expired_card': 'expired',
            'pickup_card': 'pick',
            'blocked_first_used': 'blocked',
            'invalid_txn': 'invalid trans',
            'restricted_card': 'restricted',
            'not_permitted': 'not permitted',
            'expired card': 'expired card',
            'unable to determine format': 'determine format',
            'system error': 'system error',
            'no reply': 'no reply',
            'no charge model found': 'no charge model found',
            'issuer unavailable': 'issuer unavailable',
            'litle http response code': 'litle http response code',
            'ioexception': 'ioexception',
            'invalid merchant': 'invalid merchant',
            'international filtering': 'international filtering',
            'corrupt input data': 'corrupt input data',
            'server error': 'server error',
            'acquirer error': 'acquirer error',
            'transaction refused[30]': 'transaction refused[30]',
            'transaction refused[002]': 'transaction refused[002]',
            'txn_refused': 'refuse',
            'declined non generic': 'declined non generic',
            'declined': 'decline',
            'transaction not allowed at terminal': 'transaction not allowed at terminal',
            'error validating xml data': 'error validating xml data',
            'communication problems': 'communication problems',
            'new account info': 'new account info',
            'unable to connect to gateway': 'unable to connect to gateway'
        }

    def __group_response_msg(self, msg):
        """Groups decline type based on the given message"""
        other = self.BASE

        if not isinstance(msg, str):
            return other

        msg_lower = msg.lower()
        for key, val in self._msg_group.items():
            if isinstance(val, list):
                for msg in val:
                    if msg in msg_lower:
                        return key
            else:
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

    def date_bin(self, row):
        result = self.group_dict.get("date_bin_dict", {}).get((str(row['bin']), row['day_of_month']), None)
        return result

    def date_mid_bin(self, row):
        result = self.group_dict.get("date_mid_bin_dict", {}).get(
            (str(row['bin']), str(row['merchant_number']), row['day_of_month']), None)
        return result

    def processor_mid_bin(self, row):
        result = self.group_dict.get("processor_mid_bin_dict", {}).get(
            (str(row['bin']), str(row['payment_service_id']), str(row['merchant_number'])), None)
        return result

    def date_decline_type_bin(self, row):
        result = self.group_dict.get("date_decline_type_bin_dict", {}).get(
            (str(row['bin']), str(row['failed_decline_type']), row['day_of_month']), None)
        return result

    def day_weekmonth_bin(self, row):
        result = self.group_dict.get("day_weekmonth_bin_dict", {}).get(
            (str(row['bin']), int(row['week_of_month']), int(row['day_of_week'])), None)
        return result

    def date_max_diff(self, row):
        result = 0
        date_max = self.group_dict.get("max_per_date_dict", {}).get((str(row['bin']), row['day_of_month']), None)
        if isinstance(date_max, float) or isinstance(date_max, int):
            result = date_max - float(row[PAY_AMOUNT_USD])
        return result

    def date_max(self, row):
        date_max = self.group_dict.get("max_per_date_dict", {}).get((str(row['bin']), row['day_of_month']), None)
        return date_max

    def date_mean_diff(self, row):
        result = 0
        date_mean = self.group_dict.get("mean_per_date_dict", {}).get((str(row['bin']), row['day_of_month']), None)
        if isinstance(date_mean, float) or isinstance(date_mean, int):
            result = date_mean - float(row[PAY_AMOUNT_USD])
        return result

    def date_country(self, row):
        result = self.group_dict.get("date_country_dict", {}).get((str(row['issuer_country']), row['day_of_month']),
                                                                  None)
        return result

    def date_processor_bin(self, row):
        result = self.group_dict.get("date_processor_bin_dict", {}).get(
            (str(row['bin']), str(row['payment_service_id']), row['day_of_month']), None)
        return result

    def days_between_decline_type(self, row):
        result = self.group_dict.get("days_between_decline_type_dict", {}).get(
            (row['failed_decline_type'], row['days_between']), None)
        return result

    def days_between_bin_decline_type(self, row):
        result = self.group_dict.get("days_between_bin_decline_type_dict", {}).get(
            (row['failed_decline_type'], str(row['bin']), row['days_between']), None)
        return result

    def days_between_date_bin_decline_type(self, row):
        result = self.group_dict.get("days_between_date_bin_decline_type_dict", {}).get(
            (row['failed_decline_type'], str(row['bin']), row['day_of_month'], row['days_between']), None)
        return result

    def days_between_bin_failed_message(self, row):
        failed_message = str(row['failed_response_message']).lower().replace(' ', '')
        result = self.group_dict.get("days_between_bin_failed_message_dict", {}).get(
            (failed_message, str(row['bin']), row['days_between']), None)
        return result

    def days_between_country_decline_type(self, row):
        result = self.group_dict.get("days_between_country_decline_type_dict", {}).get(
            (row['failed_decline_type'], row['issuer_country'], row['days_between']), None)
        return result

    def days_between_bin_decline_code(self, row):
        result = self.group_dict.get("days_between_bin_decline_code_dict", {}).get(
            (str(row['failed_response_code']), str(row['bin']), row['days_between']), None)

        return result

    def days_between_site_decline_type(self, row):
        result = self.group_dict.get("days_between_site_decline_type_dict", {}).get(
            (row['failed_decline_type'], row['site_id'], row['days_between']), None)

        return result

    def days_between_site_decline_code(self, row):
        result = self.group_dict.get("days_between_site_decline_code_dict", {}).get(
            (str(row['failed_response_code']), row['site_id'], row['days_between']), None)

        return result

    def date_site_decline_type(self, row):
        result = self.group_dict.get("date_site_decline_type_dict", {}).get(
            (row['failed_decline_type'], row['site_id'], row['day_of_month']), None)

        return result

    def date_site_decline_code(self, row):
        result = self.group_dict.get("date_site_decline_code_dict", {}).get(
            (str(row['failed_response_code']), row['site_id'], row['day_of_month']), None)

        return result

    def date_country_decline_type(self, row):
        result = self.group_dict.get("date_country_decline_type_dict", {}).get(
            (row['failed_decline_type'], row['issuer_country'], row['day_of_month']), None)

        return result

    def date_country_decline_code(self, row):
        result = self.group_dict.get("date_country_decline_code_dict", {}).get(
            (str(row['failed_response_code']), row['issuer_country'], row['day_of_month']), None)
        return result

    def days_between_att_num_decline_type(self, row):
        result = self.group_dict.get("days_between_att_num_decline_type_dict", {}).get(
            (row['failed_decline_type'], row['renew_att_num'], row['days_between']), None)
        return result

    def days_between_country_site(self, row):
        result = self.group_dict.get("days_between_country_site_dict", {}).get(
            (row['site_id'], row['issuer_country'], row['days_between']), None)
        return result

    def date_country_site(self, row):
        result = self.group_dict.get("date_country_site_dict", {}).get(
            (row['site_id'], row['issuer_country'], row['day_of_month']), None)
        return result

    def nod_days_between_decline_type(self, row):
        result = self.group_dict.get("nod_days_between_decline_type_dict", {}).get(
            (row['failed_decline_type'], row['days_between'], row['num_of_days']), None)
        return result

    def nod_days_between_decline_code(self, row):
        result = self.group_dict.get("nod_days_between_decline_code_dict", {}).get(
            (row['failed_response_code'], row['days_between'], row['num_of_days']), None)
        return result

    def date_nod_bin(self, row):
        result = self.group_dict.get("date_nod_bin_dict", {}).get(
            (row['bin'], row['num_of_days'], row['day_of_month']), None)
        return result


class EcoBinUtil:
    def __init__(self, date_increment_bin_dict, added_years_bin_dict):
        self.date_increment_bin_dict = date_increment_bin_dict
        self.added_years_bin_dict = added_years_bin_dict

    def date_inc_bin(self, row):
        result = self.date_increment_bin_dict.get((str(row['bin']), str(row['date_increment'])), -1)
        return result

    def added_years_bin(self, row):
        result = self.added_years_bin_dict.get((str(row['bin']), str(row['added_expiry_years'])), -1)
        return result


# class PaymentMidBinUtil:
#     def __init__(self, payment_mid_bin_dict):
#         self.payment_mid_bin_dict = payment_mid_bin_dict

#     def payment_mid_bin(self, row):
#         result = self.payment_mid_bin_dict.get((row['bin'], row['payment_service_id'], row['merchant_number']), -1)
#         return result

class TxnHourUtil:
    #     def __init__(self, txn_hour_bin_dict, txn_hour_processor_dict, txn_hour_country_dict, txn_hour_site_dict):
    #         self.txn_hour_bin_dict = txn_hour_bin_dict
    #         self.txn_hour_processor_dict = txn_hour_processor_dict
    #         self.txn_hour_country_dict = txn_hour_country_dict
    #         self.txn_hour_site_dict = txn_hour_site_dict
    def __init__(self, group_dict):
        self.group_dict = group_dict

    def txn_hour_bin(self, row):
        #         result = self.txn_hour_bin_dict.get((str(row['bin']), str(row['transaction_hour'])), -1)
        result = self.group_dict.get("txn_hour_bin_dict", {}).get((str(row['bin']), str(row['transaction_hour'])), -1)

        return result

    def txn_hour_processor(self, row):
        #         result = self.txn_hour_processor_dict.get((str(row['payment_service_id']), str(row['transaction_hour'])), -1)
        result = self.group_dict.get("txn_hour_processor_dict", {}).get(
            (str(row['payment_service_id']), str(row['transaction_hour'])), -1)
        return result

    def txn_hour_country(self, row):
        #         result = self.txn_hour_country_dict.get((str(row['issuer_country']), str(row['transaction_hour'])), -1)
        result = self.group_dict.get("txn_hour_country_dict", {}).get(
            (str(row['issuer_country']), str(row['transaction_hour'])), -1)
        return result

    def txn_hour_site(self, row):
        #         result = self.txn_hour_site_dict.get((str(row['site_id']), str(row['transaction_hour'])), -1)
        result = self.group_dict.get("txn_hour_site_dict", {}).get((str(row['site_id']), str(row['transaction_hour'])),
                                                                   -1)
        return result

    def txn_hour_bank_name(self, row):
        bank_name = str(row['bank_name']).lower().replace(' ', '')
        result = self.group_dict.get("txn_hour_bank_name_dict", {}).get((bank_name, str(row['transaction_hour'])), -1)
        return result

    def txn_hour_currency(self, row):
        result = self.group_dict.get("txn_hour_currency_dict", {}).get(
            (str(row['payment_currency']), str(row['transaction_hour'])), -1)
        return result

    def txn_hour_mid(self, row):
        result = self.group_dict.get("txn_hour_mid_dict", {}).get(
            (str(row['merchant_number']), str(row['transaction_hour'])), -1)
        return result

    def txn_hour_date(self, row):
        result = self.group_dict.get("txn_hour_date_dict", {}).get(
            (str(row['day_of_month']), str(row['transaction_hour'])), -1)
        return result

    def bin_hour_max_diff(self, row):
        result = 0
        hour_max = self.group_dict.get("bin_max_amt_per_hour_dict", {}).get(
            (str(row['bin']), str(row['transaction_hour'])), 0)
        if isinstance(hour_max, float) or isinstance(hour_max, int):
            result = hour_max - float(row[PAY_AMOUNT_USD])
        return result

    def processor_hour_max_diff(self, row):
        result = 0
        processor_max = self.group_dict.get("processor_max_amt_per_hour_dict", {}).get(
            (str(row['payment_service_id']), str(row['transaction_hour'])), 0)
        if isinstance(processor_max, float) or isinstance(processor_max, int):
            result = processor_max - float(row[PAY_AMOUNT_USD])
        return result


class PreProcessing(BaseEstimator, TransformerMixin):
    """Custom Pre-Processing estimator for Best Retry
    """

    def __init__(self, df_bin_profile):
        self.df_decline_type = None
        self.bin_profile = None
        self.eco_bin_util = None
        self.txn_hour_util = None
        self.group_util = None

        if df_bin_profile is None:
            print('df_bin_profile is NONE')
        else:
            self.bin_profile = df_bin_profile
            self.bin_profile[BIN] = self.bin_profile[BIN].astype(str).str.replace('.0', '', regex=False)
            print('# df_bin_profile: {}'.format(df_bin_profile.shape))

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
        self.use_cat_encoder = True

        pass

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

    def handle_feat_grouped(self, df):
        self.features_grouped_encoded = []
        for feat_group in self.feat_grouped:
            group_name = "-".join(feat_group)
            self.features_grouped_encoded.append(group_name)
            df[group_name] = df[feat_group].fillna('').astype(str).apply(lambda x: '-'.join(x), axis=1)

    def handle_feat_encoded(self, df):
        if WEEK_OF_MONTH in self.features_encoded:
            df[WEEK_OF_MONTH] = df[TXN_DATE_IN_STR].apply(week_of_month)

        if DAY_OF_WEEK in self.features_encoded:
            df[DAY_OF_WEEK] = df[TXN_DATE_IN_STR].apply(to_weekday)

        if 'cc_expiration_date' in df.columns:
            df[IS_EXPIRED] = df[~df['cc_expiration_date'].isna()].apply(is_expired, axis=1)

        if IS_WEEKEND in self.features_cat_and_encoded:
            if DAY_OF_WEEK not in self.features_encoded:
                df[DAY_OF_WEEK] = df[TXN_DATE_IN_STR].apply(to_weekday)

            df[IS_WEEKEND] = df[DAY_OF_WEEK].apply(is_weekend)

        if FAILED_ATTEMPT_DATE in df.columns:
            df[DAYS_BETWEEN] = df.apply(days_between_ds, axis=1)

        # if 'num_of_days' in self.features_cat_and_encoded:
        df['num_of_days'] = df[TXN_DATE_IN_STR].apply(num_of_days)

        if FAILED_DAY_OF_MONTH in self.features_cat_and_encoded:
            df[FAILED_DAY_OF_MONTH] = df[FAILED_ATTEMPT_DATE].apply(to_day)

        if MONTH in self.features_cat_and_encoded:
            df[MONTH] = df[TXN_DATE_IN_STR].apply(to_month)

        if FAILED_DECLINE_TYPE in self.features_cat_and_encoded and FAILED_DECLINE_TYPE not in df.columns:
            df[FAILED_DECLINE_TYPE] = df[FAILED_RESPONSE_MESSAGE].apply(
                DeclineTypeUtil(self.df_decline_type).decline_type)

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

        if "issuer_country" in self.features_cat_and_encoded and "billing_country" in df.columns:
            df["issuer_country"] = df["issuer_country"].replace('', np.nan).fillna(df["billing_country"])

        if "bank_code" in self.features_cat_and_encoded and "bank_code" in df.columns:
            bank_code = df["bank_code"]
            df.loc[(df.bank_code.str.lower() != "non3ds") & (df.bank_code.str.lower() != "rb"), 'bank_code'] = 'other'

        if "date_increment" in self.features_cat_and_encoded:
            df["date_increment"] = df["date_increment"].replace('', np.nan).fillna('NONE')
            df.loc[df.date_increment == 'nan', 'date_increment'] = 'NONE'

        print("# Finish handle_feat_encoded.")
        return df

    def handle_feat_num_encoded(self, df):
        if BIN in df.columns:
            df[BIN] = pd.to_numeric(df[BIN], errors='coerce')
            df[BIN] = df[BIN].astype(str).str.replace('.0', '', regex=False)

            # #             drops self.features_num_encoded from df if they exist and have null value
            #             if set(self.features_num_encoded).issubset(df.columns) and df[self.features_num_encoded].isnull().values.any():
            #                 df = df.drop(self.features_num_encoded, axis=1)

            #             if MEAN in self.features_num_encoded and MEAN not in df.columns and self.features_bin_profile is not None:
            #                 df[BIN] = df[BIN].astype(str).str.replace('.0', '', regex=False)
            #                 df = pd.merge(df, self.bin_profile[[BIN] + self.features_bin_profile], left_on=BIN, right_on=BIN, how='left')

            if "Max_99" in self.features_num_encoded and "Max_99" not in df.columns and self.features_bin_profile is not None:
                df[BIN] = df[BIN].astype(str).str.replace('.0', '', regex=False)
                df = pd.merge(df, self.bin_profile[[BIN] + self.features_bin_profile], left_on=BIN, right_on=BIN,
                              how='left')

            #             if 'amount_over_max' in self.features_num_calculated and "Max_99" in self.features_num_encoded and "payment_amount_usd" in self.features_num:
            #                 df['amount_over_max'] = df["payment_amount_usd"] > df["Max_99"]

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

        if "txn_hour_bin" in self.features_num_encoded:
            print("# apply txn_hour_bin")
            df["txn_hour_bin"] = df.apply(self.txn_hour_util.txn_hour_bin, axis=1)

        if "txn_hour_processor" in self.features_num_encoded:
            df["txn_hour_processor"] = df.apply(self.txn_hour_util.txn_hour_processor, axis=1)

        if "txn_hour_country" in self.features_num_encoded:
            df["txn_hour_country"] = df.apply(self.txn_hour_util.txn_hour_country, axis=1)

        if "txn_hour_site" in self.features_num_encoded:
            df["txn_hour_site"] = df.apply(self.txn_hour_util.txn_hour_site, axis=1)

        if "txn_hour_bank_name" in self.features_num_encoded:
            df["txn_hour_bank_name"] = df.apply(self.txn_hour_util.txn_hour_bank_name, axis=1)

        if "txn_hour_currency" in self.features_num_encoded:
            df["txn_hour_currency"] = df.apply(self.txn_hour_util.txn_hour_currency, axis=1)

        if "txn_hour_mid" in self.features_num_encoded:
            df["txn_hour_mid"] = df.apply(self.txn_hour_util.txn_hour_mid, axis=1)

        if "txn_hour_date" in self.features_num_encoded:
            df["txn_hour_date"] = df.apply(self.txn_hour_util.txn_hour_date, axis=1)

        if "bin_max_diff_per_hour" in self.features_num_encoded:
            df["bin_max_diff_per_hour"] = df.apply(self.txn_hour_util.bin_hour_max_diff, axis=1)

        if "processor_max_diff_per_hour" in self.features_num_encoded:
            df["processor_max_diff_per_hour"] = df.apply(self.txn_hour_util.processor_hour_max_diff, axis=1)

        if "date_bin" in self.features_num_encoded:
            df["date_bin"] = df.apply(self.group_util.date_bin, axis=1)

        if "date_mid_bin" in self.features_num_encoded:
            df["date_mid_bin"] = df.apply(self.group_util.date_mid_bin, axis=1)

        if "processor_mid_bin" in self.features_num_encoded:
            df["processor_mid_bin"] = df.apply(self.group_util.processor_mid_bin, axis=1)

        if "date_decline_type_bin" in self.features_num_encoded:
            df["date_decline_type_bin"] = df.apply(self.group_util.date_decline_type_bin, axis=1)

        if "day_weekmonth_bin" in self.features_num_encoded:
            df["day_weekmonth_bin"] = df.apply(self.group_util.day_weekmonth_bin, axis=1)

        if "date_max_diff" in self.features_num_encoded:
            df["date_max_diff"] = df.apply(self.group_util.date_max_diff, axis=1)

        if "date_max" in self.features_num_encoded:
            df["date_max"] = df.apply(self.group_util.date_max, axis=1)

        if "date_mean_diff" in self.features_num_encoded:
            print("# apply date_mean_diff")
            df["date_mean_diff"] = df.apply(self.group_util.date_mean_diff, axis=1)

        if "date_country" in self.features_num_encoded:
            print("# apply date_country")
            df["date_country"] = df.apply(self.group_util.date_country, axis=1)

        if "date_processor_bin" in self.features_num_encoded:
            df["date_processor_bin"] = df.apply(self.group_util.date_processor_bin, axis=1)

        if "days_between_decline_type" in self.features_num_encoded:
            df["days_between_decline_type"] = df.apply(self.group_util.days_between_decline_type, axis=1)

        if "days_between_bin_decline_type" in self.features_num_encoded:
            df["days_between_bin_decline_type"] = df.apply(self.group_util.days_between_bin_decline_type, axis=1)

        if "days_between_date_bin_decline_type" in self.features_num_encoded:
            df["days_between_date_bin_decline_type"] = df.apply(self.group_util.days_between_date_bin_decline_type,
                                                                axis=1)

        if "days_between_bin_failed_message" in self.features_num_encoded:
            df["days_between_bin_failed_message"] = df.apply(self.group_util.days_between_bin_failed_message, axis=1)

        if "days_between_country_decline_type" in self.features_num_encoded:
            df["days_between_country_decline_type"] = df.apply(self.group_util.days_between_country_decline_type,
                                                               axis=1)

        if "days_between_bin_decline_code" in self.features_num_encoded:
            df["days_between_bin_decline_code"] = df.apply(self.group_util.days_between_bin_decline_code, axis=1)

        if "days_between_site_decline_type" in self.features_num_encoded:
            df["days_between_site_decline_type"] = df.apply(self.group_util.days_between_site_decline_type, axis=1)

        if "days_between_site_decline_code" in self.features_num_encoded:
            df["days_between_site_decline_code"] = df.apply(self.group_util.days_between_site_decline_code, axis=1)

        if "date_site_decline_type" in self.features_num_encoded:
            df["date_site_decline_type"] = df.apply(self.group_util.date_site_decline_type, axis=1)

        if "date_site_decline_code" in self.features_num_encoded:
            df["date_site_decline_code"] = df.apply(self.group_util.date_site_decline_code, axis=1)

        if "date_country_decline_type" in self.features_num_encoded:
            df["date_country_decline_type"] = df.apply(self.group_util.date_country_decline_type, axis=1)

        if "date_country_decline_code" in self.features_num_encoded:
            df["date_country_decline_code"] = df.apply(self.group_util.date_country_decline_code, axis=1)

        if "days_between_att_num_decline_type" in self.features_num_encoded:
            df["days_between_att_num_decline_type"] = df.apply(self.group_util.days_between_att_num_decline_type,
                                                               axis=1)

        if "days_between_country_site" in self.features_num_encoded:
            df["days_between_country_site"] = df.apply(self.group_util.days_between_country_site, axis=1)

        if "date_country_site" in self.features_num_encoded:
            df["date_country_site"] = df.apply(self.group_util.date_country_site, axis=1)

        if "nod_days_between_decline_type" in self.features_num_encoded:
            df["nod_days_between_decline_type"] = df.apply(self.group_util.nod_days_between_decline_type, axis=1)

        if "nod_days_between_decline_code" in self.features_num_encoded:
            df["nod_days_between_decline_code"] = df.apply(self.group_util.nod_days_between_decline_code, axis=1)

        if "date_nod_bin" in self.features_num_encoded:
            df["date_nod_bin"] = df.apply(self.group_util.date_nod_bin, axis=1)

        df = self.handle_feat_cyclical(df)

        return df

    def convert_mid(self, row):
        mid = row['merchant_number']
        if row['payment_service_id'] is not None and (
                row['payment_service_id'].startswith('netgiro-') or row['payment_service_id'].startswith('drwp-')):
            mid = mid.split('-')[0]
        #             mid = mid + "-" + row['payment_currency'].upper() + "-pacific"

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

    def encode(self, df):
        # Consolidated feature processing
        df_encoded_all = pd.DataFrame(columns=self.features_all)
        df = self.handle_mid(df)
        df = self.handle_feat_float(df)
        df = self.handle_feat_encoded(df)
        df = self.handle_feat_num_encoded(df)
        df = df.reset_index()
        df[self.features_cat_and_encoded] = df[self.features_cat_and_encoded].fillna('nan')
        df[self.features_cat_and_encoded] = df[self.features_cat_and_encoded].astype(str).apply(
            lambda x: x.str.lower().replace(' ', '', regex=True).replace("nodatafound',value:'n/a", "nan",
                                                                         regex=False).replace("nodatafound", "nan",
                                                                                              regex=False))
        self.handle_feat_grouped(df)
        time_start = time.time()

        if self.use_cat_encoder:
            df_encoded_cat = self.encoder.transform(df[self.features_cat_and_encoded + self.features_grouped_encoded])
        else:
            df_encoded_cat = df[self.features_cat_and_encoded + self.features_grouped_encoded]

        transform_time = time.time() - time_start
        print("# transform_time:", transform_time)

        df_encoded_all[self.features_cat_and_encoded + self.features_grouped_encoded] = df_encoded_cat

        # Num processing
        df_num = df[self.features_num + self.features_num_encoded + self.features_num_calculated].astype(float)
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
        self.features_num_encoded = features_dict[FEATURES_NUM_ENCODED_KEY]
        if FEATURES_NUM_BIN_PROFILE_KEY in features_dict:
            self.features_bin_profile = features_dict[FEATURES_NUM_BIN_PROFILE_KEY]
        if self.features_num_encoded:
            if self.bin_profile is None:
                self.bin_profile = features_dict.get('df_bin_profile', None)

        if self.group_util is None:
            self.group_util = GroupUtil(features_dict.get('group_dict', {}))

        if self.eco_bin_util is None:
            self.eco_bin_util = EcoBinUtil(features_dict.get('date_increment_bin_dict', {}),
                                           features_dict.get('added_years_bin_dict', {}))

        if self.txn_hour_util is None:
            self.txn_hour_util = TxnHourUtil(features_dict.get('txn_hour_group_dict', {}))

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
        self.features_grouped_encoded = []
        if FAILED_DECLINE_TYPE in self.features_cat_and_encoded:
            if self.df_decline_type is None:
                self.df_decline_type = features_dict['df_decline_type']
                print('In fit, self.df_decline_type: {}'.format(self.df_decline_type.shape))

            self.decline_type_util = DeclineTypeUtil(self.df_decline_type)

        df = self.handle_feat_float(df)
        df = self.handle_feat_encoded(df)
        df = self.handle_feat_num_encoded(df)
        df = df.reset_index()
        print("self.features_all: ", self.features_all)
        print('In fit, self.features_cat_and_encoded: {}'.format(self.features_cat_and_encoded))
        df[self.features_cat_and_encoded] = df[self.features_cat_and_encoded].astype(str).apply(
            lambda x: x.str.lower().replace(' ', '', regex=True).replace("nodatafound',value:'n/a", "nan",
                                                                         regex=False).replace("nodatafound", "nan",
                                                                                              regex=False))

        self.handle_feat_grouped(df)
        self.features_all = self.features_cat_and_encoded + self.features_grouped_encoded + self.features_num + self.features_num_encoded  + self.features_num_calculated

        time_start = time.time()
        #         te = EnhancedTargetEncoder(cols=self.features_cat_and_encoded, handle_unknown='impute', min_samples_leaf=25, impute_missing=True)
        te = EnhancedLeaveOneOutEncoder(cols=self.features_cat_and_encoded + self.features_grouped_encoded,
                                        handle_unknown='impute', impute_missing=True)
        print(self.features_cat_and_encoded)
        print("fit df[self.features_cat_and_encoded] size: {}".format(
            df[self.features_cat_and_encoded + self.features_grouped_encoded].shape))

        if self.use_cat_encoder:
            self.encoder = te.fit(df[self.features_cat_and_encoded + self.features_grouped_encoded], y)
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
