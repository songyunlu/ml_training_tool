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
FEATURES_ENCODED_KEY = 'FEATURES_ENCODED'

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
             'do_not_honor' : 'do not honor', 
             'attempt_lower_amount' : 'lower amount',
            'Insufficient Funds' : 'insufficient',
            'correct_cc_retry' : 'correct card',
            'invalid_cc' : 'invalid card',
            'lost_stolen' : 'lost or stolen',
            'invalid_account' : 'invalid account',
            'do_not_try_merchant_review' : 'do not try again/merchant review',
            'expired_card' : 'expired',
            'pickup_card' : 'pick',
            'blocked_first_used' : 'blocked',
            'invalid_txn' : 'invalid trans',
            'restricted_card' : 'restricted',
            'not_permitted' : 'not permitted',
            'expired card' : 'expired card',
            'unable to determine format' : 'determine format',
            'system error' : 'system error',
            'no reply' : 'no reply',
             'no charge model found' : 'no charge model found',
             'issuer unavailable' : 'issuer unavailable',
             'litle http response code' : 'litle http response code',
             'ioexception' : 'ioexception',
             'invalid merchant' : 'invalid merchant',
             'international filtering':'international filtering',
             'corrupt input data':'corrupt input data',
             'server error' : 'server error',
             'acquirer error' : 'acquirer error',
             'transaction refused[30]':'transaction refused[30]',
             'transaction refused[002]':'transaction refused[002]',
             'txn_refused' : 'refuse',
             'declined non generic': 'declined non generic',
             'declined' : 'decline', 
             'transaction not allowed at terminal': 'transaction not allowed at terminal',
             'error validating xml data': 'error validating xml data',
             'communication problems': 'communication problems',
             'new account info': 'new account info',
             'unable to connect to gateway': 'unable to connect to gateway',
             '':''
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
    datestr = str(datestr).replace('/','')
    struct = time.strptime(datestr, "%m%y")
    original_year = struct.tm_year
    txn_year = time.strptime(row['transaction_date_in_string'], '%Y-%m-%d %H:%M:%S').tm_year
    return abs(txn_year - original_year)

def years_over(row):
    try:
        date_increment = row['date_increment']
        expiration_years_diff = expired_years_diff(row)

        if int(date_increment) < 1:
            return -expiration_years_diff

        total_increment = date_increment
        while total_increment < expiration_years_diff:
            total_increment += date_increment

        if total_increment == date_increment:
            credict_card_month = cc_month(row['cc_expiration_date'])
            txn_month = to_month(row['transaction_date_in_string'])
            if credict_card_month < txn_month:    
                total_increment += date_increment
    except:
        return -expiration_years_diff

    
class EcoBinUtil:
    def __init__(self, date_increment_bin_dict, added_years_bin_dict):
        self.date_increment_bin_dict = date_increment_bin_dict
        self.added_years_bin_dict = added_years_bin_dict
        
#     def eco_bin_value(self, row, field, prefix):
#         try:
#             bin_row = self.eco_bin[self.eco_bin['bin'] == row['bin']]
#             value = bin_row[prefix + row[field]].iloc[0]
#             if np.isnan(value):
#                 return -1
#             return value
#         except Exception as ex:
#             return -1

    def date_inc_bin(self, row):
        result = self.date_increment_bin_dict.get((row['bin'], row['date_increment']), -1) 
        return result
    
    def added_years_bin(self, row):
        result = self.added_years_bin_dict.get((row['bin'], row['added_expiry_years']), -1) 
        return result
    
class PaymentMidBinUtil:
    def __init__(self, payment_mid_bin_dict):
        self.payment_mid_bin_dict = payment_mid_bin_dict
        
    def payment_mid_bin(self, row):
        result = self.payment_mid_bin_dict.get((row['bin'], row['payment_service_id'], row['merchant_number']), -1) 
        return result
    

class PreProcessing(BaseEstimator, TransformerMixin):
    """Custom Pre-Processing estimator for Best Retry
    """

    def __init__(self, df_bin_profile):
        self.df_decline_type = None
        self.bin_profile = None
        self.eco_bin_util = None
        self.payment_mid_bin_util = None

#         self.date_increment_bin_dict = None
#         self.added_years_bin_dict = None

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
        self.features_num_encoded =  None
        self.features_num_calculated = None
        self.features_float = None
        self.features_all = None
        self.encoder = None
        self.scaler = None

        pass
    
    
    def handle_feat_float(self, df):
        for feat in self.features_float:
            if feat in self.features_cat:
                df[feat] = df[feat].fillna('').astype(str).str.replace('.0', '', regex=False)
        return df

    
    def handle_feat_encoded(self, df):
        if WEEK_OF_MONTH in self.features_encoded:
            df[WEEK_OF_MONTH] = df[TXN_DATE_IN_STR].apply(week_of_month)

        if DAY_OF_WEEK in self.features_encoded:
            df[DAY_OF_WEEK] = df[TXN_DATE_IN_STR].apply(to_weekday)

        if IS_EXPIRED in self.features_encoded:
            df[IS_EXPIRED] = df[~df['cc_expiration_date'].isna()].apply(is_expired, axis=1)

        if IS_WEEKEND in self.features_cat_and_encoded:
            if DAY_OF_WEEK not in self.features_encoded:
                df[DAY_OF_WEEK] = df[TXN_DATE_IN_STR].apply(to_weekday)

            df[IS_WEEKEND] = df[DAY_OF_WEEK].apply(is_weekend)

        if DAYS_BETWEEN in self.features_cat_and_encoded:
            df[DAYS_BETWEEN] = df.apply(days_between_ds, axis=1)

        if FAILED_DAY_OF_MONTH in self.features_cat_and_encoded:
            df[FAILED_DAY_OF_MONTH] = df[FAILED_ATTEMPT_DATE].apply(to_day)

        if MONTH in self.features_cat_and_encoded:
            df[MONTH] = df[TXN_DATE_IN_STR].apply(to_month)
            
        if FAILED_DECLINE_TYPE in self.features_cat_and_encoded and FAILED_DECLINE_TYPE not in df.columns:
            df[FAILED_DECLINE_TYPE] = df[FAILED_RESPONSE_MESSAGE].apply(DeclineTypeUtil(self.df_decline_type).decline_type)
 
            
        if 'expiration_date_changed' in self.features_cat_and_encoded and 'expiration_date_changed' not in df.columns:
            failed_cc = df['failed_cc_expiration_date'].replace('/', '')
            current_cc = df['cc_expiration_date'].replace('/', '')
            df['expiration_date_changed'] = (failed_cc != current_cc)
                
        if 'processor_mid_changed' in self.features_cat_and_encoded and 'processor_mid_changed' not in df.columns:        
            df['processor_mid_changed'] = df.apply(processor_mid_changed, axis=1)
            
        if "cc_month" in self.features_cat_and_encoded and "cc_month" not in df.columns:
            df['cc_month'] = df["cc_expiration_date"].apply(cc_month)
        
        if "issuer_country" in self.features_cat_and_encoded and "billing_country" in df.columns:
            df["issuer_country"].fillna(df["billing_country"], inplace=True)
            
        if "bank_code" in self.features_cat_and_encoded and "bank_code" in df.columns:
            bank_code = df["bank_code"]
            df.loc[(df.bank_code.str.lower() != "non3ds") & (df.bank_code.str.lower() != "rb"), 'bank_code'] = 'other'

            
        print("# Finish handle_feat_encoded.")
        
        return df

    def handle_feat_num_encoded(self, df):
        if BIN in self.features_cat and self.features_num_encoded:
            df[BIN] = pd.to_numeric(df[BIN], errors='coerce')
            df[BIN] = df[BIN].astype(str).str.replace('.0', '', regex=False)
            
            # drops self.features_num_encoded from df if they exist and have null value
#             if set(self.features_num_encoded).issubset(df.columns) and df[self.features_num_encoded].isnull().values.any():
#                 df = df.drop(self.features_num_encoded, axis=1)

            if MEAN in self.features_num_calculated and MEAN not in df.columns:
#                 df[BIN] = df[BIN].astype(str).str.replace('.0', '', regex=False)
                df = pd.merge(df, self.bin_profile[[BIN] + self.features_num_encoded], left_on=BIN, right_on=BIN, how='left')
        
        
            if MEAN_DIFF in self.features_num_calculated and MEAN_DIFF not in df.columns:
                df[MEAN_DIFF] = df[MEAN] - df[PAY_AMOUNT_USD]
                
            if MEDIAN_DIFF in self.features_num_calculated and MEDIAN_DIFF not in df.columns:
                df[MEDIAN_DIFF] = df[MEDIAN] - df[PAY_AMOUNT_USD]

            if MAX_95_DIFF in self.features_num_calculated and MAX_95_DIFF not in df.columns:
                df[MAX_95_DIFF] = df['Max_95'] - df[PAY_AMOUNT_USD]

            if MAX_99_DIFF in self.features_num_calculated and MAX_99_DIFF not in df.columns:
                df[MAX_99_DIFF] = df['Max_99'] - df[PAY_AMOUNT_USD]

            if STD_DIFF in self.features_num_calculated and STD_DIFF not in df.columns:
                if MEAN_DIFF not in self.features_num_calculated:
                     df[MEAN_DIFF] = df[MEAN] - df[PAY_AMOUNT_USD]

                df[STD_DIFF] = df[STD_DEV] - abs(df[MEAN_DIFF]) 
        

        if "expired_years_diff" in self.features_num_encoded:
            df["expired_years_diff"] = df.apply(expired_years_diff, axis=1)

        if "years_over" in self.features_num_encoded and 'date_increment' in self.features_cat_and_encoded:
            df["years_over"] = df.apply(years_over, axis=1)

        if "date_inc_bin" in self.features_num_encoded:
            df["date_inc_bin"] = df.apply(self.eco_bin_util.date_inc_bin, axis=1)

        if "add_expiry_years_bin" in self.features_num_encoded:
            if "added_expiry_years" not in df.columns:
                df["added_expiry_years"] = df["expired_years_diff"] + df["years_over"]
                df["added_expiry_years"] =  df["added_expiry_years"].fillna('').astype(str).str.replace('.0', '', regex=False) 

            df["add_expiry_years_bin"] = df.apply(self.eco_bin_util.added_years_bin, axis=1)
        
        if "payment_mid_bin" in self.features_num_encoded:
            df["payment_mid_bin"] = df.apply(self.payment_mid_bin_util.payment_mid_bin, axis=1)
        
        return df

    
    def convert_mid(self, row):
        mid = row['merchant_number']
        if row['payment_service_id'].startswith('netgiro-') or row['payment_service_id'].startswith('drwp-'):
            mid = mid.split('-')[0]
            
        return mid
    
   
    def convert_failed_mid(self, row):
        mid = row['failed_merchant_number']
        if row['failed_payment_service_id'].startswith('netgiro-') or row['failed_payment_service_id'].startswith('drwp-'):
            mid = mid.split('-')[0]
            
        return mid
    
    
    def handle_mid(self, df):
        if "merchant_number" in self.features_cat_and_encoded and "payment_service_id" in self.features_cat_and_encoded:
            df["merchant_number"] = df.apply(self.convert_mid, axis=1)
            
        if "failed_merchant_number" in self.features_cat_and_encoded and "failed_payment_service_id" in self.features_cat_and_encoded:
            df["failed_merchant_number"] = df.apply(self.convert_failed_mid, axis=1)
        
        return df
    

    def transform(self, df):
        """Regular transform() that is a help for training, validation & testing datasets
           (NOTE: The operations performed here are the ones that we did prior to this cell)
        """
        # Consolidated feature processing
        df_encoded_all = pd.DataFrame(columns=self.features_all)
        df = self.handle_mid(df)
        df = self.handle_feat_float(df)
        df = self.handle_feat_encoded(df)
        df = self.handle_feat_num_encoded(df)
        df = df.reset_index()
        df[self.features_cat_and_encoded].fillna('nan', inplace=True)
        df[self.features_cat_and_encoded] = df[self.features_cat_and_encoded].astype(str).apply(lambda x: x.str.lower().replace(' ', '', regex=True).replace("nodatafound',value:'n/a", "nan", regex=False).replace("nodatafound", "nan", regex=False))
        
        time_start = time.time()
 
        df_encoded_cat = self.encoder.transform(df[self.features_cat_and_encoded])
        transform_time = time.time() - time_start
        print("# transform_time:", transform_time)
        
        df_encoded_all[self.features_cat_and_encoded] = df_encoded_cat

        # Num processing
        df_num = df[self.features_num + self.features_num_encoded + self.features_num_calculated].astype(float)
        if not df_num.empty:
            df_num = self.scaler.transform(df_num.fillna(0))
        df_encoded_all[self.features_num + self.features_num_encoded + self.features_num_calculated] = df_num
        
        return df_encoded_all.values

    def fit(self, df, y=None, features_dict={}, **fit_params):
        """fit is called only when training, this should not be called when predicting"""
        self.features_num_encoded = features_dict[FEATURES_NUM_ENCODED_KEY]
        if self.features_num_encoded:
            if self.bin_profile is None:
                self.bin_profile = features_dict['df_bin_profile']
        
        if self.eco_bin_util is None:
            self.eco_bin_util = EcoBinUtil(features_dict.get('date_increment_bin_dict', {}), features_dict.get('added_years_bin_dict', {}) )
        if self.payment_mid_bin_util is None:
            self.payment_mid_bin_util = PaymentMidBinUtil(features_dict.get('payment_mid_bin_dict', {}))
            
        
        self.features_cat = features_dict[FEATURES_CAT_KEY]
        self.features_num = features_dict[FEATURES_NUM_KEY]
        self.features_float = features_dict[FEATURES_FLOAT_KEY]
        self.features_num_calculated = features_dict[FEATURES_NUM_CALCULATED_KEY]
        self.features_encoded = [e for e in features_dict[FEATURES_ENCODED_KEY] if e not in (self.features_num_encoded + self.features_num_calculated)]
        print("self.features_encoded: {}".format(self.features_encoded))
        self.features_cat_and_encoded = self.features_cat + self.features_encoded

        if FAILED_DECLINE_TYPE in self.features_cat_and_encoded:
            if self.df_decline_type is None:
                self.df_decline_type = features_dict['df_decline_type']
                print('In fit, self.df_decline_type: {}'.format(self.df_decline_type.shape))

            self.decline_type_util = DeclineTypeUtil(self.df_decline_type)
        
        
        df = self.handle_feat_float(df)
        df = self.handle_feat_encoded(df)
        df = self.handle_feat_num_encoded(df)
        df = df.reset_index()
        self.features_all = self.features_cat_and_encoded + self.features_num + self.features_num_encoded + self.features_num_calculated
        print("self.features_all: ", self.features_all)
        print('In fit, self.features_cat_and_encoded: {}'.format(self.features_cat_and_encoded))
        df[self.features_cat_and_encoded] = df[self.features_cat_and_encoded].astype(str).apply(lambda x: x.str.lower().replace(' ', '', regex=True).replace("nodatafound',value:'n/a", "nan", regex=False).replace("nodatafound", "nan", regex=False))
        time_start = time.time()
#         te = EnhancedTargetEncoder(cols=self.features_cat_and_encoded, handle_unknown='ignore', min_samples_leaf=20, impute_missing=False)
        te = EnhancedLeaveOneOutEncoder(cols=self.features_cat_and_encoded, handle_unknown='impute', impute_missing=True)
        print(self.features_cat_and_encoded)
        print("fit df[self.features_cat_and_encoded] size: {}".format(df[self.features_cat_and_encoded].shape))
        print("fit y size: {}".format(len(y)))
        self.encoder = te.fit(df[self.features_cat_and_encoded], y)
        fit_time = time.time() - time_start
        print("# fit_time:", fit_time)

        print('In fit, self.encoder: ')
        print(self.encoder)

        # Fit a scaler
        df_num = df[self.features_num + self.features_num_encoded + self.features_num_calculated].astype(float)
        if not df_num.empty:
            self.scaler = preprocessing.StandardScaler().fit(df_num.fillna(0))
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