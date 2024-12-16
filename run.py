from recbole.quick_start import run_recbole

parameter_dict = {
   'train_neg_sample_args': None,
    'USER_ID_FIELD': 'user_id',
    'ITEM_ID_FIELD': 'venue_id',
}
run_recbole(model='SRGNN', dataset='foursquare_NYC', config_dict=parameter_dict)