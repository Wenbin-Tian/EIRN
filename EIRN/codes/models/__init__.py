import logging
logger = logging.getLogger('base')


def create_model(opt):
    model = opt['model']

    if model == 'IRN':
        from .EIRN_model import IRNModel as M
    elif model == 'IRN+':
        from .IRNp_model import IRNpModel as M
    elif model == 'IRN-CRM':
        from .IRN_model_CRM import IRNCRMModel as M
    elif model == 'IRN-Color':
        from .IRN_color_model import IRNColorModel as M
    else:
        raise NotImplementedError('Model [{:s}] not recognized.'.format(model))
    m = M(opt)
    logger.info('Model [{:s}] is created.'.format(m.__class__.__name__))
    return m
