def setup(args):
    try:
        mod = __import__('.'.join(['models', args.model]), fromlist=['Model'])
        model = getattr(mod, 'Model')(args)
    except:
        raise Exception("Model not supported: {}".format(args.model))

    return model

