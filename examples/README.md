Importance sampling examples
============================

In this directory we copy some of the Keras examples and apply importance
sampling. The change is minimal and in most cases amounts to the following
code:

    from importance_sampling.training import ImportanceTraining

    ...
    ...

    model.compile(...)

    # instead of model.fit(....)
    ImportanceTraining(model).fit(....)
