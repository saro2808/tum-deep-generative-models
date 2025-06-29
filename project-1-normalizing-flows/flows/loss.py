def likelihood(X_train, model, device):
    ##########################################################
    model.to(device)
    X_train = X_train.to(device)
    log_prob = model.log_prob(X_train)
    loss = -log_prob.mean()
    ##########################################################

    return loss
