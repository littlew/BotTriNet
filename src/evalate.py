def print_metrics(y_true, y_pred):
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    f1score = f1_score(y_true, y_pred, average='macro')
    print("accuracy:", accuracy, "precision:", precision, "recall:", recall, "f1score:", f1score)


def predict(train_X, train_y, test_X, test_y, algo='rf'):
    # 作分类器和评估
    #print(type(train_X), type(train_X[0]), type(train_X[0][0]))
    train_X = [list(x) for x in train_X]
    test_X = [list(x) for x in test_X]
    if algo == 'rf_grid':
        from sklearn.model_selection import GridSearchCV
        from sklearn.ensemble import RandomForestClassifier
        param_grid = {
            'n_estimators': [50, 100, 150],
            'max_depth': [2, 4, 6, 10],
            'max_features': ['auto', 'sqrt', 'log2']
        }
        rf = RandomForestClassifier()
        # 定义网格搜索器
        grid_search = GridSearchCV(rf, param_grid=param_grid, cv=5)
        # 使用网格搜索器训练模型
        grid_search.fit(train_X+test_X, train_y+test_y)
        # 输出最优参数和最优得分
        print("Best parameters found: ", grid_search.best_params_)
        print("Best score: ", grid_search.best_score_)
    if algo == 'rf_opt':
        from sklearn.ensemble import RandomForestClassifier
        # grid_search: 'max_depth': 10, 'max_features': 'sqrt', 'n_estimators': 150}
        clf = RandomForestClassifier(n_estimators=150,
                                     # max_depth=10,
                                     max_features='auto',
                                     random_state=42)
        clf.fit(train_X, train_y)
        print(algo, "fit [done]")
    if algo == 'rf':
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(n_estimators=100,
                                     random_state=12)
        #print(type(train_X), type(train_y))
        clf.fit(train_X, train_y)
        print(algo, "fit [done]")

    if algo == 'svm':
        from sklearn.linear_model import SGDClassifier
        clf = SGDClassifier(loss='hinge')
        clf.fit(train_X, train_y)
        print(algo, "fit [done]")

    if algo == 'lr':
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression(random_state=49)
        clf.fit(train_X, train_y)
        print(algo, "fit [done]")

    if algo == 'mlp':
        from sklearn.neural_network import MLPClassifier
        clf = MLPClassifier(solver='sgd', hidden_layer_sizes=(6, 3), random_state=1)
        clf.fit(train_X, train_y)
        print(algo, "fit [done]")

    if algo == 'gbdt':
        from sklearn.ensemble import GradientBoostingClassifier
        clf = GradientBoostingClassifier(n_estimators=100, random_state=39)
        clf.fit(train_X, train_y)
        print(algo, "fit [done]")

    probs = clf.predict_proba(test_X)[:, 1]
    y_pred = clf.predict(test_X)
    print_metrics(test_y, y_pred)