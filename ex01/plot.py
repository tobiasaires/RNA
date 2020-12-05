def plot_decision_region(realization, configurations, iris_cfg, class_names):
    plot_colors = "rb"
    plot_step = 0.01

    test_base = realization['test_base']
    training_base = realization['training_base']

    x_min, x_max = test_base[:, 1].min() - 0.2, test_base[:, 1].max() + 0.2
    y_min, y_max = test_base[:, 2].min() - 0.2, test_base[:, 2].max() + 0.2

    xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                         np.arange(y_min, y_max, plot_step))

    data = np.c_[xx.ravel(), yy.ravel()]
    bias_col = -np.ones(data.shape[0])

    data = np.insert(data, 0, bias_col, axis=1)
    perceptron_g = ps.NeuronioMP(3)
    perceptron_g.weights = realization['weights']

    Z = np.array([perceptron_g.predict(x) for x in data])
    Z = Z.reshape(xx.shape)

    cs = plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu)

    y_test = test_base[:, -1]
    y_training = training_base[:, -1]

    if configurations['chosen_base'] == 'Iris':
        attributes = iris_cfg['features']
        plt.xlabel(attributes[0])
        plt.ylabel(attributes[1])
    else:
        plt.xlabel('Attribute x')
        plt.ylabel('Attribute y')

    plt.legend(class_names)

    for i, color in zip(range(len(class_names)), plot_colors):
        idx = np.where(y_test == i)
        plt.scatter(test_base[idx, 1], test_base[idx, 2], marker='^', c=color, label=class_names[i],
                    edgecolor='black', s=20)
        idx = np.where(y_training == i)
        plt.scatter(training_base[idx, 1], training_base[idx, 2], marker='s', c=color,
                    edgecolor='black', s=20)

    plt.show()