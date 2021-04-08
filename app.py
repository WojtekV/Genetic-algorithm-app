from flask import Flask, render_template, request
import numpy as np
import json
import plotly
import plotly.graph_objects as go
from genetic_algorithm import GeneticAlgorithm as ga
from fitness_function import fitness_function_1, fitness_function_2
import matplotlib.pyplot as plt
import timeit

app = Flask(__name__)


@app.route('/', methods=["GET"])
def gen_app():
    return render_template('genForm.html')


@app.route('/plot', methods=["POST"])
def plot():
    start = timeit.default_timer()
    gen_alg = ga(
        population_size=int(request.form['populationSize']),
        search_min=float(request.form['searchMin']),
        search_max=float(request.form['searchMax']),
        bin_length=int(request.form['chrLength']),
        crossover_prob=float(request.form['crossProb']),
        crossover_type=request.form['crossType'],
        problem=request.form['target'],
        mutation_type=request.form['mutType'],
        mutation_prob=float(request.form['mutProb']),
        num_of_epochs=int(request.form['epochsNumber']),
        selection_type=request.form['selType'],
        percent_of_best=float(request.form['bestNum']),
        number_of_elite=int(request.form['eliteNum']),
        tournament_size=int(request.form['tSize']),
        inversion_prob=float(request.form['invProb']),
        function_nr=request.form['function']
    )
    stop = timeit.default_timer()
    best_point = gen_alg.get_point(gen_alg.get_best_sub())

    pl = create_plot(fun_num=request.form['function'],
                     r_min=int(request.form['searchMin']),
                     r_max=int(request.form['searchMax']),
                     x=best_point[0],
                     y=best_point[1]
                     )

    plt.plot(np.arange(len(gen_alg.best_score_for_epoch)), gen_alg.best_score_for_epoch)
    plt.title('best_score')
    plt.savefig("./plots/best_score.png")
    plt.show()
    plt.plot(np.arange(len(gen_alg.std_devs)), gen_alg.std_devs)
    plt.title('std dev')
    plt.savefig("./plots/std_dev.png")
    plt.show()
    plt.plot(np.arange(len(gen_alg.means)), gen_alg.means)
    plt.title('mean')
    plt.savefig("./plots/mean.png")
    plt.show()

    calc_time = stop - start
    best_point = gen_alg.get_point(gen_alg.get_best_sub())
    return render_template('plot.html', plot=pl, lay=create_layout(f"Calc time = {calc_time:.2f}s, Best point: (x: {best_point[0]:.5f}, y: {best_point[1]:.5f})"))

def create_layout(title):
    return json.dumps(go.Layout(title=title), cls=plotly.utils.PlotlyJSONEncoder)


def create_plot(fun_num, r_min, r_max, x, y):
    r_min -= 1
    r_max += 1
    x_p = np.arange(r_min, r_max, (r_max-r_min)/100)
    y_p = np.arange(r_min, r_max, (r_max-r_min)/100)
    X, Y = np.meshgrid(x_p, y_p)
    if fun_num == '1':
        ff = fitness_function_1
    else:
        ff = fitness_function_2
    zs = np.array([ff(x_p, y_p) for x_p, y_p in zip(np.ravel(X), np.ravel(Y))])
    Z = zs.reshape(X.shape)

    data = [
        go.Surface( z=Z, y=Y, x=X, colorscale='RdBu', opacity=0.7),
        go.Scatter3d(x=[x], y=[y], z=[ff(x, y)], mode='markers',
                     marker=dict(size=6, color='black', colorscale='Viridis', opacity=1))
    ]
    graphJSON = json.dumps(data, cls=plotly.utils.PlotlyJSONEncoder)
    return graphJSON


if __name__ == '__main__':
    app.run()
