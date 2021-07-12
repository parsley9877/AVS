from sacred import Experiment

dummy_experiment = Experiment(name='dummy_exp')

@dummy_experiment.automain
def main():
    print('helloo!!')
    return 'hello'