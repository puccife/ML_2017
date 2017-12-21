from __future__ import division
from __future__ import print_function

import pandas as pd

import tensorflow as tf

from config.argument_loader import ArgumentLoader
from dmn.dmn_plus import Config, DMN_PLUS
from model.dmn_trainer import DMNTrainer

al = ArgumentLoader()
FLAGS = al.get_configuration()

def main(args):
    dmn_trainer, config, model, init, saver = load_prerequisites()
    results = predict(saver, init, model, config, dmn_trainer)
    create_prediction("DMN_prediction", results)

def load_prerequisites():
    dmn_trainer = DMNTrainer(FLAGS)
    config = Config()
    model = DMN_PLUS(config, dmn_trainer)
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    return dmn_trainer, config, model, init, saver

def predict(saver, init, model, config, dmn_trainer):
    with tf.Session() as session:
        session.run(init)

        print('==> restoring weights')
        saver.restore(session, './weights/DMN_weights/taskDMN1epocha1.weights')

        print('==> running DMN')
        predictions = model.run_epoch(session, model.test)

    final_prediction = []
    for prediction in predictions:
        final_prediction.extend(dmn_trainer.ivocab[prediction])
    return final_prediction

def create_prediction(name, final_prediction):
        # print(final_prediction)
        df = pd.DataFrame(final_prediction.copy())
        df.index += 1
        df = df.reset_index()
        df.columns = ['Id', 'Prediction']
        df = df.set_index('Id')
        df[df['Prediction'] == 0] = -1
        df.to_csv('./predictions_csv/DMN_prediction.csv')
        df = pd.read_csv('./predictions_csv/DMN_prediction.csv')
        df = df.set_index('Id')
        df[df['Prediction'] == 0] = -1
        df.to_csv("./predictions_csv/DMN_prediction.csv")

if __name__ == "__main__":
  tf.app.run()
