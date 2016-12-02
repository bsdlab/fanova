import csv
import logging
import os
import shutil
import numpy as np
import pandas as pd
from shutil import copyfile
from pyfanova.fanova import Fanova


class FanovaFromCSV(Fanova):

    def __init__(self, csv_file, **kwargs):

        self._scenario_dir = "tmp_smac_files"

        if not os.path.isdir(self._scenario_dir):
            os.mkdir(self._scenario_dir)

        X, y = self._read_csv_file(csv_file)

        self._write_instances_file()
        self._write_runs_and_results_file(y)
        self._write_param_file()
        self._write_paramstrings_file(X)
        self._write_scenario_file()

        logging.debug("Write temporary smac files in " + self._scenario_dir)
        super(FanovaFromCSV, self).__init__(self._scenario_dir)

    def __del__(self):
        shutil.rmtree(self._scenario_dir)
        super(FanovaFromCSV, self).__del__()

    def _write_scenario_file(self):

        fh = open(os.path.join(self._scenario_dir, "scenario.txt"), "w")

        fh.write("algo = .\n")
        fh.write("execdir = .\n")
        fh.write("deterministic = 0\n")
        fh.write("run_obj = qual\n")
        fh.write("overall_obj = mean\n")
        fh.write("cutoff_time = 1e100\n")
        fh.write("cutoff_length = 0\n")
        fh.write("tunerTimeout = 0\n")
        fh.write("paramfile = .\n")
        fh.write("instance_file = .\n")
        fh.write("test_instance_file = .\n")

        fh.close()

    def _write_instances_file(self):

        fh = open(os.path.join(self._scenario_dir, "instances.txt"), "w")
        fh.write(".")
        fh.close()

    def _write_runs_and_results_file(self, values):

        fh = open(os.path.join(self._scenario_dir, "runs_and_results.csv"), "wb")
        writer = csv.writer(fh)
        writer.writerow(("Run Number", "Run History Configuration ID", "Instance ID", "Response Value (y)", "Censored?", "Cutoff Time Used",
                                      "Seed", "Runtime", "Run Length", "Run Result Code", "Run Quality", "SMAC Iteration", "SMAC Cumulative Runtime", "Run Result"))

        for i in range(0, len(values)):
            line = (i, i, 1, 0, 0, 0, 1, 0, 0, 0, values[i], 0, 0, "SAT")
            writer.writerow(line)

        fh.close()

    def _write_param_file(self):
        fh = open(os.path.join(self._scenario_dir, "param-file.txt"), "w")

        for key, value in self._paramDescription.iteritems():
            bounds = map(str,value[0])
            if value[-1] == np.int64:
                sufix = 'i'
            else:
                sufix = ''
            if value[-1] == np.int64 or value[-1] == np.float64:
                bounds = '['+(','.join(bounds))+']'
            else:
                bounds = '{'+(','.join(bounds))+'}'
            init_val = '['+str(value[1])+']'
            param_string = ' '.join([key,bounds,init_val])+sufix + "\n"
            fh.write(param_string)

        fh.close()

    def _write_paramstrings_file(self, params):

        fh = open(os.path.join(self._scenario_dir, "paramstrings.txt"), "w")
        for row in params.itertuples():
            line = str(row[0]) + ': '
            names = row._fields[1:]
            data = row[1:]
            for idx_item, item in enumerate(data):
                line = line + names[idx_item] + "='{}'".format(item) + ', '
            line = line[:-2]
            print(line)
            line = line + '\n'

            fh.write(line)
        fh.close()

    def _read_csv_file(self, filename):
        X = pd.read_csv(filename)
        y = X[X.columns[-1]]
        X = X[X.columns[0:-1]]

        self._num_of_params = len(X.columns)

        logging.debug("number of parameters: " + str(self._num_of_params))
        self._paramDescription = dict()
        for idx, dtype in enumerate(X.dtypes):
            cX = X[X.columns[idx]]
            if dtype == np.float64 or dtype == np.int64:
                cBounds = [cX.min(), cX.max()]
            else:
                cBounds = cX.unique()
            self._paramDescription[X.columns[idx]] = [cBounds, cBounds[0], dtype]

        return X, y
