
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 08:34:37 2016
@author: fr_tk210
"""

from os import listdir

import re
import argparse
import sys
import os

from os import path as path
class Resultparser:

    path_output =  ''

    def __init__(self,dir_state_run):
        """


        """
        self.smac_output = os.path.dirname(dir_state_run)
        self.dir_state_run = dir_state_run

    def parse_runConfigs(self):

        file_name = ''

        file_found = False
        dir_paramstrings = self.dir_state_run
        for line in listdir(dir_paramstrings): #get the filename of the detailed_traj_run file
            if line.startswith("paramstrings"):
                file_name = line
                file_found = True

        if file_found == False:
            raise ValueError('Cannot find the detailed-traj-run file in directory: {0}'.format(dir_paramstrings))
            return

        path_run_configs = self.dir_state_run + '/'+ file_name
        params_dict = []
        with open(path_run_configs) as f:
            for line in f:
                # In this version of smac the paramstrings line
                # only has one occurence of: ,
                # after which the config parameter value pairs begin
                config_line = line.split(', ')
                config_line[0] =  re.sub("\A\d+:","",config_line[0]) # remove the "1: " substring  #config param - values begin after ", "
                #print(config_line)
                parameter_values = config_line
                params_line_dict = dict()
                for param_value in parameter_values:
                    m = re.match(r"(?P<param>.+)='(?P<value>.+)'",param_value)

                    if m is None:
                        raise ValueError("Couldn't parse config string!")
                    param = m.group('param')
                    param = param.strip() #remove whitespaces at beginning and end
                    params_line_dict[param] = m.group('value')

                params_dict.append(params_line_dict)
        return params_dict


def main():
    cmd_parser = argparse.ArgumentParser()
    cmd_parser.add_argument('output_folder',type=str)

    cmd_args = vars(cmd_parser.parse_args())

    path_output =  cmd_args["output_folder"]

    config_parser = Resultparser(path_output)
    config_parser.parse_runConfigs()


if __name__ == "__main__":
    sys.exit(main())
