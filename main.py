"""
The project template was adopted from the github repo: https://github.com/moemen95/Pytorch-Project-Template

__author__ = "Mohamed S. Elmahdy"
__paper__ = "https://arxiv.org/abs/2105.01844"

Main
-Capture the config file
-Process the json config passed
-Create an agent instance
-Run the agent
"""

import argparse
from shutil import copyfile

from agents import *
from utils.config import *
from utils.util import parser


def main():
    arg_parser = argparse.ArgumentParser(description="")
    arg_parser.add_argument('args',
                            metavar='args_json_file',
                            default='None',
                            help='The arguments file in json format')

    args_obj = arg_parser.parse_args()
    # parse the config json file
    args, _ = get_config_from_json(args_obj.args)
    setup_logging(args.log_dir, args)

    if args.is_debug:
        data_config = parser(args.debug_config_path)
    else:
        data_config = parser(args.config_path)

    # Create the Agent and pass all the configuration to it then run it..
    agent_class = globals()[args.agent]
    agent_path = os.path.abspath(sys.modules[agent_class.__module__].__file__)
    if not os.path.exists(os.path.join(args.log_dir, agent_path.split('/')[-1])):
        copyfile(agent_path, os.path.join(args.log_dir, agent_path.split('/')[-1]))
    agent = agent_class(args, data_config)
    agent.run()
    agent.finalize()


if __name__ == '__main__':
    main()