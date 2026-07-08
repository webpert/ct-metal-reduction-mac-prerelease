import os
import sys
import uuid
import os.path as osp
from argparse import Namespace
import yaml
from datetime import datetime

try:
    from tensorboardX import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

sys.path.append("./")
from mac_gaussian.utils.cfg_utils import args2string


def prepare_output_and_logger(args):
    # Update model path if not specified
    if not args.model_path:
        if os.getenv("OAR_JOB_ID"):
            unique_str = os.getenv("OAR_JOB_ID")
        else:
            # unique_str = str(uuid.uuid4())
            unique_str = datetime.now().strftime("%m-%d-%H-%M-%S")
        filename = os.path.basename(args.config)        # "default.yaml"
        config_name, _ = os.path.splitext(filename)   
        # args.model_path = f"./output/{config_name}_{unique_str}"
        source_name = os.path.basename(args.source_path.rstrip('/'))
        args.model_path = f"./output/{source_name}_{unique_str}"

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(osp.join(args.model_path, "cfg_args"), "w", encoding='utf-8') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Save to yaml
    args_dict = vars(args)
    with open(osp.join(args.model_path, "cfg_args.yml"), "w") as f:
        yaml.dump(args_dict, f, default_flow_style=False, sort_keys=False)

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
        tb_writer.add_text("args", args2string(args_dict), global_step=0)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer
