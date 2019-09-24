from tensorflow import flags
import utils.yaml_config as yaml_config
import yaml

FLAGS_0 = flags.FLAGS

# read additional params from console
flags.DEFINE_string(
    "model_dir", "/models/graph_models/models_something_something_new/tmp_model", "")
flags.DEFINE_string("config_file", "./configs/smt_config.yaml", "")
flags.DEFINE_integer("rand_no", 1241322, "")

# read params from config file
def read_params(save=True):
    FLAGS = yaml_config.read_config(FLAGS_0.config_file)

    # add console params to config params
    FLAGS.model_dir = FLAGS_0.model_dir
    FLAGS.rand_no = FLAGS_0.rand_no

    # save git info
    FLAGS.git_info = yaml_config.get_git_info()
    FLAGS.num_eval_clips = FLAGS.num_eval_spatial_clips * FLAGS.num_eval_temporal_clips
    name_yaml = FLAGS.model_dir + f'/config_{FLAGS.rand_no}.yaml'
    print(name_yaml)
    if save:
        # save current config file
        with open(name_yaml, 'w') as outfile:
            yaml.dump(yaml_config.namespace_to_dict(FLAGS),
                      outfile, default_flow_style=False)

    print(yaml_config.config_to_string(FLAGS))
    return FLAGS
