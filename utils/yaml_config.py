import yaml
from termcolor import colored as clr
from argparse import Namespace
import subprocess


class YamlNamespace(Namespace):
    """ PyLint will trigger `no-member` errors for Namespaces constructed
    from yaml files. I am using this inherited class to target an
    `ignored-class` rule in `.pylintrc`.
    """


def create_paths(args: Namespace) -> Namespace:
    """ Creates directories for containing experiment results.
    """
    time_stamp = "{:%Y%b%d-%H%M%S}".format(datetime.now())
    if not hasattr(args, "out_dir") or args.out_dir is None:
        if not os.path.isdir("./results"):
            os.mkdir("./results")
        out_dir = f"./results/{time_stamp}_{args.experiment:s}"
        os.mkdir(out_dir)
        args.out_dir = out_dir
    elif not os.path.isdir(args.out_dir):
        raise Exception(f"Directory {args.out_dir} does not exist.")

    if not hasattr(args, "run_id"):
        args.run_id = 0

    return args


def dict_to_namespace(dct: dict) -> Namespace:
    """Deep (recursive) transform from Namespace to dict"""
    namespace = YamlNamespace()
    for key, value in dct.items():
        name = key.rstrip("_")
        if isinstance(value, dict) and not key.endswith("_"):
            setattr(namespace, name, dict_to_namespace(value))
        else:
            setattr(namespace, name, value)
    return namespace


def namespace_to_dict(namespace: Namespace) -> dict:
    """Deep (recursive) transform from Namespace to dict"""
    dct: dict = {}
    for key, value in namespace.__dict__.items():
        if isinstance(value, Namespace):
            dct[key] = namespace_to_dict(value)
        else:
            dct[key] = value
    return dct


def config_to_string(
    cfg: Namespace, indent: int = 0, color: bool = True
) -> str:
    """Creates a multi-line string with the contents of @cfg."""

    text = ""
    for key, value in cfg.__dict__.items():
        ckey = clr(key, "green", attrs=["bold"]) if color else key
        text += " " * indent + ckey + ": "
        if isinstance(value, Namespace):
            text += "\n" + config_to_string(value, indent + 2, color=color)
        else:
            text += str(value) + "\n"
    return text


def read_config(cfg_path):
    """ Read a config file and return a namespace.
    """
    with open(cfg_path) as handler:
        config_data = yaml.load(handler, Loader=yaml.SafeLoader)
    return dict_to_namespace(config_data)


def get_git_info():
    """ Return sha@branch.
    This can maybe be used when restarting experiments. We can trgger a
    warning if the current code-base does not match the one we are trying
    to resume from.
    """
    cmds = [
        ["git", "rev-parse", "--short", "HEAD"],  # short commit sha
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],  # branch name
    ]
    res = []
    for cmd in cmds:
        res.append(subprocess.check_output(cmd).strip().decode("utf-8"))
    return "@".join(res)
