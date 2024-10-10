import json
import argparse
import os
import itertools


def field_crossprod(f):
    """Return `list` of `dict` with all possible combinations of the elements in the `dict` values lists."""
    keys = list(f.keys())
    l = list(itertools.product(*[f[k] for k in keys]))
    out = [{keys[i]:ll[i] for i in range(len(keys))} for ll in l]
    return out


def add_prefix_to_keys(in_arg, pref=""):
    """Recursively append a prefix to all the keys in a `dict` of `dict`.
    The recursion stops when the values are not `dict`."""
    return {pref+k: add_prefix_to_keys(v, pref=pref+k+":") if isinstance(v, dict) else v for k, v in in_arg.items()}
    # return [add_prefix_to_keys(in_arg[k], pref=pref+k+":") for k in in_arg] if isinstance(in_arg, dict)  else in_arg

    # {pref + k: in_d[k] if ~isinstance(in_d[k], dict) else  for k in in_d.keys()}


def get_leafs(d):
    """Recursively get a `list` of all the values which are not `dict` in a `dict` of `dict`.
    The recursion stops when the values are not `dict`."""
    return [get_leafs(v) if isinstance(v, dict) else {k: v} for k, v in d.items()]


def flatten_list(li):
    """Recursively flatten a `list` of `list`.
    The recursion stops when the input are not `lists`.
    """
    return sum(([x] if not isinstance(x, list) else flatten_list(x)
                for x in li), [])


def flatten_dict(ld):
    """flatten a `dict` of `dict`.
    e.g.
    Input:{"a":{"a1":"a11", "a2":"a12"} ,"b":"b1"}
    Output: {"a:a1":"a11","a:a2":"a12","b":"b1"}
    """
    flat_keys = add_prefix_to_keys(ld)
    flat_list_of_kv = flatten_list(get_leafs(flat_keys))
    flat_dict = {k: v for ll in flat_list_of_kv for k, v in ll.items()}
    return flat_dict


def get_correct_sub_dict_list(ld, p):
    """Get sub-dictionary with keys containing the string \"p:\" """
    return {k.replace(p+":", ""): v for k, v in ld.items() if (p+":" in k)}


def get_sub_dict_list_of_leafs(correct_ld):
    """Get sub-dictionary with keys NOT containing the character \":\""""
    return {k: v for k, v in correct_ld.items() if not (":" in k)}


def get_sub_dict_list_of_non_leafs(correct_ld):
    """Get sub-dictionary with keys containing the character \":\""""
    return {k: v for k, v in correct_ld.items() if ":" in k}


def nest_dict(flat_d):
    """From a dict with keys containing concatenation of sub-keys, returns a nested dictionary."""
    if len(flat_d) == 0:
        return flat_d

    all_keys = [k.split(":") for k in flat_d.keys()]
    level_keys = list(set([x[0] for x in all_keys]))

    return {lk: dict(nest_dict(get_sub_dict_list_of_non_leafs(get_correct_sub_dict_list(flat_d, lk))),
                     **get_sub_dict_list_of_leafs(get_correct_sub_dict_list(flat_d, lk)))
            for lk in level_keys}




def check_fields(in_dict):
    """Check that all the values of the subdisctionnaries are lists"""
    return all([check_fields(in_dict[k]) if isinstance(in_dict[k], dict) else isvalue(in_dict[k])
          for k in in_dict.keys()])


def isvalue(in_value):
    return isinstance(in_value, list)


def read_overview_json(fname):
    with open(fname, "r") as fp:
        d_ov = json.load(fp)
    assert(check_fields(d_ov))
    return d_ov


def write_spec_json(fname):
    with open(fname, "w") as fp:
        json.dump(nest_dict(d), fp, indent=3)


parser = argparse.ArgumentParser()
parser.add_argument("-i", type=str, metavar="in_json",
                    help="Input .json file. Keys/field names should not contain ':'.", required=True)

parser.add_argument("-o", type=str, metavar="out_folder",
                    help="Output folder containing all .json files. default: dirname(input_json)/basename(input_json)", default=None)



if __name__ == "__main__":
    args = parser.parse_args()
    infile = args.i
    if args.o is None:
        outfolder = os.path.join(os.path.dirname(infile), os.path.basename(infile).replace(".json", ""))
    else:
        outfolder = args.o

    os.makedirs(outfolder, exist_ok=True)

    d_ov = read_overview_json(infile)

    flat_test_d = flatten_dict(d_ov)
    all_comb = field_crossprod(flat_test_d)
    
    for i, d in enumerate(all_comb):
        fname = os.path.join(outfolder, str(i + 1)+".json")
        write_spec_json(fname)
