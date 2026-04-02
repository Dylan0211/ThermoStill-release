def print_args(args):
    """
    Nicely formatted printout of argparse arguments.
    """
    print("\n" + "=" * 60)
    print("Experiment Configuration")
    print("=" * 60)

    # 将 args 转成 dict
    args_dict = vars(args)

    # 按字母排序打印
    max_key_len = max(len(k) for k in args_dict.keys())
    for k in sorted(args_dict.keys()):
        v = args_dict[k]
        print(f"{k:<{max_key_len}} : {v}")

    print("=" * 60 + "\n")
