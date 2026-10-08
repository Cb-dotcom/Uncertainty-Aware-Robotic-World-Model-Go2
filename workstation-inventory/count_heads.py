import sys, re, torch
for p in sys.argv[1:]:
    ck = torch.load(p, map_location="cpu", weights_only=False)
    def walk(o, pre=""):
        if isinstance(o, dict):
            for k, v in o.items():
                yield from walk(v, f"{pre}{k}.")
        elif torch.is_tensor(o):
            yield pre[:-1], o
    tens = [(k, t) for k, t in walk(ck) if "optim" not in k]
    def count(pat):
        return len({m.group(1) for k, _ in tens for m in [re.search(pat, k)] if m})
    def nparams(pat):
        return sum(t.numel() for k, t in tens if re.search(pat, k))
    print("\n" + p.split("/workspace/")[-1])
    print("  top-level keys:", list(ck.keys()) if isinstance(ck, dict) else type(ck))
    print("  state_heads:", count(r"state_heads\.(\d+)\."), " auxiliary_heads:", count(r"auxiliary_heads\.(\d+)\."))
    print("  params: state_base={:,} one_state_head={:,} aux_base={:,} all_non_optim={:,}".format(
        nparams(r"state_base"), nparams(r"state_heads\.0\."), nparams(r"auxiliary_base"), sum(t.numel() for _, t in tens)))
