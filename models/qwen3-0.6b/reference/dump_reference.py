#!/usr/bin/env python3
"""PyTorch reference for Qwen3-0.6B: per-block dumps, greedy ids, manifests.

usage:
  dump_reference.py dump --model <snapshot dir> --dumps <dir> --repo <reference dir>
  dump_reference.py decode --model <snapshot dir> --ids <ids.json> --out <text file>
  dump_reference.py chain --model <snapshot dir> --dumps <dir> --repo <reference dir>

dump: loads the checkpoint in fp32 on the CPU and, for each fixed prompt (plain
text, no chat template, the tokenizer's default special tokens),
  - runs one prefill forward and writes to <dumps>/<prompt id>/ as raw fp32:
    emb.f32 [S, 1024] (embedding output), block_00.f32 .. block_27.f32 [S, 1024]
    (each decoder layer's output), final_norm.f32 [S, 1024], logits_last.f32
    [151936] (the last position's logits), argmax.json (argmax per position),
    ids.txt ("<id> <token ids>", the graph binary's input);
  - greedy-decodes 16 tokens with a full re-forward per token and checks that
    generate(do_sample=False) gives the same ids.
  Writes to <repo>: prompts.json, ref_ids.json ({"prompts": [{"id", "ids",
  "text"}]}, ids = prompt + 16 generated) and manifest_<id>.json per prompt (the
  output-match bar, fixed here before any graph run: cosine >= 0.9999 and
  max_abs <= max(2e-2, 1e-3 * max|ref|), logits max(5e-2, 1e-3 * max|ref|)).
  Every tensor is taken from a forward hook. In this transformers version
  output_hidden_states[-1] is the final norm output, not layer 27's output;
  the script asserts the hooks agree with output_hidden_states.

decode: writes one line per prompt of an ids JSON: "<id>\t<decoded text>".

chain: the stateless reference for chained decode. For each prompt in
  <repo>/ref_ids.json (prompt + 16 greedy ids), runs one fp32 forward over all
  ids with no cache (use_cache=False) and writes the logits rows at positions
  S-1 .. S+14 to <dumps>/<prompt id>/chain_00.f32 .. chain_15.f32 ([151936]
  each): row i is the input of generated token i + 1, so the graph's prefill
  logits (i = 0) and its 15 decode steps compare with it. Asserts that row i's
  argmax is generated token i + 1. Writes <repo>/manifest_chain_<id>.json with
  the bar fixed here: cosine >= 0.9999, max_abs <= max(5e-2, 1e-3 * max|ref|)
  (the logits bar of the dump manifests).
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPTS = [
    ("p0", "The capital of France is"),
    ("p1", "def fibonacci(n):"),
    ("p2", "In 1969, humans first landed on the Moon. The mission was called"),
]
NEW_TOKENS = 16
COSINE = 0.9999


def bar_max_abs(max_ref, floor):
    return max(floor, 1e-3 * max_ref)


def write_f32(path, t):
    arr = t.detach().to(torch.float32).contiguous().numpy()
    arr.astype("<f4").tofile(path)
    return arr


def greedy(model, ids, n):
    seq = list(ids)
    for _ in range(n):
        with torch.no_grad():
            logits = model(torch.tensor([seq])).logits
        seq.append(int(torch.argmax(logits[0, -1])))
    return seq


def dump(args):
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float32, attn_implementation="eager")
    model.eval()
    layers = model.model.layers
    captured = {}

    def hook(name):
        def fn(_module, _inputs, output):
            captured[name] = output[0] if isinstance(output, tuple) else output
        return fn

    model.model.embed_tokens.register_forward_hook(hook("emb"))
    for i, layer in enumerate(layers):
        layer.register_forward_hook(hook(f"block_{i:02d}"))
    model.model.norm.register_forward_hook(hook("final_norm"))
    names = ["emb"] + [f"block_{i:02d}" for i in range(len(layers))] + ["final_norm"]

    os.makedirs(args.repo, exist_ok=True)
    prompts_out, ref_out = [], []
    for pid, text in PROMPTS:
        ids = tok(text)["input_ids"]
        ids_plain = tok(text, add_special_tokens=False)["input_ids"]
        seq = len(ids)
        d = os.path.join(args.dumps, pid)
        os.makedirs(d, exist_ok=True)

        captured.clear()
        with torch.no_grad():
            out = model(torch.tensor([ids]), output_hidden_states=True)
        hs = out.hidden_states
        # hidden_states[0] = embedding output, [i + 1] = layer i output for
        # i < 27, [-1] = final norm output (layer 27's output is not in it).
        assert torch.equal(hs[0][0], captured["emb"][0])
        for i in range(len(layers) - 1):
            assert torch.equal(hs[i + 1][0], captured[f"block_{i:02d}"][0])
        assert torch.equal(hs[-1][0], captured["final_norm"][0])
        last_is_block = torch.equal(hs[-1][0], captured[f"block_{len(layers) - 1:02d}"][0])

        tensors = []
        for name in names:
            arr = write_f32(os.path.join(d, f"{name}.f32"), captured[name][0])
            max_ref = float(np.max(np.abs(arr)))
            tensors.append({"name": name, "file": f"{name}.f32", "shape": [seq, arr.shape[-1]],
                            "max_abs_ref": max_ref, "bar": {"max_abs": bar_max_abs(max_ref, 2e-2)}})
        logits = out.logits[0]
        arr = write_f32(os.path.join(d, "logits_last.f32"), logits[-1])
        max_ref = float(np.max(np.abs(arr)))
        tensors.append({"name": "logits_last", "file": "logits_last.f32", "shape": [arr.shape[0]],
                        "max_abs_ref": max_ref, "bar": {"max_abs": bar_max_abs(max_ref, 5e-2)}})
        argmax = [int(x) for x in torch.argmax(logits, dim=-1)]
        with open(os.path.join(d, "argmax.json"), "w") as f:
            json.dump({"argmax": argmax}, f)
        with open(os.path.join(d, "ids.txt"), "w") as f:
            f.write(pid + " " + " ".join(str(t) for t in ids) + "\n")

        manifest = {
            "bar": {"cosine": COSINE},
            "note": ("CPU fp32 reference, transformers %s, torch %s, eager attention. Shapes are "
                     "[S, hidden] for prompt %s (S = %d). emb = embed_tokens output; block_NN = "
                     "decoder layer NN output (forward hook); final_norm = model.norm output "
                     "(= output_hidden_states[-1] in this version, which therefore is not "
                     "block_27: %s). Bar fixed at dump time: cosine >= %g, max_abs <= max(2e-2, "
                     "1e-3 * max|ref|), logits_last max(5e-2, 1e-3 * max|ref|)."
                     % (transformers.__version__, torch.__version__, pid, seq,
                        "verified, they differ" if not last_is_block else "equal here", COSINE)),
            "tensors": tensors,
        }
        with open(os.path.join(args.repo, f"manifest_{pid}.json"), "w") as f:
            json.dump(manifest, f, indent=1)
            f.write("\n")

        hand = greedy(model, ids, NEW_TOKENS)
        with torch.no_grad():
            gen = model.generate(torch.tensor([ids]), do_sample=False, max_new_tokens=NEW_TOKENS)
        gen = [int(x) for x in gen[0]]
        same = gen == hand
        print(f"{pid}: S={seq} ids={ids} add_special_tokens default adds "
              f"{len(ids) - len(ids_plain)} tokens; argmax last {argmax[-1]}; "
              f"greedy {hand[seq:]}; generate {'same' if same else 'DIFFERS: ' + str(gen[seq:])}")
        prompts_out.append({"id": pid, "text": text, "ids": ids, "S": seq,
                            "add_special_tokens_default_adds": len(ids) - len(ids_plain)})
        ref_out.append({"id": pid, "ids": hand, "text": tok.decode(hand),
                        "generate_same": same})

    with open(os.path.join(args.repo, "prompts.json"), "w") as f:
        json.dump({"prompts": prompts_out}, f, indent=1)
        f.write("\n")
    with open(os.path.join(args.repo, "ref_ids.json"), "w") as f:
        json.dump({"prompts": ref_out}, f, indent=1)
        f.write("\n")
    with open(os.path.join(args.dumps, "ids_all.txt"), "w") as f:
        for p in prompts_out:
            f.write(p["id"] + " " + " ".join(str(t) for t in p["ids"]) + "\n")
    print("DONE reference dump")
    return 0


def decode(args):
    tok = AutoTokenizer.from_pretrained(args.model)
    with open(args.ids) as f:
        prompts = json.load(f)["prompts"]
    with open(args.out, "w") as f:
        for p in prompts:
            f.write(p["id"] + "\t" + json.dumps(tok.decode(p["ids"]), ensure_ascii=False) + "\n")
    print(f"DONE decode {len(prompts)} prompts")
    return 0


def chain(args):
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float32, attn_implementation="eager")
    model.eval()
    with open(os.path.join(args.repo, "ref_ids.json")) as f:
        ref = json.load(f)["prompts"]
    with open(os.path.join(args.repo, "prompts.json")) as f:
        lengths = {p["id"]: p["S"] for p in json.load(f)["prompts"]}
    for p in ref:
        pid, ids, seq = p["id"], p["ids"], lengths[p["id"]]
        steps = len(ids) - seq
        d = os.path.join(args.dumps, pid)
        os.makedirs(d, exist_ok=True)
        with torch.no_grad():
            logits = model(torch.tensor([ids]), use_cache=False).logits[0]
        tensors = []
        for i in range(steps):
            row = logits[seq - 1 + i]
            assert int(torch.argmax(row)) == ids[seq + i], (pid, i)
            name = f"chain_{i:02d}"
            arr = write_f32(os.path.join(d, f"{name}.f32"), row)
            max_ref = float(np.max(np.abs(arr)))
            tensors.append({"name": name, "file": f"{name}.f32", "shape": [arr.shape[0]],
                            "max_abs_ref": max_ref, "bar": {"max_abs": bar_max_abs(max_ref, 5e-2)}})
        manifest = {
            "bar": {"cosine": COSINE},
            "note": ("Stateless reference for chained decode, prompt %s (S = %d): one CPU fp32 "
                     "forward (transformers %s, torch %s, eager attention, use_cache=False) over "
                     "prompt + %d generated ids; row i = logits at position S-1+i, whose argmax is "
                     "generated token i+1 (asserted). chain_00 is the prefill's last-position "
                     "logits, chain_01.. are the decode steps. Bar fixed at dump time: cosine >= "
                     "%g, max_abs <= max(5e-2, 1e-3 * max|ref|)."
                     % (pid, seq, transformers.__version__, torch.__version__, steps, COSINE)),
            "tensors": tensors,
        }
        with open(os.path.join(args.repo, f"manifest_chain_{pid}.json"), "w") as f:
            json.dump(manifest, f, indent=1)
            f.write("\n")
        print(f"{pid}: S={seq}, {steps} chain rows, argmax == generated ids")
    print("DONE reference chain")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("dump")
    a.add_argument("--model", required=True, help="Hugging Face snapshot directory")
    a.add_argument("--dumps", required=True, help="where the f32 dumps go (not committed)")
    a.add_argument("--repo", required=True, help="where prompts.json, ref_ids.json, manifests go")
    b = sub.add_parser("decode")
    b.add_argument("--model", required=True)
    b.add_argument("--ids", required=True)
    b.add_argument("--out", required=True)
    c = sub.add_parser("chain")
    c.add_argument("--model", required=True, help="Hugging Face snapshot directory")
    c.add_argument("--dumps", required=True, help="where the f32 dumps go (not committed)")
    c.add_argument("--repo", required=True,
                   help="reference dir: reads ref_ids.json and prompts.json, writes manifests")
    args = ap.parse_args()
    torch.manual_seed(0)
    return {"dump": dump, "decode": decode, "chain": chain}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
