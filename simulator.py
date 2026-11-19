"""
simulator.py — QIS-NET Project 2 runner (STUDENT FILE: complete the TODOs).

Drive the provided harness (qnetsim.py) to implement the three graded
entry points. Signatures are normative; the autograder imports them.

Only numpy + standard library + qnetsim may be used.
"""

import math

import numpy as np

from qnetsim import DeliveredPair, Link, Simulator, summarise

# ---------------------------------------------------------------- helpers


def make_links(params, which="variant"):
    """Build Link objects from the variant parameter dict.

    `which="one_hop"` / `"two_hop"` select the reference scenarios defined in
    params["reference"]; `which="variant"` builds the full variant topology.
    PROVIDED — you should not need to change this.
    """
    links = []
    if which == "one_hop":
        ref = params["reference"]["one_hop"]
        links.append(Link(link_id=0, nodes=(0, 1),
                          attempt_rate=ref["attempt_rate"],
                          p_success=ref["p_success"],
                          f_link=params["physics"]["f_link"]))
    elif which == "two_hop":
        ref = params["reference"]["two_hop"]
        for i, lp in enumerate(ref["links"]):
            links.append(Link(link_id=i, nodes=(i, i + 1),
                              attempt_rate=lp["attempt_rate"],
                              p_success=lp["p_success"],
                              f_link=params["physics"]["f_link"]))
    else:
        for i, lp in enumerate(params["topology"]["links"]):
            links.append(Link(link_id=i, nodes=tuple(lp["nodes"]),
                              attempt_rate=lp["attempt_rate"],
                              p_success=lp["p_success"],
                              f_link=params["physics"]["f_link"]))
    return links


def request_times(params, sim_time_ms, seed):
    """Deterministic request arrival times (ms) for the variant workload.
    Uses a generator derived from `seed` so runs are reproducible.
    PROVIDED."""
    wl = params["workload"]
    rng = np.random.default_rng(seed + 7919)
    t, times = 0.0, []
    while True:
        t += rng.exponential(1000.0 / wl["arrival_rate_per_s"])
        if t > sim_time_ms:
            break
        times.append(t)
    return times

class RecordEveryPair:
    """Tiny policy used only by run_one_hop.
    """

    def on_pair_ready(self, link, pair, state, sim):
        sim.delivered.append(DeliveredPair(request_id=-1,
                                           delivered_at=state.now,
                                           fidelity=pair.f0,
                                           consumed=1))

class SwapWhenBothReady:
    """Tiny policy used only by run_two_hop- swap as soon as both links
    hold a pair
    """

    def on_pair_ready(self, link, pair, state, sim):
        if state.pairs.get(0) and state.pairs.get(1):
            sim.swap(0, 1)

def p95_of(values):
    """95th-percentile of a list, using same rule as qnetsim.summarise.
    """
    if not values:
        return math.inf
    
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(math.ceil(0.95 * len(ordered))) - 1)
    return ordered[index]


def match_requests(request_list, delivered):
    """Serve users from a shared pool of ready pairs, first come first served.
    """
    ready_times = sorted(d.delivered_at for d in delivered)
    waits = []
    for i, asked_at in enumerate(request_list):
        if i >= len(ready_times):
            break
        waits.append(max(0.0, ready_times[i] - asked_at))
    return waits


# ---------------------------------------------------------------- graded entry points


def run_one_hop(params, sim_time_ms, seed) -> dict:
    """Single link, no purification.

    Expected answer for variant: 2.7299 * 0.0362 * 1000 = 98.8 /s.
    The run (seed 1017, 30000 ms): rate 100.33 (1.5% off, but allowed 8%),
    fidelity 0.9322 (no decay).
    """
    links = make_links(params, "one_hop")

    sim = Simulator(links, t2=math.inf, q_swap=1.0, f_swap=1.0,
                    purification_allowed=False, seed=seed)

    delivered = sim.run(sim_time_ms, RecordEveryPair())

    summary = summarise(delivered, sim_time_ms)

    return {"rate": summary["rate"],
            "mean_fidelity": summary["mean_fidelity"]}


def run_two_hop(params, sim_time_ms, seed) -> dict:
    """Two links, one repeater, immediate swap.

    Expected answer: rate ~25.1 (slower link 1 is the bottleneck),
    fidelity = 0.9322 * 0.9322 * 0.9747 = 0.847.
    The run (seed 1017): rate 25.67 (2.3% off, but allowed 8%), 
    fidelity 0.8470.
    """
    links = make_links(params, "two_hop")
    sim = Simulator(links, t2=math.inf, q_swap=1.0,
                    f_swap=params["physics"]["f_swap"],
                    purification_allowed=False, seed=seed)
    delivered = sim.run(sim_time_ms, SwapWhenBothReady())
    summary = summarise(delivered, sim_time_ms)
    return {"rate": summary["rate"],
            "mean_fidelity": summary["mean_fidelity"]}


def run_variant_scenario(params, sim_time_ms, seed, policy) -> dict:
    """Full variant scenario with created policy.
    """
    physics = params["physics"]
    links = make_links(params, "variant")
    sim = Simulator(links, t2=physics["t2_ms"], q_swap=physics["q_swap"],
                    f_swap=physics["f_swap"],
                    purification_allowed=physics["purification_allowed"],
                    seed=seed)
    delivered = sim.run(sim_time_ms, policy)
    requests = request_times(params, sim_time_ms, seed)
    waits = match_requests(requests, delivered)
    n = len(delivered)
    rate = n / (sim_time_ms / 1000.0)
    mean_fidelity = sum(d.fidelity for d in delivered) / n if n else 0.0
    consumed = sim.state.stats["generated"] / max(1, n)
    return {"rate": rate,
            "mean_fidelity": mean_fidelity,
            "p95_latency_ms": p95_of(waits),
            "pairs_consumed_per_delivery": consumed}


if __name__ == "__main__":
    import json
    from policy import FreshestPolicy, make_policy
    params = json.load(open("variant_params.json"))
    seed = params["grading"]["seed"]
    sim_time = params["grading"]["sim_time_ms"]
    print("one_hop ", run_one_hop(params, sim_time, seed))
    print("two_hop ", run_two_hop(params, sim_time, seed))
    print("reference", run_variant_scenario(params, sim_time, seed,
                                            FreshestPolicy(params)))
    for s in [seed] + params["grading"]["held_out_seeds"]:
        print("ours", s, run_variant_scenario(params, sim_time, s,
                                              make_policy(params)))