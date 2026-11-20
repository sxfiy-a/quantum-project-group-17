"""
policy.py - QIS-NET Project 2 policy (STUDENT FILE: replace the baseline).

The policy decides, each time a fresh pair arrives on a link, what to do:
swap immediately, purify first, hold, or discard aged pairs. Your variant's
SLOs (rate, fidelity, p95 latency) will typically NOT be met by the
baseline policy below.

Design space (see spec section 3.2):
  * purify while the freshest link fidelity is below a threshold AND a
    second pair is available on the same link;
  * hold vs swap rules (e.g., wait for a purified partner);
  * discard pairs whose current fidelity has fallen below a floor.
"""

import math


class BaselinePolicy:
    """Swap-as-soon-as-possible; never purify. PROVIDED as the baseline.
    It meets the reference scenarios' expectations but is deliberately
    insufficient for most variant SLOs."""

    def __init__(self, params):
        self.params = params
        # map each repeater node to its adjacent link pairs for swapping
        self.chains = params["policy"]["swap_chains"]  # list of [left, right]

    def on_pair_ready(self, link, pair, state, sim):
        for left, right in self.chains:
            if state.pairs.get(left) and state.pairs.get(right):
                sim.swap(left, right)


class FreshestPolicy:
    """swap the NEWEST pair on each side, never purify."""

    def __init__(self, params):
        self.params = params
        self.chains = params["policy"]["swap_chains"]

    def on_pair_ready(self, link, pair, state, sim):
        # for our network: left=0 (fast link), right=1 (slow link)
        for left, right in self.chains:
            # waiting pairs on each side (empty list if none)
            left_pairs = state.pairs.get(left, [])
            right_pairs = state.pairs.get(right, [])
            # need one pair on each side to swap
            if left_pairs and right_pairs:
                # last index = newest pair (lists are oldest-first)
                sim.swap(left, right,
                         left_index=len(left_pairs) - 1,
                         right_index=len(right_pairs) - 1)


class PurifyFreshPolicy:
    """Final policy: keep few fresh pairs on fast link, purify, then swap.

    - fast link makes way more pairs than slow link can use
    - spend the extras on purification there
    - never touch the slow link (bottleneck)
    - keep=2 best fidelity, keep=3 slightly higher rate
    """

    def __init__(self, params, keep=2):
        self.params = params
        self.chains = params["policy"]["swap_chains"]
        self.keep = keep

        # pairs/ms per link
        links = params["topology"]["links"]
        self.fast_link = 0
        best_speed = 0.0
        for i in range(len(links)):
            speed = links[i]["attempt_rate"] * links[i]["p_success"]
            if speed > best_speed:
                best_speed = speed
                self.fast_link = i

    def trim_old_pairs(self, link_id, state, sim):
        """Discard oldest pairs until only `keep` are left."""
        inventory = state.pairs.get(link_id, [])
        while len(inventory) > self.keep:
            sim.discard(link_id, 0)

    def on_pair_ready(self, link, pair, state, sim):
        """Per new pair:

        - trim fast link
        - wait if one side is empty
        - purify fast link if allowed
        - swap newest pairs
        """
        self.trim_old_pairs(self.fast_link, state, sim)
        for left, right in self.chains:
            # one side empty. wait for more pairs
            if not state.pairs.get(left) or not state.pairs.get(right):
                continue

            # check and purify
            if state.purification_allowed and len(state.pairs[self.fast_link]) >= 2:
                sim.purify(self.fast_link)

            # failed purify can empty the link.. check again
            left_pairs = state.pairs.get(left, [])
            right_pairs = state.pairs.get(right, [])
            if left_pairs and right_pairs:
                sim.swap(left, right,
                         left_index=len(left_pairs) - 1,
                         right_index=len(right_pairs) - 1)


def make_policy(params):
    """Return your policy object. Replace BaselinePolicy with your own class
    (same constructor signature and on_pair_ready hook)."""
    return PurifyFreshPolicy(params)
