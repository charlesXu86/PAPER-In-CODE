"""used in beam search"""
class Derivation(object):
    def __init__(self, stack, action_top, output_actions, output_tokens, logp, nt_allowed, ter_allowed, reducable, total_nt, next_act=None, next_tok=None):
        self.stack = list(stack)
        self.action_top = action_top
        self.output_actions = list(output_actions)
        self.output_tokens = list(output_tokens)
        self.logp = logp
        self.nt_allowed = nt_allowed
        self.ter_allowed = ter_allowed
        self.reducable = reducable
        self.total_nt = total_nt
        self.next_act = next_act
        self.next_tok = next_tok


class Derivation_b(object):
    def __init__(self, stack, action_top, output_actions, output_tokens, logp, consecutive_nt, consecutive_ter, total_ter, found_root, next_act=None, next_tok=None):
        self.stack = list(stack)
        self.action_top = action_top
        self.output_actions = list(output_actions)
        self.output_tokens = list(output_tokens)
        self.logp = logp
        self.consecutive_nt = consecutive_nt
        self.consecutive_ter = consecutive_ter
        self.total_ter = total_ter
        self.found_root = found_root
        self.next_act = next_act
        self.next_tok = next_tok
