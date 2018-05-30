"""used in beam search"""
class Derivation(object):
    def __init__(self, stack, action_top, output_actions, output_tokens, logp, stack_token, next_act=None, next_tok=None):
        self.stack = list(stack)
        self.stack_token = list(stack_token)
        self.action_top = action_top
        self.output_actions = list(output_actions)
        self.output_tokens = list(output_tokens)
        self.logp = logp
        self.next_act = next_act
        self.next_tok = next_tok

    def __str__(self):
        if len(self.stack_token) >= 1:
            return ' '.join(self.stack_token)
        else:
            return ''

    def __repr__(self):
        if len(self.stack_token) >= 1:
            return ' '.join(self.stack_token)
        else:
            return ''

