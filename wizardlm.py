import pandas as pd
import numpy as np
from enum import Enum

import time
import torch
from transformers import pipeline


class Mutation(Enum):
    ADD_CONSTRAINTS = 1
    DEEPEN = 2
    CONCRETIZE = 3
    INCREASE_REASONING = 4
    COMPLICATE = 5
    SWITCH_TOPIC = 6


class WizardLM:
    def __init__(
            self,
            llm=None,
            seed_data=None,
            num_output_qa_pairs=10,
            context_len=2048,
    ):
        """
        Create base class

        :param llm: Method that takes a string and returns a string
        :param seed_data: Optional data to create Q:A pairs from
        :param num_output_qa_pairs: Number of desired Q:A pairs
        :param context_len: Context length, used to limit the created prompts
        """
        self.llm = llm
        self.num_output_qa_pairs = num_output_qa_pairs
        self.context_len = context_len
        self.seed_text_list = []
        self.seed_data = seed_data or [
            "What is e^(i*Pi)?",
            "What's trending in science & technology?",
            "What's the meaning of life?",
            "Why is drinking water important?",
            "What's the difference between dogs and cats?",
            "Tell me a joke.",
            "Tell me about quantum physics.",
            "Tell me about modern art.",
            "Why should I learn to play the violin?",
            "Is it better to spend money or save money?",
            "What are OKRs useful for?",
        ]
        self.prompts = []
        self.final_prompts = []
        self.prompt_templates = dict()
        self.prompt_templates['base'] = "Act like a very intelligent person."
        self.prompt_templates[Mutation.COMPLICATE] = \
            self.prompt_templates['base'] + \
"""
Rewrite #Given Prompt# to make it slightly more complicated. Create #New Prompt#.

#Given Prompt#:
<PROMPT>
"""

        self.prompt_templates[Mutation.ADD_CONSTRAINTS] = \
            self.prompt_templates['base'] + \
"""
Add a few more constraints or requirements to #Given Prompt#. Create #New Prompt#.

#Given Prompt#:
<PROMPT>
"""

        self.prompt_templates[Mutation.DEEPEN] = \
            self.prompt_templates['base'] + \
"""
Slightly increase the depth and breadth of #Given Prompt#. Create #New Prompt#.

#Given Prompt#:
<PROMPT>
"""

        self.prompt_templates[Mutation.CONCRETIZE] = \
            self.prompt_templates['base'] + \
"""
Make #Given Prompt# slightly more concrete. Create #New Prompt#.

#Given Prompt#:
<PROMPT>
"""

        self.prompt_templates[Mutation.INCREASE_REASONING] = \
            self.prompt_templates['base'] + \
"""
If #Given Prompt# can be solved with just a few simple thinking processes, rewrite it to
explicitly request multiple-step reasoning. Create #New Prompt#.

#Given Prompt#:
<PROMPT>
"""

        self.prompt_templates[Mutation.SWITCH_TOPIC] = \
            self.prompt_templates['base'] + \
"""
Rewrite #Given Prompt# by switching the topic, keeping the domain and difficulty level similar. Create #New Prompt#.

#Given Prompt#:
<PROMPT>
"""

    def run(self):
        self.create_seed_prompts()
        self.create_prompts()
        import pickle
        with open("prompts.pickle", "wb") as f:
            f.write(pickle.dumps(self.final_prompts))

    def create_seed_prompts(self):
        """
        Turn self.seed_data into a list of strings of text self.source_text_list
        Each text string can represent as little as a word, or as much as document.
        Just has to be representative of some concept or body of text.

        :return: None
        """

        assert self.seed_data, "must have seed data"
        import os
        if isinstance(self.seed_data, str) and os.path.exists(self.seed_data):
            data = pd.DataFrame(self.seed_data)
            if data.shape[1] > 1:
                print("Warning: picking only column 0")
            self.seed_text_list = data.iloc[:, 0].values.tolist()
            assert self.seed_text_list, "data import failed, got empty list"
        else:
            assert isinstance(self.seed_text_list, list)
            self.seed_text_list = self.seed_data

    def create_prompts(self):
        assert self.seed_text_list, "must have seed text list"
        self.prompts.clear()
        for i in range(self.num_output_qa_pairs):
            self.prompts.append(np.random.choice(self.seed_text_list))
        i = 0
        while self.mutate():
            print("Iteration: %d" % i)
            i += 1

    def mutate(self):
        assert len(self.prompts) == self.num_output_qa_pairs
        all_something_changed = False
        for i in range(self.num_output_qa_pairs):
            mutation = np.random.choice(Mutation)
            before = self.prompts[i]
            # print("===========================")
            # print("Before: %s" % before)
            # print("===========================")
            assert "<PROMPT>" in self.prompt_templates[mutation]
            prompt = self.prompt_templates[mutation].replace("<PROMPT>", before)
            print("===========================")
            print("Mutation: %s" % (mutation.name))
            # print("Mutation: %s\nPrompt:\n%s" % (mutation.name, prompt))
            # print("===========================")
            t0 = time.time()
            after = self.llm(prompt)
            t1 = time.time()
            print("LLM took %.4f seconds" % (t1 - t0))
            after = after.split("Prompt#:")[-1].strip()
            # print("===========================")
            print("%s" % after)
            print("===========================")

            something_changed, why = self.change_approved(before, after)
            if something_changed:
                all_something_changed = True
                self.prompts[i] = after
                print("Mutation successful")
            else:
                if why == "too long":
                    self.final_prompts.append(after)
                    print("Prompt accepted, now have %d good prompts." % len(self.final_prompts))
                    self.prompts[i] = np.random.choice(self.seed_text_list)
                    print("Creating new prompt")
                else:
                    print("Mutation rejected, will try again. Reason: %s" % why)
            print("", flush=True)
        return len(self.final_prompts) < self.num_output_qa_pairs

    def change_approved(self, before, after):
        if before == after:
            return False, "same"
        if len(after) > self.context_len / 4 / 2:  # approx. 4 bytes per token, don't use more than half of context
            return False, "too long"
        if self.prompt_templates['base'] in after:
            return False, "prompt leaked 1"
        if "#New Prompt#" in after:
            return False, "prompt leaked 2"
        if "new prompt" in after:
            return False, "prompt leaked 3"
        if "sorry" in after.lower() and "sorry" not in before.lower() and len(after) < 400:
            return False, "sorry"
        if False:
            # too slow in general, not needed
            prompt = """Are the two following prompts equal to each other?
    To be equal, they must meet two requirements:
    1. Both prompts have the same constraints and requirements.
    2. Both prompts have the same depth and breath of the inquiry.
    First prompt: %s
    Second prompt: %s
    Answer with 'Equal' or 'Not Equal'. No need to explain the reason.""" % (before, after)
            answer = self.llm(prompt)
            if 'not equal' not in answer.lower():
                return False, "equal"
        return True, "ok"


class LLM:
    def __init__(self, model, max_new_tokens=None, **kwargs):
        self.generate_text = pipeline(
            model=model,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto",
            **kwargs,
        )
        self.max_new_tokens = max_new_tokens

    def __call__(self, prompt):
        res = self.generate_text(prompt, max_new_tokens=self.max_new_tokens)
        return res[0]["generated_text"]


if __name__ == "__main__":
    seed_data = None
    wizardlm = WizardLM(
        llm=LLM(
            "junelee/wizard-vicuna-13b",
            # "h2oai/h2ogpt-oasst1-512-12b",
            # "h2oai/h2ogpt-oasst1-512-20b",
            max_new_tokens=512,
            # temperature=0.3,
            # top_k=4,
            # do_sample=True,
            # num_beams=2,
        ),
        seed_data=seed_data,
        num_output_qa_pairs=5,
        context_len=2048,
    )
    wizardlm.run()


def test_check():
    import pickle
    with open("prompts.pickle", "rb") as f:
        X = pickle.loads(f.read())
        print(X)
